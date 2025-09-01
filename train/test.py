import os
import pickle
import time
from os.path import join, dirname

import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn

from data.utils import *
from model import Model
from .metrics import psnr_calculate, ssim_calculate
from .utils import AverageMeter, img2video


def test(config, logger):
    # load model with checkpoint
    if not config.test_only:
        config.test_checkpoint = join(logger.save_dir, 'model_best.pth.tar')
    if config.test_save_dir is None:
        config.test_save_dir = logger.save_dir
    model = Model(config).cuda()
    model = nn.DataParallel(model)

    checkpoint_path = config.test_checkpoint
    # map_location=torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    logger('Load %s as the checkpoint' % (checkpoint_path))

    model.load_state_dict(checkpoint['state_dict'])

    ds_name = config.dataset
    logger('{} results generating ...'.format(ds_name), prefix='\n')

    if ds_name == 'BSD' or ds_name == 'DVD':
        ds_type = 'valid'
        _test_torch(config, logger, model, ds_type)
    elif ds_name == 'gopro_ori' or ds_name == 'gopro_ds':
        ds_type = 'valid'
        _test_lmdb(config, logger, model, ds_type)
    else:
        raise NotImplementedError


def _test_torch(para, logger, model, ds_type):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()

    val_range = 2.0 ** 8 - 1 if para.data_format == 'RGB' else 2.0 ** 16 - 1
    if para.dataset == 'BSD':
        H, W = 480, 640
        dataset_path = join(para.data_root, para.dataset, '{}_{}'.format(para.dataset, para.ds_config), ds_type)
        seq_length = 150
    else:
        H, W = 720, 1280
        dataset_path = join(para.data_root, para.dataset, ds_type)
        seq_length = 100
    seqs = sorted(os.listdir(dataset_path))

    if not hasattr(para, 'past_frames'):
        para.past_frames = 0
    if not hasattr(para, 'future_frames'):
        para.future_frames = 0

    for seq in seqs:
        logger('seq {} image results generating ...'.format(seq))
        dir_name = '_'.join((para.dataset, para.model, 'valid'))
        save_dir = join(para.test_save_dir, dir_name, seq)
        os.makedirs(save_dir, exist_ok=True)
        suffix = 'png' if para.data_format == 'RGB' else 'tiff'
        start = 0
        end = para.test_frames

        while True:
            input_seq = []
            label_seq = []
            for frame_idx in range(start, end):
                if para.dataset == 'DVD':
                    blur_img_path = join(dataset_path, seq, 'input', '{:05d}.jpg'.format(frame_idx))
                    sharp_img_path = join(dataset_path, seq, 'GT', '{:05d}.jpg'.format(frame_idx))
                else:
                    blur_img_path = join(dataset_path, seq, 'Blur', para.data_format,
                                         '{:08d}.{}'.format(frame_idx, suffix))
                    sharp_img_path = join(dataset_path, seq, 'Sharp', para.data_format,
                                          '{:08d}.{}'.format(frame_idx, suffix))
                if para.data_format == 'RGB':
                    blur_img = cv2.imread(blur_img_path).transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
                    gt_img = cv2.imread(sharp_img_path)
                else:
                    blur_img = cv2.imread(blur_img_path, -1)[..., np.newaxis].astype(np.int32)
                    blur_img = blur_img.transpose(2, 0, 1)[np.newaxis, ...]
                    gt_img = cv2.imread(sharp_img_path, -1).astype(np.uint16)
                input_seq.append(blur_img)
                label_seq.append(gt_img)

            input_seq = np.concatenate(input_seq)[np.newaxis, :]  # (1, T, 3, H, W)
            model.eval()
            logger('Shape of the current input sequence: %s' % (str(input_seq.shape)))
            with torch.no_grad():
                input_seq = normalize(torch.from_numpy(input_seq).float().cuda(), centralize=para.centralize,
                                      normalize=para.normalize, val_range=val_range)
                time_start = time.time()

                # output_seq = model([input_seq, None])
                outs = []
                # crop_lq, idxes = grids(input_seq)
                crop_lq = local_partition(input_seq)
                i = 0
                n = crop_lq.size(1)
                while i < n:
                    j = i + para.test_frames
                    if j >= n:
                        j = n
                    output_seq = model([crop_lq[:, i:j], None])
                    if isinstance(output_seq, (list, tuple)):
                        output_seq = output_seq[0]
                    outs.append(output_seq)
                    i = j
                outs = torch.cat(outs, dim=1)
                # output_seq = grids_inverse(input_seq, outs, idxes)
                output_seq = local_reverse(input_seq, outs)

                output_seq = output_seq.squeeze(dim=0)
                timer.update((time.time() - time_start) / len(output_seq), n=len(output_seq))

            logger('Evaluating PSNR and SSIM...')
            for frame_idx in range(para.past_frames, end - start - para.future_frames):
                blur_img = input_seq.squeeze(dim=0)[frame_idx]
                blur_img = normalize_reverse(blur_img, centralize=para.centralize, normalize=para.normalize,
                                             val_range=val_range)
                blur_img = blur_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
                blur_img = blur_img.astype(np.uint8) if para.data_format == 'RGB' else blur_img.astype(np.uint16)
                blur_img_path = join(save_dir, '{:08d}_input.{}'.format(frame_idx + start, suffix))
                gt_img = label_seq[frame_idx]
                gt_img_path = join(save_dir, '{:08d}_gt.{}'.format(frame_idx + start, suffix))
                deblur_img = output_seq[frame_idx - para.past_frames]
                deblur_img = normalize_reverse(deblur_img, centralize=para.centralize, normalize=para.normalize,
                                               val_range=val_range)
                deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
                deblur_img = np.clip(deblur_img, 0, val_range)
                deblur_img = deblur_img.astype(np.uint8) if para.data_format == 'RGB' else deblur_img.astype(np.uint16)
                deblur_img_path = join(save_dir, '{:08d}_{}.{}'.format(frame_idx + start, para.model.lower(), suffix))
                if para.test_save_img is True:
                    cv2.imwrite(blur_img_path, blur_img)
                    cv2.imwrite(gt_img_path, gt_img)
                    cv2.imwrite(deblur_img_path, deblur_img)
                if deblur_img_path not in results_register:
                    results_register.add(deblur_img_path)
                    PSNR.update(psnr_calculate(deblur_img, gt_img))
                    SSIM.update(ssim_calculate(deblur_img, gt_img))

            if end == seq_length:
                break
            else:
                start = end - para.future_frames - para.past_frames
                end = start + para.test_frames
                if end > seq_length:
                    end = seq_length
                    start = end - para.test_frames
            logger('Finish writing the image to the file...')

        if para.video:
            if para.data_format != 'RGB':
                continue
            logger('seq {} video result generating ...'.format(seq))
            marks = ['Input', para.model, 'GT']
            path = dirname(save_dir)
            frame_start = para.past_frames
            frame_end = seq_length - para.future_frames
            img2video(path=path, size=(3 * W, 1 * H), seq=seq, frame_start=frame_start, frame_end=frame_end,
                      marks=marks, fps=10)

    logger('Test images : {}'.format(PSNR.count), prefix='\n')
    logger('Test PSNR : {}'.format(PSNR.avg))
    logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Average time per image: {}'.format(timer.avg))


def _test_lmdb(config, logger, model, ds_type):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()
    if config.dataset == 'gopro_ds':
        H, W, C = 540, 960, 3
    elif config.dataset == 'gopro_ori':
        H, W, C = 720, 1280, 3
    else:
        raise ValueError

    data_test_path = join(config.data_root, config.dataset, '{}_{}'.format(config.dataset, ds_type))
    data_test_gt_path = join(config.data_root, config.dataset, '{}_{}_gt'.format(config.dataset, ds_type))
    data_test_info_path = join(config.data_root, config.dataset, '{}_info_{}.pkl'.format(config.dataset, ds_type))

    if not hasattr(config, 'past_frames'):
        config.past_frames = 0
    if not hasattr(config, 'future_frames'):
        config.future_frames = 0

    env_blur = lmdb.open(data_test_path, map_size=1073741824)  # 1099511627776 / 1073741824
    env_gt = lmdb.open(data_test_gt_path, map_size=1073741824)  # 1099511627776(1T) / 1073741824(1G)
    txn_blur = env_blur.begin()
    txn_gt = env_gt.begin()

    with open(data_test_info_path, 'rb') as f:
        seqs_info = pickle.load(f)
    for seq_idx in range(seqs_info['num']):
        seq_length = seqs_info[seq_idx]['length']
        seq = '{:03d}'.format(seq_idx)
        logger('Start generating {}-th sequence results...'.format(seq))
        dir_name = '_'.join((config.dataset, config.model, 'test'))
        save_dir = join(config.test_save_dir, dir_name, seq)
        os.makedirs(save_dir, exist_ok=True)
        start = 0
        end = seq_length if config.test_frames == -1 else config.test_frames

        while (True):
            input_seq = []
            label_seq = []
            for frame_idx in range(start, end):
                code = '%03d_%08d' % (seq_idx, frame_idx)
                code = code.encode()
                blur_img = txn_blur.get(code)
                blur_img = np.frombuffer(blur_img, dtype='uint8')
                blur_img = blur_img.reshape(H, W, C).transpose((2, 0, 1))[np.newaxis, :]  # (1, 3, H, W)
                gt_img = txn_gt.get(code)
                gt_img = np.frombuffer(gt_img, dtype='uint8')
                gt_img = gt_img.reshape(H, W, C)
                input_seq.append(blur_img)
                label_seq.append(gt_img)

            input_seq = np.concatenate(input_seq)[np.newaxis, :]  # (1, T, 3, H, W)
            model.eval()
            logger('Shape of the current input sequence: %s'%(str(input_seq.shape)))
            with torch.no_grad():
                input_seq = normalize(torch.from_numpy(input_seq).float().cuda(), centralize=config.centralize,
                                      normalize=config.normalize)
                time_start = time.time()

                # output_seq = model([input_seq, None])
                outs = []
                # crop_lq, idxes = grids(input_seq)
                crop_lq = local_partition(input_seq)
                i = 0
                n = crop_lq.size(1)
                while i < n:
                    j = i + config.test_frames
                    if j >= n:
                        j = n
                    output_seq = model([crop_lq[:, i:j], None])
                    if isinstance(output_seq, (list, tuple)):
                        output_seq = output_seq[0]
                    outs.append(output_seq)
                    i = j
                outs = torch.cat(outs, dim=1)
                # output_seq = grids_inverse(input_seq, outs, idxes)
                output_seq = local_reverse(input_seq, outs)

                output_seq = output_seq.squeeze(dim=0)
                timer.update((time.time() - time_start) / len(output_seq), n=len(output_seq))

            logger('Evaluating PSNR and SSIM...')
            for frame_idx in range(config.past_frames, end - start - config.future_frames):
                blur_img = input_seq.squeeze()[frame_idx]
                blur_img = normalize_reverse(blur_img, centralize=config.centralize, normalize=config.normalize)
                blur_img = blur_img.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
                blur_img_path = join(save_dir, '{:08d}_input.png'.format(frame_idx + start))
                gt_img = label_seq[frame_idx]
                gt_img_path = join(save_dir, '{:08d}_gt.png'.format(frame_idx + start))
                deblur_img = output_seq[frame_idx - config.past_frames]
                deblur_img = normalize_reverse(deblur_img, centralize=config.centralize, normalize=config.normalize)
                deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0))
                deblur_img = np.clip(deblur_img, 0, 255).astype(np.uint8)
                deblur_img_path = join(save_dir, '{:08d}_{}.png'.format(frame_idx + start, config.model.lower()))
                if config.test_save_img is True:
                    cv2.imwrite(blur_img_path, blur_img)
                    cv2.imwrite(gt_img_path, gt_img)
                    cv2.imwrite(deblur_img_path, deblur_img)
                if deblur_img_path not in results_register:
                    results_register.add(deblur_img_path)
                    psnr = psnr_calculate(deblur_img, gt_img)
                    ssim = ssim_calculate(deblur_img, gt_img)
                    PSNR.update(psnr)
                    SSIM.update(ssim)
            
            if end == seq_length:
                break
            else:
                start = end - config.future_frames - config.past_frames
                end = start + config.test_frames
                if end > seq_length:
                    end = seq_length
                    start = end - config.test_frames
            logger('Finish writing the image to the file...')

        if config.video:
            logger('Generate seq {} video result...'.format(seq))
            marks = ['Input', config.model, 'GT']
            path = dirname(save_dir)
            frame_start = config.past_frames
            frame_end = seq_length - config.future_frames
            img2video(path=path, size=(3 * W, 1 * H), seq=seq, frame_start=frame_start, frame_end=frame_end,
                      marks=marks, fps=10)

    logger('Test images : {}'.format(PSNR.count), prefix='\n')
    logger('Test PSNR : {}'.format(PSNR.avg))
    logger('Test SSIM : {}'.format(SSIM.avg))
    logger('Average time per image: {}'.format(timer.avg))
