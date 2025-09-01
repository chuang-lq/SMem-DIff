import random

import numpy as np
import torch


class Crop(object):
    """
    Crop randomly the image in a sample.
    Args: output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if image.shape[0] > image.shape[1]:  # h > w
            image = image.transpose(1, 0, 2)
            label = label.transpose(1, 0, 2)
        new_h, new_w = self.output_size
        top, left = sample['top'], sample['left']
        if image.shape[0] == 1080:  # DVD: (1080, 1920, 3)
            top = random.randint(0, image.shape[0] - new_h)
            left = random.randint(0, image.shape[1] - new_w)
        sample['image'] = image[top: top + new_h,
                                left: left + new_w]
        sample['label'] = label[top: top + new_h,
                                left: left + new_w]

        return sample


class Flip(object):
    """
    shape is (h,w,c)
    """

    def __call__(self, sample):
        flag_lr = sample['flip_lr']
        if flag_lr == 1:  # left-right
            sample['image'] = np.fliplr(sample['image'])
            sample['label'] = np.fliplr(sample['label'])

        # flag_ud = sample['flip_ud']
        # if flag_ud == 1:  # up-down
        #     sample['image'] = np.flipud(sample['image'])
        #     sample['label'] = np.flipud(sample['label'])

        return sample


class Rotate(object):
    """
    shape is (h,w,c)
    use vertical flip and transpose for rotation implementation
    """

    def __call__(self, sample):
        flag = sample['rotate']
        if flag == 1:
            sample['image'] = sample['image'].transpose(1, 0, 2)
            sample['label'] = sample['label'].transpose(1, 0, 2)

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[np.newaxis, :])
        label = np.ascontiguousarray(label.transpose((2, 0, 1))[np.newaxis, :])
        sample['image'] = torch.from_numpy(image).float()
        sample['label'] = torch.from_numpy(label).float()
        return sample


def flip_seq(lq, model):
    # lq: b, t, c, h, w
    output_list = []
    for k in range(0, 8):
        print(k)
        if k < 4:
            output_list.append(model(lq.rot90(k, [3, 4]).cuda()).rot90(-k, [3, 4]).detach().cpu())
        else:
            output_list.append(
                model(lq.rot90(k, [3, 4]).flip(1).cuda()).flip(1).rot90(-k, [3, 4]).detach().cpu())
    return torch.median(torch.cat(output_list, dim=0), dim=0)[0].unsqueeze(dim=0)

    # output_sum = 0.0
    # for k in range(0, 8):
    #     # print(k)
    #     if k < 4:
    #         output_sum += model(lq.rot90(k, [3, 4]).cuda()).rot90(-k, [3, 4]).detach().cpu()
    #     else:
    #         output_sum += model(lq.rot90(k, [3, 4]).flip(1).cuda()).flip(1).rot90(-k, [3, 4]).detach().cpu()
    #
    # return 0.125 * output_sum


def transpose(t, trans_idx):
    # print('transpose jt .. ', t.size())
    if trans_idx >= 4:
        t = torch.flip(t, [4])
    return torch.rot90(t, trans_idx % 4, [3, 4])


def transpose_inverse(t, trans_idx):
    # print( 'inverse transpose .. t', t.size())
    t = torch.rot90(t, 4 - trans_idx % 4, [3, 4])
    if trans_idx >= 4:
        t = torch.flip(t, [4])
    return t


def grids(lq, crop_size=256, trans_num=8):
    b, t, c, h, w = lq.size()

    # crop_size = para.crop_size
    num_row = (h - 1) // crop_size + 1
    num_col = (w - 1) // crop_size + 1

    import math
    step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
    step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


    parts = []
    idxes = []

    # cnt_idx = 0

    i = 0  # 0~h-1
    last_i = False
    while i < h and not last_i:
        j = 0
        if i + crop_size >= h:
            i = h - crop_size
            last_i = True

        last_j = False
        while j < w and not last_j:
            if j + crop_size >= w:
                j = w - crop_size
                last_j = True
            # from i, j to i+crop_szie, j + crop_size
            # print(' trans 8')
            for trans_idx in range(trans_num):
                parts.append(transpose(lq[:, :, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                # cnt_idx += 1
            # parts.append(lq[:, :, :, i:i + crop_size, j:j + crop_size])
            # idxes.append({'i': i, 'j': j})
            j = j + step_j
        i = i + step_i

    # if para.get('random_crop_num', 0) > 0:
    #     for _ in range(para.get('random_crop_num')):
    #         import random
    #         i = random.randint(0, h - crop_size)
    #         j = random.randint(0, w - crop_size)
    #         trans_idx = random.randint(0, trans_num - 1)
    #         parts.append(transpose(lq[:, :, :, i:i + crop_size, j:j + crop_size], trans_idx))
    #         idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})

    crop_lq = torch.stack(parts, dim=1)  # b, n, t, c, crop_size, crop_size
    crop_lq = crop_lq.view(b, -1, c, crop_size, crop_size)  # b, n*t, c, crop_size, crop_size
    idxes = idxes
    return crop_lq, idxes


def grids_inverse(lq, output, idxes, crop_size=256):
    b, t, c, h, w = lq.size()
    preds = torch.zeros(lq.size()).to(lq.device)

    # output: b, n*t, c, crop_size, crop_size
    output = output.view(b, -1, t, c, crop_size, crop_size)  # b, n, t, c, crop_size, crop_size
    count_mt = torch.zeros((1, t, 1, h, w)).to(lq.device)
    # crop_size = para.get('crop_size')

    for cnt, each_idx in enumerate(idxes):
        i = each_idx['i']
        j = each_idx['j']
        trans_idx = each_idx['trans_idx']
        preds[:, :, :, i:i + crop_size, j:j + crop_size] += transpose_inverse(
            output[:, cnt], trans_idx)
        # preds[:, :, :, i:i + crop_size, j:j + crop_size] += output[:, cnt]
        count_mt[0, :, 0, i:i + crop_size, j:j + crop_size] += 1.

    output = preds / count_mt
    return output


def local_partition(x, patch_size=256):
    dh = dw = patch_size
    h_step = w_step = patch_size - patch_size // 4
    # h_step = w_step = patch_size // 2
    b, t, c, h, w = x.size()

    local_x = []
    for i in range(0, h + h_step - dh, h_step):
        top = i
        down = i + dh
        if down > h:
            top = h - dh
            down = h
        for j in range(0, w + w_step - dw, w_step):
            left = j
            right = j + dw
            if right > w:
                left = w - dw
                right = w
            local_x.append(x[:, :, :, top:down, left:right])
    local_x = torch.stack(local_x, dim=1)  # b n t c dh dw
    local_x = local_x.view(b, -1, c, patch_size, patch_size)  # b, n*t, c, dh, dw
    return local_x


def local_reverse(x, local_out, patch_size=256):
    dh = dw = patch_size
    h_step = w_step = patch_size - patch_size // 4
    # h_step = w_step = patch_size // 2
    b, t, c, h, w = x.size()
    output = torch.zeros_like(x).to(x.device)

    # local_out: b, n*t, c, dh, dw
    local_out = local_out.view(b, -1, t, c, dh, dw)  # b n t c dh dw
    count = torch.zeros((1, t, 1, h, w), device=x.device)

    index = 0
    for i in range(0, h + h_step - dh, h_step):
        top = i
        down = i + dh
        if down > h:
            top = h - dh
            down = h
        for j in range(0, w + w_step - dw, w_step):
            left = j
            right = j + dw
            if right > w:
                left = w - dw
                right = w
            output[:, :, :, top:down, left:right] += local_out[:, index]  # local_out: b n t c dh dw
            count[0, :, 0, top:down, left:right] += 1
            index += 1
    output = output / count
    return output


def normalize(x, centralize=False, normalize=False, val_range=255.0):
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range

    return x


def normalize_reverse(x, centralize=False, normalize=False, val_range=255.0):
    if normalize:
        x = x * val_range
    if centralize:
        x = x + val_range / 2

    return x


if __name__ == '__main__':
    x = torch.randn(2, 7, 3, 720, 1280)  # b, t, c, h, w
    print('x: ', x.shape)
    outs = []
    crop_lq, idxes = grids(x)
    print('crop_lq: ', crop_lq.shape)
    i = 0
    n = crop_lq.size(1)
    while i < n:
        j = i + 7
        if j >= n:
            j = n
        outs.append(crop_lq[:, i:j])
        i = j
    outs = torch.cat(outs, dim=1)
    print('outs: ', outs.shape)
    print(outs.equal(crop_lq))
    y = grids_inverse(x, outs, idxes)
    print('y: ', y.shape)

    local_outs = []
    local_x = local_partition(x)
    print('local_x: ', local_x.shape)
    i = 0
    n = local_x.size(1)
    while i < n:
        j = i + 7
        if j >= n:
            j = n
        local_outs.append(local_x[:, i:j])
        i = j
    x_outs = torch.cat(local_outs, dim=1)
    print('local_outs: ', x_outs.shape)
    print(x_outs.equal(local_x))
    y_ = local_reverse(x, x_outs)
    print('y_: ', y_.shape)
