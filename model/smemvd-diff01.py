import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
from einops import rearrange
from functools import partial
from .modules import *

import argparse
import yaml
from types import SimpleNamespace


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()

        self.mid_channels = para.mid_channels
        self.num_blocks_forward = para.num_blocks_forward
        self.num_blocks_backward = para.num_blocks_backward

        # ----------------- Diffusion branch -----------------
        self.net_le = LE_arch(n_feats=64)

        # ----------------- Deblurring branch -----------------
        self.feat_extractor = nn.Conv3d(3, self.mid_channels, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.recons = nn.Conv3d(self.mid_channels, 3, (1, 3, 3), 1, (0, 1, 1), bias=True)

        # wave tf
        # self.wave = HaarDownsampling(self.mid_channels)
        self.dwt = DWT(fuseh=True)
        self.idwt = IDWT()
        self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self.create_wave_conv()
        self.x_wave_1_conv1 = nn.Conv2d(self.mid_channels * 3, self.mid_channels * 3, 1, 1, 0, groups=3)
        self.x_wave_1_conv2 = nn.Conv2d(self.mid_channels * 3, self.mid_channels * 3, 1, 1, 0, groups=3)
        # wave pro
        self.x_wave_2_conv1 = nn.Conv2d(self.mid_channels * 3, self.mid_channels * 3, 1, 1, 0, groups=3)
        self.x_wave_2_conv2 = nn.Conv2d(self.mid_channels * 3, self.mid_channels * 3, 1, 1, 0, groups=3)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.encoder = Encoder(dim=self.mid_channels, num_frames_tocache=3)
        self.decoder = Decoder(dim=self.mid_channels, num_frames_tocache=3)

        # Feature fusion Module
        self.featback_fusion = DTFF1(self.mid_channels)
        self.featward_fusion = DTFF1(self.mid_channels)

        self.forward_backbone = ResidualBlocksWithInputConv(self.mid_channels, self.num_blocks_forward)
        self.backward_backbone = ResidualBlocksWithInputConv(self.mid_channels, self.num_blocks_backward)

    def forward(self, inputs, gts=None, profile_flag=False):
        if profile_flag:
            return self.profile_forward(inputs)

        b, t, c, h, w = inputs.size()

        prior_zs = self.net_le(rearrange(inputs, 'b t c h w -> (b t) c h w'),
                               rearrange(gts, 'b t c h w -> (b t) c h w'))
        prior_zs = rearrange(prior_zs, '(b t) c-> b t c', b=b)

        inputs = self.check_image_size(inputs)

        # feature extraction
        in_feats = self.feat_extractor(rearrange(inputs, 'b t c h w -> b c t h w'))

        tf_in_feats = rearrange(in_feats, 'b c t h w -> (b t) c h w')

        # tf_wave1_l, tf_wave1_h = self.wave(tf_in_feats)
        # tf_wave1_h = self.x_wave_1_conv2(self.lrelu(self.x_wave_1_conv1(tf_wave1_h)))
        # tf_wave2_l, tf_wave2_h = self.wave(tf_wave1_l)
        # tf_wave2_h = self.x_wave_2_conv2(self.lrelu(self.x_wave_2_conv1(tf_wave2_h)))

        tf_wave1_l, tf_wave1_h = self.dwt(tf_in_feats)
        tf_wave1_h = self.lrelu(self.x_wave_1_conv1(tf_wave1_h))
        tf_wave2_l, tf_wave2_h = self.dwt(tf_wave1_l)
        tf_wave2_h = self.lrelu(self.x_wave_2_conv1(tf_wave2_h))

        k_cached = [None] * 7
        v_cached = [None] * 7
        feats_list = []

        tf_wave2_l = rearrange(tf_wave2_l, '(b t) c h w -> b t c h w', b=b)

        for i in range(0, t):
            down_tf2 = tf_wave2_l[:, i]

            encoder_outs, k_to_cache, v_to_cache = self.encoder(down_tf2, prior_zs[:, i],
                                                                k_cached=k_cached, v_cached=v_cached)
            for k in range(0, len(k_to_cache)):
                k_cached[k] = k_to_cache[k]
                v_cached[k] = v_to_cache[k]

            decoder_out, k_to_cache, v_to_cache = self.decoder(encoder_outs, prior_zs[:, i],
                                                               k_cached=k_cached, v_cached=v_cached)
            for k in range(0, len(k_to_cache)):
                k_cached[4 + k] = k_to_cache[k]
                v_cached[4 + k] = v_to_cache[k]

            feats_list.append(decoder_out)

        tf_wave2_l = torch.stack(feats_list, dim=1)
        tf_wave2_l = rearrange(tf_wave2_l, 'b t c h w -> (b t) c h w')

        wave2_hl, wave2_lh, wave2_hh = (tf_wave2_h[:, :self.mid_channels, ...],
                                        tf_wave2_h[:, self.mid_channels:self.mid_channels * 2, ...],
                                        tf_wave2_h[:, 2 * self.mid_channels:, ...])
        wave2_hl = self.horizontal_conv(tf_wave2_l) + wave2_hl
        wave2_lh = self.vertical_conv(tf_wave2_l) + wave2_lh
        wave2_hh = self.diagonal_conv(tf_wave2_l) + wave2_hh
        tf_wave2_h = torch.cat([wave2_hl, wave2_lh, wave2_hh], dim=1)
        tf_wave2_h = self.x_wave_2_conv2(tf_wave2_h)

        down_tfs1 = self.idwt(torch.cat([tf_wave2_l, tf_wave2_h], dim=1))
        down_tfs1 = rearrange(down_tfs1, '(b t) c h w -> b t c h w', b=b)
        tf_wave1_l = rearrange(tf_wave1_l, '(b t) c h w -> b t c h w', b=b)

        for i in range(t - 1, -1, -1):
            if i == t - 1:
                down_tf1 = tf_wave1_l[:, i]
                feat = down_tfs1[:, i]
            else:
                down_tf1 = self.featback_fusion(tf_wave1_l[:, i], tf_wave1_l[:, i + 1])
                feat = self.featback_fusion(down_tfs1[:, i], feats_list[i + 1])

            feat = self.forward_backbone(down_tf1, feat)
            feats_list[i] = feat

        for i in range(0, t):
            if i == 0:
                down_tf1 = tf_wave1_l[:, i]
                feat = feats_list[i]
            else:
                down_tf1 = self.featward_fusion(tf_wave1_l[:, i], tf_wave1_l[:, i - 1])
                feat = self.featward_fusion(feats_list[i], feats_list[i - 1])

            feat = self.forward_backbone(down_tf1, feat)
            feats_list[i] = feat

        tf_wave1_l = torch.stack(feats_list, dim=1)
        tf_wave1_l = rearrange(tf_wave1_l, 'b t c h w -> (b t) c h w')

        wave1_hl, wave1_lh, wave1_hh = (tf_wave1_h[:, :self.mid_channels, ...],
                                        tf_wave1_h[:, self.mid_channels:self.mid_channels * 2, ...],
                                        tf_wave1_h[:, 2 * self.mid_channels:, ...])
        wave1_hl = self.horizontal_conv(tf_wave1_l) + wave1_hl
        wave1_lh = self.vertical_conv(tf_wave1_l) + wave1_lh
        wave1_hh = self.diagonal_conv(tf_wave1_l) + wave1_hh
        tf_wave1_h = torch.cat([wave1_hl, wave1_lh, wave1_hh], dim=1)
        tf_wave1_h = self.x_wave_1_conv2(tf_wave1_h)

        outs = rearrange(self.idwt(torch.cat([tf_wave1_l, tf_wave1_h], dim=1)), '(b t) c h w -> b t c h w',
                         b=b)
        outs = rearrange(self.recons(rearrange(outs, 'b t c h w -> b c t h w')), 'b c t h w -> b t c h w')

        outs = outs.contiguous() + inputs
        outs = outs[:, :, :, :h, :w]

        return outs

    def profile_forward(self, inputs):
        return self.forward(inputs)

    def create_conv_layer(self, kernel):
        conv = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, padding=1,
                         bias=False)
        conv.weight.data = kernel.repeat(self.mid_channels, self.mid_channels, 1, 1)
        return conv

    def create_wave_conv(self):
        horizontal_kernel = torch.tensor([[1, 0, -1],
                                          [1, 0, -1],
                                          [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        vertical_kernel = torch.tensor([[1, 1, 1],
                                        [0, 0, 0],
                                        [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        diagonal_kernel = torch.tensor([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        horizontal_conv = self.create_conv_layer(horizontal_kernel)
        vertical_conv = self.create_conv_layer(vertical_kernel)
        diagonal_conv = self.create_conv_layer(diagonal_kernel)
        return horizontal_conv, vertical_conv, diagonal_conv

    def check_image_size(self, x, padder_size=32):
        _, _, _, h, w = x.size()
        mod_pad_h = (padder_size - h % padder_size) % padder_size
        mod_pad_w = (padder_size - w % padder_size) % padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, dim=64, num_blocks=30):
        super().__init__()

        self.cga = CGAFusion(dim)

        main = []
        # main.append(nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=dim))

        self.main = nn.Sequential(*main)

    def forward(self, x, feat):
        # feat = self.main(torch.cat([x, feat], dim=1))

        feat = self.cga(x, feat)
        feat = self.main(feat)

        return feat


def cost_profile(model, H, W, seq_length=6):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    y = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, y, profile_flag), verbose=False)

    return flops / seq_length, params


def feed(model, iter_samples):
    inputs, labels = iter_samples
    outputs = model(inputs, labels)
    return outputs


if __name__ == '__main__':
    with open("../config/rmemvd_gopro.yml", mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)

    net = Model(config).cuda()
    input_dim = torch.randn(1, 10, 3, 720, 1280).cuda()
    gt = torch.randn(1, 10, 3, 720, 1280).cuda()
    flops, params = profile(model=net, inputs=(input_dim, gt))
    flops = flops / 10
    print(flops / 10 ** 9, params / 10 ** 6)

    # x = torch.randn(1, 5, 3, 256, 256).cuda()
    # y = torch.randn(1, 5, 3, 256, 256).cuda()
    #
    # x = net(x, y)
    # print(x.shape)
