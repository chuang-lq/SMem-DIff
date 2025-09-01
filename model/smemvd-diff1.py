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
        self.mem_every = para.mem_every
        self.num_blocks_forward = para.num_blocks_forward
        self.num_blocks_backward = para.num_blocks_backward

        # ----------------- Diffusion branch -----------------
        self.net_le = latent_encoder_gelu(in_chans=6, embed_dim=self.mid_channels, block_num=6, stage=1,
                                          group=4, patch_expansion=0.5, channel_expansion=4)

        # ----------------- Deblurring branch -----------------
        self.n_feats = 16

        self.downsampling = nn.Sequential(
            conv3x3(3, self.n_feats, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.n_feats, self.mid_channels, kernel_size=3, stride=2, padding=1)  # b, 64, h/2, w/2
        )

        self.wave = HaarDownsampling(self.mid_channels)

        # SA transformer
        transformer_scale4 = []
        for _ in range(3):
            transformer_scale4.append(
                nn.Sequential(
                    CWGDN(dim=self.mid_channels, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'))
            )
        self.transformer_scale4 = nn.Sequential(*transformer_scale4)

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(self.mid_channels, self.n_feats, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv5x5(self.n_feats, 3, stride=1)
        )

        self.encoder = Encoder(dim=self.mid_channels, num_frames_tocache=3)
        self.decoder = Decoder(dim=self.mid_channels, num_frames_tocache=3)

        # Feature fusion Module
        self.feat_fusion = DTFF1(self.mid_channels)

        self.forward_backbone = ResidualBlocksWithInputConv(self.mid_channels, self.num_blocks_forward)
        self.backward_backbone = ResidualBlocksWithInputConv(self.mid_channels, self.num_blocks_backward)

        self.tfr = TFR(n_feat=self.mid_channels, kernel_size=3, reduction=4, act=nn.PReLU(),
                       bias=False, scale_unetfeats=int(self.mid_channels / 2), num_cab=5)

    def forward(self, inputs, gts=None, noise=None, profile_flag=False):
        if profile_flag:
            return self.profile_forward(inputs)

        b, t, c, h, w = inputs.size()

        prior_zs = []
        for i in range(0, t):
            prior_z = self.net_le(inputs[:, i], gts[:, i])  # b, n, c
            prior_zs.append(prior_z)

        outputs = []

        # feature extraction
        down_feats = rearrange(self.downsampling(rearrange(inputs, 'b t c h w -> (b t) c h w')),
                               '(b t) c h w -> b t c h w', b=b)
        # tf_down_feats = rearrange(down_feats, 'b c t h w -> (b t) c h w')
        # tf_wave_l, tf_wave_h = self.wave(tf_down_feats)
        # tf_wave_l = rearrange(self.transformer_scale4(rearrange(tf_wave_l, '(b t) c h w -> b t c h w', b=b)),
        #                        'b t c h w -> (b t) c h w')
        # down_feats = self.wave(torch.cat([tf_wave_l, tf_wave_h], dim=1), rev=True)
        # down_feats = rearrange(down_feats, '(b t) c h w -> b t c h w')
        # down_feats = self.transformer_scale4(down_feats)  # b, t, c, h, w

        prev_feat = None
        k_cached = [None] * 8
        v_cached = [None] * 8
        encoder_outs_list = []
        decoder_outs_list = []

        # ------------ backward ------------
        for i in range(t - 1, -1, -1):
            down_feat = down_feats[:, i]
            encoder_outs, latent, diff_prior, k_to_cache, v_to_cache = self.encoder(down_feat, prior_zs[i],
                                                                                    k_cached=k_cached,
                                                                                    v_cached=v_cached)
            for k in range(0, len(k_to_cache)):
                k_cached[k] = k_to_cache[k]
                v_cached[k] = v_to_cache[k]

            decoder_outs, k_to_cache, v_to_cache = self.decoder(encoder_outs, latent, diff_prior,
                                                                k_cached=k_cached, v_cached=v_cached)
            for k in range(0, len(k_to_cache)):
                k_cached[5 + k] = k_to_cache[k]
                v_cached[5 + k] = v_to_cache[k]

            if i == t-1:
                feat = decoder_outs[0]
            else:
                down_feat = self.feat_fusion(down_feat, down_feats[:, i + 1])
                feat = self.feat_fusion(decoder_outs[0], prev_feat)

            feat = self.backward_backbone(down_feat, feat)
            decoder_outs[0] = feat

            encoder_outs_list.append(encoder_outs)
            decoder_outs_list.append(decoder_outs)

            prev_feat = feat

        encoder_outs_list = list(reversed(encoder_outs_list))
        decoder_outs_list = list(reversed(decoder_outs_list))

        # ----------- forward --------------
        encoder_forward_list = []
        decoder_forward_list = []
        for i in range(0, t):
            input_curr = inputs[:, i]
            down_feat = down_feats[:, i]

            encoder_outs, latent, diff_prior, k_to_cache, v_to_cache = self.encoder(down_feat, prior_zs[i],
                                                                                    k_cached=k_cached,
                                                                                    v_cached=v_cached)
            for k in range(0, len(k_to_cache)):
                k_cached[k] = k_to_cache[k]
                v_cached[k] = v_to_cache[k]

            decoder_outs, k_to_cache, v_to_cache = self.decoder(encoder_outs, latent, diff_prior,
                                                                k_cached=k_cached, v_cached=v_cached)
            for k in range(0, len(k_to_cache)):
                k_cached[5 + k] = k_to_cache[k]
                v_cached[5 + k] = v_to_cache[k]

            if i == 0:
                feat = decoder_outs[0]
            else:
                down_feat = self.feat_fusion(down_feat, down_feats[:, i-1])
                feat = self.feat_fusion(decoder_outs[0], prev_feat)

            feat = self.forward_backbone(down_feat, feat)
            decoder_outs[0] = feat

            encoder_forward_list.append(encoder_outs)
            decoder_forward_list.append(decoder_outs)

            for j in range(3):
                encoder_outs_list[i][j] = encoder_outs_list[i][j] + encoder_forward_list[i][j]
                decoder_outs_list[i][j] = decoder_outs_list[i][j] + decoder_forward_list[i][j]

            # TFR
            out = self.tfr(down_feats[:, i], encoder_outs_list[i], decoder_outs_list[i])
            out = self.upsampling(out)
            out += input_curr
            outputs.append(out)

            prev_feat = feat

        results = torch.stack(outputs, dim=1)

        return results

    def profile_forward(self, inputs):
        return self.forward(inputs)

    def check_image_size(self, x):
        _, _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, dim=64, num_blocks=30):
        super().__init__()

        # self.cga = CGAFusion(dim)

        main = []
        main.append(nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=dim))

        self.main = nn.Sequential(*main)

    def forward(self, x, feat):
        feat = self.main(torch.cat([x, feat], dim=1))

        # feat = self.cga(x, feat)
        # feat = self.main(feat)

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

    x = torch.randn(1, 3, 3, 256, 256).cuda()
    y = torch.randn(1, 3, 3, 256, 256).cuda()
    model = Model(config).cuda()

    x = model(x, y)
    print(x.shape)





