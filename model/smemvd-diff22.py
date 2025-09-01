import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
from einops import rearrange
from functools import partial
from .modules import *
from .ldm import *

# import argparse
# import yaml
# from types import SimpleNamespace


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()

        self.mid_channels = para.mid_channels
        self.mem_every = para.mem_every
        self.num_blocks_forward = para.num_blocks_forward
        self.num_blocks_backward = para.num_blocks_backward
        self.diffusion_schedule = para.diffusion_schedule

        # ----------------- Diffusion branch -----------------
        self.net_le = latent_encoder_gelu(in_chans=6, embed_dim=self.mid_channels, block_num=6, stage=1,
                                          group=4, patch_expansion=0.5, channel_expansion=4)

        self.net_le_dm = latent_encoder_gelu(in_chans=3, embed_dim=self.mid_channels, block_num=6, stage=2,
                                             group=4, patch_expansion=0.5, channel_expansion=4)

        self.net_d = denoising(in_channel=256, out_channel=256, inner_channel=512, block_num=4,
                               group=4, patch_expansion=0.5, channel_expansion=2)

        # apply LDM implementation
        self.diffusion = DDPM(denoise=self.net_d,
                              condition=self.net_le_dm,
                              n_feats=self.mid_channels,
                              group=4,
                              linear_start=self.diffusion_schedule['linear_start'],
                              linear_end=self.diffusion_schedule['linear_end'],
                              timesteps=self.diffusion_schedule['timesteps'])

        # ----------------- Deblurring branch -----------------
        self.n_feats = 16

        self.downsampling = nn.Sequential(
            conv3x3(3, self.n_feats, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.n_feats, self.mid_channels, kernel_size=3, stride=2, padding=1)  # b, 64, h/2, w/2
        )

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
            # Converse2D(self.n_feats, self.n_feats, 3, 2, 2, 'replicate', 1e-5),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv5x5(self.n_feats, 3, stride=1)
        )

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

        prior_zs = []
        priors = []

        if self.training:
            for i in range(0, t):
                prior_z = self.net_le(inputs[:, i], gts[:, i])  # b, n, c
                prior, _ = self.diffusion(inputs[:, i], prior_z)

                prior_zs.append(prior_z)
                priors.append(prior)
        else:
            for i in range(0, t):
                if gts is not None:
                    prior_z = self.net_le(inputs[:, i], gts[:, i])
                    prior_zs.append(prior_z)

                prior = self.diffusion(inputs[:, i])
                priors.append(prior)

        outputs = []

        # feature extraction
        down_feats = rearrange(self.downsampling(rearrange(inputs, 'b t c h w -> (b t) c h w')),
                               '(b t) c h w -> b t c h w', b=b)
        down_feats = self.transformer_scale4(down_feats)  # b, t, c, h, w

        k_cached = [None] * 8
        v_cached = [None] * 8
        feats_list = []

        for i in range(t):
            down_feat = down_feats[:, i]

            encoder_outs, latent, diff_prior, k_to_cache, v_to_cache = self.encoder(down_feat, priors[i],
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

            feats_list.append(decoder_outs[0])

        for i in range(t - 1, -1, -1):
            if i == t - 1:
                down_feat = down_feats[:, i]
                feat = feats_list[i]
            else:
                down_feat = self.featback_fusion(down_feats[:, i], down_feats[:, i + 1])
                feat = self.featback_fusion(feats_list[i], feats_list[i + 1])

            feat = self.forward_backbone(down_feat, feat)
            feats_list[i] = feat

        for i in range(0, t):
            input_curr = inputs[:, i]

            if i == 0:
                down_feat = down_feats[:, i]
                feat = feats_list[i]
            else:
                down_feat = self.featward_fusion(down_feats[:, i], down_feats[:, i - 1])
                feat = self.featward_fusion(feats_list[i], feats_list[i - 1])

            feat = self.forward_backbone(down_feat, feat)

            out = self.upsampling(feat)
            out += input_curr
            outputs.append(out)

        results = torch.stack(outputs, dim=1)
        prior_zs = torch.stack(prior_zs, dim=1)
        priors = torch.stack(priors, dim=1)

        return results, prior_zs, priors

    def profile_forward(self, inputs):
        return self.forward(inputs)


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, dim=64, num_blocks=30):
        super().__init__()

        # self.cga = CGAFusion(dim)

        main = []
        main.append(nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        # main.append(
        #     make_layer(
        #         ResidualBlockNoBN, num_blocks, mid_channels=dim))
        main.append(
            make_layer(
                DEBlock, num_blocks, dim=dim))

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


# if __name__ == '__main__':
#     with open("../config/rmemvd_gopro.yml", mode='r') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     config = SimpleNamespace(**config)
#
#     x = torch.randn(2, 5, 3, 256, 256).cuda()
#     y = torch.randn(2, 5, 3, 256, 256).cuda()
#     model = Model(config).cuda()
#
#     x, pz, p = model(x, y)
#     print(x.shape)
#     print(pz.shape)
#     print(p.shape)




