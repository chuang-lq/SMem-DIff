import torch
import torch.nn as nn
from einops import rearrange
from .ddpm_ import DDPM_
from .common import *


class LE_arch(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=6):
        super(LE_arch, self).__init__()
        E1 = [nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]
        E2 = [
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(4)

    def forward(self, x, gt):
        gt0 = self.pixel_unshuffle(gt)
        x0 = self.pixel_unshuffle(x)
        x = torch.cat([x0, gt0], dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1


class CE(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=6):
        super(CE, self).__init__()
        E1 = [nn.Conv2d(48, n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]
        E2 = [
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1


class ResMLP(nn.Module):
    def __init__(self, n_feats=512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats, n_feats),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        res = self.resmlp(x)
        return res


class denoise(nn.Module):
    def __init__(self, n_feats=64, n_denoise_res=5, timesteps=5):
        super(denoise, self).__init__()
        self.max_period = timesteps * 10
        n_featsx4 = 4 * n_feats
        resmlp = [
            nn.Linear(n_featsx4 * 2 + 1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp = nn.Sequential(*resmlp)

    def forward(self, x, t, c):
        t = t.float()
        t = t / self.max_period
        t = t.view(-1, 1)
        c = torch.cat([c, t, x], dim=1)
        fea = self.resmlp(c)

        return fea


class DM_arch(nn.Module):
    def __init__(self, num_feat=64, n_denoise_res=1, linear_start=0.1, linear_end=0.99, timesteps=4):
        super().__init__()
        self.num_feat = num_feat
        self.condition = CE(n_feats=64)  # n_encoder_res=n_encoder_res
        self.denoise = denoise(n_feats=64, n_denoise_res=n_denoise_res, timesteps=timesteps)

        self.diffusion = DDPM_(denoise=self.denoise, condition=self.condition, n_feats=64, linear_start=linear_start,
                              linear_end=linear_end, timesteps=timesteps)

    def forward(self, lq, IPR_S1=None):
        if self.training:
            IPR, IPR_list = self.diffusion(rearrange(lq, 'b t c h w -> (b t) c h w'), IPR_S1)
            return IPR, IPR_list
        else:
            IPR = self.diffusion(rearrange(lq, 'b t c h w -> (b t) c h w'))
            return IPR
