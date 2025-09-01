import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from einops import rearrange
from functools import partial
from .modules import *
from .ldm import *
from train.beta_schedule import make_beta_schedule, default
from .memory_bank1 import MemoryBank

# import argparse
# import yaml
# from types import SimpleNamespace


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()

        self.mid_channels = para.mid_channels
        self.mem_every = para.mem_every
        # self.deep_update_every = para.deep_update_every
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
            RDBs(self.n_feats, self.mid_channels, num_RDB=4, growth_rate=16, num_layer=3, bias=False)  # b, 64, h/2, w/2
        )

        # SA transformer
        transformer_scale4 = []
        for _ in range(3):
            transformer_scale4.append(
                nn.Sequential(
                    CWGDN(dim=self.mid_channels, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'))
            )
        self.transformer_scale4 = nn.Sequential(*transformer_scale4)
        # self.transformer_scale4 = CWGDN1(dim=self.mid_channels, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(self.mid_channels, self.n_feats, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv5x5(self.n_feats, 3, stride=1)
        )

        self.conv_trans = nn.Conv2d(self.mid_channels * 2, 256, 3, 1, 1)

        # Feature fusion Module
        self.feat_fusion = DTFF(self.mid_channels)

        # self.forward_backbone = ResidualBlocksWithInputConv2(
        #     self.mid_channels * 3, self.mid_channels, self.num_blocks_forward)
        # self.backward_backbone = ResidualBlocksWithInputConv2(
        #     self.mid_channels * 2, self.mid_channels, self.num_blocks_backward)
        self.forward_backbone = ResidualBlocksWithInputConv(self.mid_channels, self.num_blocks_forward)
        self.backward_backbone = ResidualBlocksWithInputConv(self.mid_channels, self.num_blocks_backward)

        self.tfr = TFR(n_feat=self.mid_channels, kernel_size=3, reduction=4, act=nn.PReLU(),
                       bias=False, scale_unetfeats=int(self.mid_channels / 2), num_cab=5)
        # self.tfr = TFR_UNet(n_feat=self.mid_channels, kernel_size=3, reduction=4, act=nn.PReLU(),
        #                     bias=False, scale_unetfeats=int(self.mid_channels / 2))

        # ----------------- Memory branch -----------------
        self.memory = Memory(para)

    def forward(self, inputs, gts=None, noise=None, profile_flag=False):
        if profile_flag:
            return self.profile_forward(inputs)

        self.memory.mem_bank_backward.init_memory()
        self.memory.mem_bank_forward.init_memory()

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

        prev_feat = None
        encoder_outs = None
        decoder_outs = None
        encoder_outs_list = []
        decoder_outs_list = []

        # ------------ backward ------------
        for i in range(t - 1, -1, -1):
            down_feat = down_feats[:, i]
            key_curr, shrinkage, selection, encoder_outs, diff_prior = self.memory.encode_key(down_feat, priors[i],
                                                                                              encoder_outs=encoder_outs,
                                                                                              decoder_outs=decoder_outs)
            f, f_1, f_2 = encoder_outs

            if i == t - 1:
                if self.memory.mem_bank_backward.get_hidden() is None:
                    self.memory.mem_bank_backward.create_hidden_state(key_curr)
                memory_readout = self.conv_trans(f_2)
                hidden = self.memory.mem_bank_backward.get_hidden()
                hidden, decoder_outs = self.memory.decoder(f_2, f_1, f, diff_prior, memory_readout, hidden)
                feat = decoder_outs[0]
                # feat = torch.cat([down_feat, decoder_outs[0]], dim=1)
            else:
                memory_readout = self.memory.mem_bank_backward.match_memory(key_curr, selection)
                hidden = self.memory.mem_bank_backward.get_hidden()
                hidden, decoder_outs = self.memory.decoder(f_2, f_1, f, diff_prior, memory_readout, hidden)
                down_feat = self.feat_fusion(down_feat, down_feats[:, i + 1])
                feat = self.feat_fusion(decoder_outs[0], prev_feat)
                # feat = torch.cat([down_feat, feat], dim=1)

            self.memory.mem_bank_backward.set_hidden(hidden)
            # feat = self.backward_backbone(feat)
            feat = self.backward_backbone(down_feat, feat)
            decoder_outs[0] = feat
            encoder_outs_list.append(encoder_outs)
            decoder_outs_list.append(decoder_outs)

            # mem_every
            hidden = self.memory.mem_bank_backward.get_hidden()
            value, _ = self.memory.encode_value(down_feats[:, i], f_2, feat, hidden)
            self.memory.mem_bank_backward.add_memory(key_curr, value=value, shrinkage=shrinkage)

            # # mem_alternate: stride of 2->odd-numbered frames
            # if i % 2 == 1:
            #     hidden = self.memory.mem_bank_backward.get_hidden()
            #     value, _ = self.memory.encode_value(down_feats[:, i], f_2, feat, hidden)
            #     self.memory.mem_bank_backward.add_memory(key_curr, value=value, shrinkage=shrinkage)

            prev_feat = feat

        encoder_outs_list = list(reversed(encoder_outs_list))
        decoder_outs_list = list(reversed(decoder_outs_list))

        # ----------- forward --------------
        encoder_forward_list = []
        decoder_forward_list = []
        hidden_future = self.memory.mem_bank_backward.get_hidden()
        for i in range(0, t):
            input_curr = inputs[:, i]
            down_feat = down_feats[:, i]
            key_curr, shrinkage, selection, encoder_outs, diff_prior = self.memory.encode_key(down_feat, priors[i],
                                                                                              encoder_outs=encoder_outs,
                                                                                              decoder_outs=decoder_outs)
            f, f_1, f_2 = encoder_outs

            # Memory from backward
            memory_readout = self.memory.mem_bank_backward.match_memory(key_curr, selection)
            # print('memory_readout', torch.max(memory_readout))
            _, feat_future = self.memory.decoder(f_2, f_1, f, diff_prior, memory_readout, hidden_future, h_out=False)
            ff, ff_1, ff_2 = feat_future

            if i == 0:
                if self.memory.mem_bank_forward.get_hidden() is None:
                    self.memory.mem_bank_forward.create_hidden_state(key_curr)
                hidden = self.memory.mem_bank_forward.get_hidden()
                hidden, decoder_outs = self.memory.decoder(ff_2, ff_1, ff, diff_prior, memory_readout, hidden)
                feat = self.feat_fusion(decoder_outs[0], prev_feat)
                # feat = torch.cat([down_feat, feat], dim=1)
            else:
                hidden = self.memory.mem_bank_forward.get_hidden()
                memory_readout = self.memory.mem_bank_forward.match_memory(key_curr, selection)
                hidden, decoder_outs = self.memory.decoder(ff_2, ff_1, ff, diff_prior, memory_readout, hidden)
                down_feat = self.feat_fusion(down_feat, down_feats[:, i - 1])
                feat = self.feat_fusion(decoder_outs[0], prev_feat)
                # feat = torch.cat([down_feat, feat], dim=1)

            self.memory.mem_bank_forward.set_hidden(hidden)
            # feat = self.forward_backbone(feat)
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

            # mem_every
            hidden = self.memory.mem_bank_forward.get_hidden()
            value, _ = self.memory.encode_value(down_feats[:, i], f_2, self.downsampling(out), hidden)
            self.memory.mem_bank_forward.add_memory(key_curr, value=value, shrinkage=shrinkage)

            # # mem_alternate: stride of 2->even-numbered frames
            # if i % 2 == 0:
            #     hidden = self.memory.mem_bank_forward.get_hidden()
            #     value, _ = self.memory.encode_value(down_feats[:, i], f_2, self.downsampling(out), hidden)
            #     self.memory.mem_bank_forward.add_memory(key_curr, value=value, shrinkage=shrinkage)

            prev_feat = feat

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


class ResidualBlocksWithInputConv2(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        # main.append(
        #     make_layer(
        #         ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        main.append(
            make_layer(
                DEBlock, num_blocks, dim=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, g):
        out_g = self.conv2(self.lrelu(self.conv1(g)))

        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g


class KeyEncoder(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()

        self.key_encoder = Encoder(n_feat=in_dim)
        # self.key_encoder = Encoder1(n_feat=in_dim)

    def forward(self, x, diff_prior, encoder_outs=None, decoder_outs=None):
        return self.key_encoder(x, diff_prior, encoder_outs=encoder_outs, decoder_outs=decoder_outs)


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim=64):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.s_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x, need_s, need_e):
        shrinkage = self.s_proj(x) ** 2 + 1 if (need_s) else None  # s: [1,+00)
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None  # e: [0,1]

        return self.key_proj(x), shrinkage, selection


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.block1 = ResnetBlock(x_in_dim + g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = ResnetBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        g = torch.cat([x, g], dim=1)
        g = self.block1(g)
        r = self.attention(g)
        g = self.block2(g + r)

        return g


class HiddenReinforcer(nn.Module):

    def __init__(self, in_dim, hidden_dim, kernel_size):
        """
        Initialize the ConvLSTM cell
        """
        super(HiddenReinforcer, self).__init__()
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = "same"
        self.hidden_dim = hidden_dim

        self.conv_gates = nn.Conv2d(
            in_channels=in_dim + hidden_dim,
            out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding
        )

        self.conv_can = nn.Conv2d(
            in_channels=in_dim + hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, input_x, h_cur):
        """
        :param input_x: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_x, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_x, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = h_cur * (1 - update_gate) + update_gate * cnm
        return h_next


class ValueEncoder(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=96):
        super().__init__()

        # self.forward_resblocks = ResidualBlocksWithInputConv2(in_dim * 2, 64, 2)
        self.pre_fusion = CGAFusion(in_dim)

        resnet = resnet50(pretrained=True, extra_chan=64 - 3)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4

        self.fuser = FeatureFusionBlock(in_dim * 2, 256, 128, 256)

        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(256, hidden_dim, kernel_size=3)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat, gt, hidden, is_deep_update=False):

        # x = self.forward_resblocks(torch.cat([image, gt], 1))
        x = self.pre_fusion(image, gt)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)  # b, 256, 1/4

        x = self.fuser(image_feat, x)

        if is_deep_update and self.hidden_reinforce is not None:
            hidden = self.hidden_reinforce(x, hidden)

        return x, hidden


class HiddenUpdater(nn.Module):

    def __init__(self, in_dim, hidden_dim, kernel_size):
        """
        Initialize the ConvLSTM cell
        """
        super(HiddenUpdater, self).__init__()
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = "same"
        self.hidden_dim = hidden_dim

        self.f_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, stride=2, padding=1))
        self.f1_conv = nn.Conv2d(in_dim + int(in_dim / 2), in_dim, kernel_size=3, stride=2, padding=1)
        self.f2_conv = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1)

        self.conv_gates = nn.Conv2d(
            in_channels=in_dim + hidden_dim,
            out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding
        )

        self.conv_can = nn.Conv2d(
            in_channels=in_dim + hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, x, h_cur):
        x = self.f_conv(x[0]) + self.f1_conv(x[1]) + self.f2_conv(x[2])

        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([x, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = h_cur * (1 - update_gate) + update_gate * cnm
        return h_next


class MemoryDecoder(nn.Module):  # Decoder
    def __init__(self, n_feat=64, hidden_dim=96):
        super().__init__()

        self.hidden_dim = hidden_dim

        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater(n_feat, hidden_dim, kernel_size=3)
        else:
            self.hidden_update = None

        self.fuser = FeatureFusionBlock(128, 256 + hidden_dim, 128, 128)

        self.decoder = Decoder(n_feat=n_feat)
        # self.decoder = Decoder1(n_feat=n_feat)

    def forward(self, f_2, f_1, f, diff_prior, memory_readout, hidden_state, h_out=True):
        if self.hidden_update is not None:
            memory_readout = torch.cat([memory_readout, hidden_state], dim=1)

        f_2 = self.fuser(f_2, memory_readout)
        decoder_outs = self.decoder([f, f_1, f_2], diff_prior)

        if h_out and self.hidden_update is not None:
            hidden_state = self.hidden_update(decoder_outs, hidden_state)

        return hidden_state, decoder_outs


class Memory(nn.Module):
    def __init__(self, para):
        super().__init__()

        self.key_encoder = KeyEncoder(64)
        self.key_proj = KeyProjection(128, keydim=64)

        self.value_encoder = ValueEncoder(64, hidden_dim=96)

        self.decoder = MemoryDecoder(n_feat=64, hidden_dim=96)

        self.mem_bank_forward = MemoryBank(para=para, count_usage=True)
        self.mem_bank_backward = MemoryBank(para=para, count_usage=True)

    def encode_value(self, image, image_feat, gt, hidden_state, is_deep_update=False):
        f, h = self.value_encoder(image, image_feat, gt, hidden_state, is_deep_update=is_deep_update)
        return f, h

    def encode_key(self, frame, diff_prior, encoder_outs=None, decoder_outs=None, need_sk=True, need_ek=True):
        encoder_outs, diff_prior = self.key_encoder(frame, diff_prior, encoder_outs=encoder_outs,
                                                    decoder_outs=decoder_outs)
        k, shrinkage, selection = self.key_proj(encoder_outs[2], need_s=need_sk, need_e=need_ek)
        return k, shrinkage, selection, encoder_outs, diff_prior


def cost_profile(model, H, W, seq_length=5):
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
