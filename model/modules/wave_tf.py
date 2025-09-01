import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer_z import TransformerBlock, HIM
from .HWDownsample import HaarDownsampling
from .ScConv import ScConv


def conv3x3(in_chn, out_chn, bias=True):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)


def conv(in_channels, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=(kernel_size // 2), bias=bias)


def dwt_init(x):
    # b, c, h, w
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    # return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)  # [b, C * 4, H // 2, W //2]
    # return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)  # [b * 4, C, H // 2, W //2]
    return x_LL, x_HL, x_LH, x_HH


# 使用haar小波变换来实现二维离散小波
def iwt_init(x):
    # b, c, h, w
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])

    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # out_batch, out_channel, out_height, out_width = int(in_batch / r ** 2), int(in_channel), r * in_height, r * in_width
    # x1 = x[0:out_batch, :, :, :] / 2
    # x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    # x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    # x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width], dtype=x.dtype, device=x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self, fuseh=False):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导
        self.fuseh = fuseh

    def forward(self, x):
        if self.fuseh:
            x_LL, x_HL, x_LH, x_HH = dwt_init(x)
            return x_LL, torch.cat((x_HL, x_LH, x_HH), dim=1)
        else:
            return dwt_init(x)


# 逆向二维离散小波
class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


# Adaptive Fine-Grained Channel Attention Layer
class FCALayer(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(FCALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        # 一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (1,c,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)  # (1,1,c)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)  # (1,c,1,1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)

        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input * out


# Half Wavelet Dual Attention Block (HWB)
class HWB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(HWB, self).__init__()
        self.dwt = DWT()
        self.iwt = IDWT()

        modules_body = [
            conv(n_feat // 2, n_feat // 2, kernel_size, bias=bias),
            act,
            conv(n_feat // 2, n_feat // 2, kernel_size, bias=bias)
        ]
        self.body = nn.Sequential(*modules_body)

        self.WSA = SALayer(5, bias=bias)
        self.WCA = CALayer(n_feat // 2, reduction, bias=bias)
        # self.WCA = FCALayer(n_feat // 2)

        self.conv1x1 = nn.Conv2d(n_feat, n_feat // 2, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.act = act

    def forward(self, x):
        identity = x

        # Split 2 part
        wavelet_path, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain (Dual attention)
        x_dwt = self.dwt(wavelet_path)
        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x_dwt
        wavelet_path = self.iwt(res)

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.act(self.conv3x3(out))
        out += identity

        return out


class CAB2(nn.Module):
    def __init__(self, dim, embed_dim, kernel_size, reduction, bias, act, group):
        super(CAB2, self).__init__()

        self.group = group

        modules_body = []
        modules_body.append(conv(dim, dim, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(dim, dim, kernel_size, bias=bias))

        self.CA = CALayer(dim, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

        if self.group == 1:
            self.kernel = nn.Linear(embed_dim * 4, dim * 2, bias=bias)

    def forward(self, x, prior):
        identity = x

        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            x = x * kv1 + kv2
        res = self.body(x)
        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            res = res * kv1 + kv2
        res = self.CA(res)
        res += identity

        return res


class CAB2s(nn.Module):
    def __init__(self, dim, embed_dim, num_heads, kernel_size, reduction, act, bias, LayerNorm_type, group, num_cab):
        super(CAB2s, self).__init__()

        self.group = group

        self.blocks = nn.ModuleList(
            [CAB2(dim, embed_dim=embed_dim, kernel_size=kernel_size,
                  reduction=reduction, bias=bias, act=act, group=group) for _ in range(num_cab)])

        if self.group > 1:
            self.him = HIM(dim=dim, num_heads=num_heads, bias=bias, embed_dim=embed_dim, LayerNorm_type=LayerNorm_type)

    def forward(self, x, prior):
        if prior is not None and self.group > 1:
            x = self.him(x, prior)

        for blk in self.blocks:
            x = blk(x, prior)

        return x


class HWB1(nn.Module):
    def __init__(self, dim, embed_dim, num_heads, kernel_size, reduction, bias, act, group, LayerNorm_type, num_blocks):
        super(HWB1, self).__init__()

        self.group = group
        self.wave = HaarDownsampling(dim // 2)

        self.blocks = nn.ModuleList(
            [CAB2(dim // 2, embed_dim=embed_dim, kernel_size=kernel_size, reduction=reduction, bias=bias,
                  act=act, group=group) for _ in range(num_blocks)])

        self.x_wave_conv1 = nn.Conv2d(dim // 2 * 3, dim // 2 * 3, 1, 1, 0, groups=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.x_wave_conv2 = nn.Conv2d(dim // 2 * 3, dim // 2 * 3, 1, 1, 0, groups=3)

        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        # self.conv3x3 = ScConv(dim)
        self.act = nn.PReLU()

        if self.group > 1:
            self.him = HIM(dim=dim, num_heads=num_heads, bias=bias, embed_dim=embed_dim, LayerNorm_type=LayerNorm_type)

    def forward(self, x, prior=None):
        identity = x

        if prior is not None and self.group > 1:
            x = self.him(x, prior)

        # Split 2 part
        wavelet_path, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain
        x_wave_l, x_wave_h = self.wave(wavelet_path)
        for blk in self.blocks:
            x_wave_l = blk(x_wave_l, prior)
        x_wave_h = self.x_wave_conv2(self.lrelu(self.x_wave_conv1(x_wave_h)))
        wavelet_path = self.wave(torch.cat([x_wave_l, x_wave_h], dim=1), rev=True)

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.act(self.conv3x3(out))
        out += identity

        return out


class BasicLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group, num_blocks):

        super().__init__()
        self.group = group

        # build blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in
             range(num_blocks)])

        if self.group > 1:
            self.him = HIM(dim=dim, num_heads=num_heads, bias=bias, embed_dim=embed_dim, LayerNorm_type=LayerNorm_type)

    def forward(self, x, prior=None):

        if prior is not None and self.group > 1:
            x = self.him(x, prior)

        for blk in self.blocks:
            x = blk(x, prior)

        return x


class HWB2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group, num_blocks):
        super(HWB2, self).__init__()

        self.group = group
        self.wave = HaarDownsampling(dim // 2)

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=dim // 2, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in
             range(num_blocks)])

        self.x_wave_conv1 = nn.Conv2d(dim // 2 * 3, dim // 2 * 3, 1, 1, 0, groups=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.x_wave_conv2 = nn.Conv2d(dim // 2 * 3, dim // 2 * 3, 1, 1, 0, groups=3)

        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        # self.conv3x3 = ScConv(dim)
        self.act = nn.PReLU()

        if self.group > 1:
            self.him = HIM(dim=dim, num_heads=num_heads, bias=bias, embed_dim=embed_dim, LayerNorm_type=LayerNorm_type)

    def forward(self, x, prior=None):
        identity = x

        if prior is not None and self.group > 1:
            x = self.him(x, prior)

        # Split 2 part
        wavelet_path, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain
        x_wave_l, x_wave_h = self.wave(wavelet_path)
        for blk in self.blocks:
            x_wave_l = blk(x_wave_l, prior)
        x_wave_h = self.x_wave_conv2(self.lrelu(self.x_wave_conv1(x_wave_h)))
        wavelet_path = self.wave(torch.cat([x_wave_l, x_wave_h], dim=1), rev=True)

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.act(self.conv3x3(out))
        out += identity

        return out


if __name__ == '__main__':
    x = torch.rand(3, 64, 128, 128)
    hwb = HWB(64, 3, 8, bias=False, act=nn.PReLU())
    x = hwb(x)
    print(x.shape)
