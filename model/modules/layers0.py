"""
    Modified from https://github.com/open-mmlab/mmediting/blob/master/mmedit/models/common/sr_backbone_utils.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .HWDownsample import HWDownsample
from .cga import CGAFusion
from .transformer_arch import *
from .deablock import DEABlock


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.lrelu = nn.PReLU()

        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualBlockNoBN2(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)

        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.lrelu = nn.PReLU()

        # if res_scale == 1.0:
        #     self.init_weights()
        self.CA = CALayer(mid_channels, reduction=4, bias=False)

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        identity = x
        out = self.CA(self.conv2(self.lrelu(self.conv1(x))))
        return identity + out * self.res_scale


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # b, h, w, c
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class RepConv(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RepConv, self).__init__()
        self.conv_1 = nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size // 2, groups=n_feat)
        self.conv_2 = nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=n_feat)

    def forward(self, x):
        res_1 = self.conv_1(x)
        res_2 = self.conv_2(x)
        return res_1 + res_2 + x


class RepConv2(nn.Module):  # 3x3深度可分离卷积
    def __init__(self, n_feat, kernel_size, bias):
        super(RepConv2, self).__init__()
        # self.conv_1 = nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, padding=kernel_size//2, groups=n_feat//8)
        self.conv_2 = nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=n_feat)

    def forward(self, x):
        # res_1 = self.conv_1(x)
        res_2 = self.conv_2(x)
        return res_2 + x


class SimpleGate(nn.Module):  # 简单门控代替GELU等激活函数
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleGate2(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)


class CAB1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction=4, bias=False):
        super(CAB1, self).__init__()

        modules_body = []
        self.norm = LayerNorm2d(n_feat)

        modules_body.append(conv(n_feat, n_feat * 2, 1, bias=bias))
        modules_body.append(RepConv2(n_feat * 2, kernel_size, bias))
        modules_body.append(SimpleGate())
        modules_body.append(RepConv(n_feat, kernel_size, bias))
        modules_body.append(conv(n_feat, 2 * n_feat, 1, bias=bias))
        modules_body.append(SimpleGate2())
        modules_body.append(CALayer(n_feat, reduction, bias=bias))
        modules_body.append(conv(n_feat, n_feat, 1, bias=bias))

        self.body = nn.Sequential(*modules_body)
        self.beta = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = self.body(self.norm(x))
        res = x + res * self.beta
        return res


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            m.weight.data *= scale


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size - 1) // 2, bias=bias, stride=stride, groups=groups)


def conv1x1(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)


def conv3x3(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv5x5(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=bias)


def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer,
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    # kernel = F.softmax(kernel, dim=-1)

    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out


def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, groups=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, groups=groups)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, groups=groups)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out


class IDynamicDWConv(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size=3,
                 group_channels=1,
                 down=4,
                 conv_group=1):
        super(IDynamicDWConv, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.group_channels = group_channels
        self.down = down
        self.groups = self.channels // self.group_channels
        self.avgpool = nn.AvgPool2d(kernel_size=down, stride=down)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block1 = [ResBlock(channels, channels, kernel_size=kernel_size, stride=1, groups=conv_group)
                  for _ in range(3)]
        Block2 = [ResBlock(channels, channels, kernel_size=kernel_size, stride=1, groups=conv_group)
                  for _ in range(3)]
        self.tokernel = nn.Conv2d(channels, kernel_size ** 2 * self.groups, 1, 1, 0)
        self.Block1 = nn.Sequential(*Block1)
        self.Block2 = nn.Sequential(*Block2)

    def forward(self, x):
        weight = self.tokernel(self.Block2(self.maxpool(self.Block1(self.avgpool(x)))))
        weight = F.interpolate(weight, scale_factor=2 * self.down, mode='bilinear')
        out = kernel2d_conv(x, weight, ksize=self.kernel_size)
        return out


class DTFF(nn.Module):
    def __init__(self, num_feat=64):
        super(DTFF, self).__init__()

        self.num_feat = num_feat
        self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.kernel_conv_pixel = IDynamicDWConv(num_feat, kernel_size=3, group_channels=1, down=4, conv_group=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x_feat, feat_prop):
        feat_fusion = torch.cat([x_feat, feat_prop], dim=1)  # b 128 256 256
        feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 128 256 256
        feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1)
        feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
        feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
        feat_prop = feat_prop1 + feat_prop2
        # dynamic conv
        feat_prop = self.kernel_conv_pixel(feat_prop)

        return feat_prop


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layer, bias):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growth_rate, bias=bias))
            in_channels_ += growth_rate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layer, bias):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growth_rate, num_layer, bias)
        self.down_sampling = conv5x5(in_channels, 4 * in_channels, stride=2, bias=bias)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out


class RDBs(nn.Module):
    def __init__(self, in_dim, out_dim, num_RDB, growth_rate, num_layer, bias):
        super(RDBs, self).__init__()
        self.RDBs = nn.ModuleList(
            [RDB(in_dim, growth_rate=growth_rate, num_layer=num_layer, bias=bias) for _ in range(num_RDB)])
        self.conv = nn.Sequential(*[nn.Conv2d(in_dim * num_RDB, out_dim, kernel_size=1, padding=0, stride=1, bias=bias),
                                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=2, bias=bias)])

        self.downsampling = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride=2, bias=bias)

    def forward(self, x):
        input = x
        RDBs_out = []
        for rdb_block in self.RDBs:
            x = rdb_block(x)
            RDBs_out.append(x)
        x = self.conv(torch.cat(RDBs_out, dim=1))
        return x + self.downsampling(input)


# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growth_rate, bias):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growth_rate, stride=1, bias=bias)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DenseLayer(torch.nn.Module):
    def __init__(self, dim, growth_rate, bias):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(dim, growth_rate, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual 3D dense block
class RDB_3D(torch.nn.Module):
    def __init__(self, dim, growth_rate, num_dense_layer, bias):
        super(RDB_3D, self).__init__()
        self.layer = [DenseLayer(dim=dim + growth_rate * i, growth_rate=growth_rate, bias=bias) for i in
                      range(num_dense_layer)]
        self.layer = torch.nn.Sequential(*self.layer)
        self.conv = nn.Conv3d(dim + growth_rate * num_dense_layer, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.layer(x)
        out = self.conv(out)
        out += x

        return out


class RRDB(nn.Module):
    def __init__(self, in_dim, out_dim, num_RDB, growth_rate, num_dense_layer, bias):
        super(RRDB, self).__init__()
        self.RDBs = nn.ModuleList(
            [RDB_3D(dim=in_dim, growth_rate=growth_rate, num_dense_layer=num_dense_layer, bias=bias) for _ in
             range(num_RDB)])
        self.conv = nn.Sequential(*[nn.Conv3d(in_dim * num_RDB, out_dim, kernel_size=1, padding=0, stride=1, bias=bias),
                                    nn.Conv3d(out_dim, out_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                              stride=(1, 2, 2), bias=bias)])

        self.downsampling = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2),
                                      bias=bias)

    def forward(self, x):
        input = x
        RDBs_out = []
        for rdb_block in self.RDBs:
            x = rdb_block(x)
            RDBs_out.append(x)
        x = self.conv(torch.cat(RDBs_out, dim=1))
        return x + self.downsampling(input)


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


# Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.spatital = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                  padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # channel attention
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))  # b, c, 1, 1
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # b, 1, H, W
        spatial_out = self.sigmoid(self.spatital(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x


class DEABs(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(DEABs, self).__init__()
        modules_body = []
        modules_body = [DEABlock(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class CABs(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(CABs, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class IDBlocks(nn.Module):
    def __init__(self, n_feat, kernel_size, group_channels, num_heads,
                 ffn_type='GDFN', ffn_expansion_factor=2, num_cab=4):
        super(IDBlocks, self).__init__()
        modules_body = []
        for _ in range(num_cab):
            modules_body.append(
                IDynamicLayerBlock(dim=n_feat, kernel_size=kernel_size, idynamic_group_channels=group_channels,
                                   idynamic_ffn_type=ffn_type, idynamic_ffn_expansion_factor=ffn_expansion_factor))
            modules_body.append(
                SparseAttentionLayerBlock(dim=n_feat, restormer_num_heads=num_heads, restormer_ffn_type=ffn_type,
                                          restormer_ffn_expansion_factor=ffn_expansion_factor))

        modules_body.append(conv(n_feat, n_feat, kernel_size=3))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x) + x

        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        # self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True),
        #                    nn.PReLU())
        # self.down = HWDownsample(in_channels, in_channels + s_factor)
        self.down = nn.Conv2d(in_channels, in_channels + s_factor, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class SkipUpSample1(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample1, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels + s_factor, in_channels * 4, 3, stride=1, padding=1, bias=False),
                                nn.PixelShuffle(2))
        self.CGAFuser = CGAFusion(in_channels, reduction=4, bias=False)

    def forward(self, x, y):
        x = self.up(x)
        x = self.CGAFuser(y, x)
        return x


class FMN(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(FMN, self).__init__()
        self.conv1x1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

        self.norm1 = LayerNorm2d(n_feat * 2)
        self.norm2 = LayerNorm2d(n_feat)
        self.pconv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, groups=n_feat, bias=bias)
        self.pooling1x1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x, encoder_out, decoder_out):
        qk = torch.cat([self.conv1x1(x), self.conv1x1(encoder_out)], dim=1)
        qk = self.pooling1x1(self.conv3x3(self.pconv1x1(self.norm1(qk))))
        qk = F.softmax(qk, dim=1)
        v = self.norm2(self.conv1x1(decoder_out))

        x = qk * v

        return x


# Hierarchical Integration Module
class HIM(nn.Module):
    def __init__(self, dim, num_heads, embed_dim, bias):
        super(HIM, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(embed_dim * 4)

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(embed_dim * 4, 2 * dim, bias=bias)

        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, prior):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior = self.norm2(prior)

        q = self.q(_x)
        kv = self.kv(prior)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads)
        out = self.proj(out)

        # sum
        x = x + out
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=4, bias=False, scale_unetfeats=32):
        super(Encoder, self).__init__()

        scale_unetfeats = int(n_feat / 2)
        act = nn.PReLU()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # U-net skip
        self.skip_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.skip_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        self.skip_enc3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)

        self.skip_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.skip_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        self.skip_dec3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.skip_enc1(encoder_outs[0]) + self.skip_dec1(decoder_outs[0])

        x = self.down12(enc1)
        # x = self.down12(enc1) + self.down12(x)
        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.skip_enc2(encoder_outs[1]) + self.skip_dec2(decoder_outs[1])

        x = self.down23(enc2)
        # x = self.down23(enc2) + self.down23(x)
        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.skip_enc3(encoder_outs[2]) + self.skip_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Encoder1(nn.Module):
    def __init__(self, n_feat, embed_dim, group=4, num_heads=8, kernel_size=3, reduction=4, bias=False,
                 scale_unetfeats=32):
        super(Encoder1, self).__init__()

        scale_unetfeats = int(n_feat / 2)
        act = nn.PReLU()

        self.him1 = HIM(n_feat, num_heads, embed_dim=embed_dim, bias=bias)
        self.him2 = HIM(n_feat + scale_unetfeats, num_heads, embed_dim=embed_dim, bias=bias)
        self.him3 = HIM(n_feat + scale_unetfeats * 2, num_heads, embed_dim=embed_dim, bias=bias)

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down_prior12 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group * group, (group * group) // 4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim * 4, embed_dim * 4)
        )
        self.down_prior23 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group * group) // 4, 1),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim * 4, embed_dim * 4)
        )
        self.down_x12 = DownSample(n_feat, scale_unetfeats)
        self.down_x23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # U-net skip
        self.skip_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.skip_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        self.skip_enc3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)

        self.skip_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.skip_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        self.skip_dec3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)

    def forward(self, x, diff_prior, encoder_outs=None, decoder_outs=None):

        prior1 = diff_prior
        x = self.him1(x, prior1)
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.skip_enc1(encoder_outs[0]) + self.skip_dec1(decoder_outs[0])

        x = self.down_x12(enc1)
        prior2 = self.down_prior12(prior1)
        x = self.him2(x, prior2)
        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.skip_enc2(encoder_outs[1]) + self.skip_dec2(decoder_outs[1])

        x = self.down_x23(enc2)
        prior3 = self.down_prior23(prior2)
        x = self.him3(x, prior3)
        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.skip_enc3(encoder_outs[2]) + self.skip_dec3(decoder_outs[2])

        return [enc1, enc2, enc3], [prior1, prior2, prior3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=4, bias=False, scale_unetfeats=32):
        super(Decoder, self).__init__()

        scale_unetfeats = int(n_feat / 2)
        act = nn.PReLU()
        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        # self.skip_dec3 = FMN(n_feat + scale_unetfeats * 2, bias=bias)
        # self.skip_dec2 = FMN(n_feat + scale_unetfeats, bias=bias)
        # self.skip_dec1 = FMN(n_feat, bias=bias)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs, encoder_outs=None, decoder_outs=None):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        # if (encoder_outs is not None) and (decoder_outs is not None):
        #     dec3 = dec3 + self.skip_dec3(enc3, encoder_outs[2], decoder_outs[2])

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        # if (encoder_outs is not None) and (decoder_outs is not None):
        #     dec2 = dec2 + self.skip_dec2(enc2, encoder_outs[1], decoder_outs[1])

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        # if (encoder_outs is not None) and (decoder_outs is not None):
        #     dec1 = dec1 + self.skip_dec1(enc1, encoder_outs[0], decoder_outs[0])

        return [dec1, dec2, dec3]


class Decoder1(nn.Module):
    def __init__(self, n_feat, embed_dim, num_heads, kernel_size=3, reduction=4, bias=False,
                 scale_unetfeats=32):
        super(Decoder1, self).__init__()
        scale_unetfeats = int(n_feat / 2)
        act = nn.PReLU()

        self.him1 = HIM(n_feat, num_heads, embed_dim=embed_dim, bias=bias)
        self.him2 = HIM(n_feat + scale_unetfeats, num_heads, embed_dim=embed_dim, bias=bias)
        self.him3 = HIM(n_feat + scale_unetfeats * 2, num_heads, embed_dim=embed_dim, bias=bias)

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs, diff_prior):
        enc1, enc2, enc3 = outs
        prior1, prior2, prior3 = diff_prior

        x = self.him3(enc3, prior3)
        dec3 = self.decoder_level3(x)

        x = self.up32(dec3, self.skip_attn2(enc2))
        x = self.him2(x, prior2)
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        x = self.him1(x, prior1)
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


class TFR(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(TFR, self).__init__()

        # self.orb1 = DEABs(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb1 = CABs(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = CABs(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = CABs(n_feat, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):

        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


class TFR_UNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats=32):
        super(TFR_UNet, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(3)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # U-net skip
        self.skip_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.skip_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        self.skip_enc3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)

        self.skip_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.skip_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
        self.skip_dec3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act)
                               for _ in range(3)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        # self.skip_attn1 = DEABlock(n_feat, kernel_size, reduction, bias=bias, act=act)
        # self.skip_attn2 = DEABlock(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x, encoder_outs, decoder_outs):

        enc1 = self.encoder_level1(x)
        enc1 = enc1 + self.skip_enc1(encoder_outs[0]) + self.skip_dec1(decoder_outs[0])
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        enc2 = enc2 + self.skip_enc2(encoder_outs[1]) + self.skip_dec2(decoder_outs[1])
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        enc3 = enc3 + self.skip_enc3(encoder_outs[2]) + self.skip_dec3(decoder_outs[2])

        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return dec1


if __name__ == '__main__':
    x = torch.randn(2, 64, 256, 256).cuda()
    net = IDynamicDWConv(channels=64, kernel_size=3, group_channels=1).cuda()
    x = net(x)
    print(x.shape)
