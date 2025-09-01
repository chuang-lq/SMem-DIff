"""
    code based on :
        -[basicsr SwinIR] github: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/swinir_arch.py
        -[Restormer] github: https://github.com/swz30/Restormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
from einops import rearrange

# for idynamic
from .idynamicdwconv_util import *


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()

        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """
        GDFN in Restormer: [github] https://github.com/swz30/Restormer
    """
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        hidden_dim = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):

        super(BaseFeedForward, self).__init__()
        hidden_dim = int(dim*ffn_expansion_factor)

        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=bias),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=bias),
        )

    def forward(self, x):
        return self.body(x)


# IDynamicDWConvBlock
class IDynamicDWConvBlock(nn.Module):
    """
        code based on: [github] https://github.com/Atten4Vis/DemystifyLocalViT/blob/master/models/dwnet.py
        but we remove reductive Norm Layers and Activation Layers for better performance in SR-task
    """
    def __init__(self, dim, kernel_size, dynamic=True, group_channels=None, bias=True):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size  # Wh, Ww
        self.dynamic = dynamic
        self.group_channels = group_channels

        # pw-linear
        # in pw-linear layer we inherit settings from DWBlock. Set bias=False
        self.conv0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)

        if dynamic:
            self.conv = IDynamicDWConv(dim, kernel_size=kernel_size, group_channels=group_channels, bias=bias)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=dim, bias=bias)

    def forward(self, x):
        # shortcut outside the block
        x = self.conv0(x)
        x = self.conv(x)
        x = self.conv1(x)
        return x


# Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    """
        MDTA in Restormer: [github] https://github.com/swz30/Restormer
    """
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()

        self.dim = dim

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature

        # attn = attn.softmax(dim=-1)
        attn = self.softmax(attn)

        out = (attn @ v)
        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        out = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class SparseAttention(nn.Module):
    """
        SparseGSA is based on MDTA
        MDTA in Restormer: [github] https://github.com/swz30/Restormer
    """
    def __init__(self, dim, num_heads, bias, activation='relu'):
        super(SparseAttention, self).__init__()

        self.dim = dim

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.Identity()

        # ['gelu', 'sigmoid'] is for ablation study
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature

        # attn = attn.softmax(dim=-1)
        attn = self.act(attn)   # Sparse Attention due to ReLU's property

        out = (attn @ v)

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        out = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# IDynamicDWBlock with GDFN
class IDynamicLayerBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, idynamic_group_channels=6, idynamic_ffn_type='GDFN',
                 idynamic_ffn_expansion_factor=2, idynamic=True):
        super(IDynamicLayerBlock, self).__init__()

        self.dim = dim

        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')

        # IDynamic Local Feature Calculate
        self.IDynamicDWConv = IDynamicDWConvBlock(dim, kernel_size=kernel_size, dynamic=idynamic,
                                                  group_channels=idynamic_group_channels)

        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        # FeedForward Network
        if idynamic_ffn_type == 'GDFN':
            self.IDynamic_ffn = FeedForward(dim, ffn_expansion_factor=idynamic_ffn_expansion_factor, bias=False)
        elif idynamic_ffn_type == 'BaseFFN':
            self.IDynamic_ffn = BaseFeedForward(dim, ffn_expansion_factor=idynamic_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{idynamic_ffn_type}')

    def forward(self, x):
        x = self.IDynamicDWConv(self.norm1(x)) + x
        x = self.IDynamic_ffn(self.norm2(x)) + x
        return x


class RestormerLayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2):
        super(RestormerLayerBlock, self).__init__()

        self.dim = dim

        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')

        # Restormer Attention
        self.restormer_attn = Attention(dim, num_heads=restormer_num_heads, bias=False)

        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        # Restormer FeedForward
        if restormer_ffn_type == 'GDFN':
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{restormer_ffn_type}')

    def forward(self, x):
        x = self.restormer_attn(self.norm1(x)) + x
        x = self.restormer_ffn(self.norm2(x)) + x
        return x


class SparseAttentionLayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2,
                 activation='relu'):
        super(SparseAttentionLayerBlock, self).__init__()

        self.dim = dim

        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')

        # We use SparseGSA inplace MDTA
        self.restormer_attn = SparseAttention(dim, num_heads=restormer_num_heads, bias=False, activation=activation)

        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        # Restormer FeedForward
        if restormer_ffn_type == 'GDFN':
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{restormer_ffn_type}')

    def forward(self, x):
        x = self.restormer_attn(self.norm1(x)) + x
        x = self.restormer_ffn(self.norm2(x)) + x
        return x


class BuildBlock(nn.Module):

    def __init__(self, dim, blocks=3, buildblock_type='edge', window_size=7, idynamic_group_channels=6,
                 idynamic_ffn_type='GDFN', idynamic_ffn_expansion_factor=2, idynamic=True,
                 restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2,
                 activation='relu'):
        super(BuildBlock, self).__init__()

        self.dim = dim
        self.blocks = blocks
        self.buildblock_type = buildblock_type
        self.window_size = window_size
        self.num_heads = (idynamic_group_channels, restormer_num_heads)
        self.ffn_type = (idynamic_ffn_type, restormer_ffn_type)
        self.ffn_expansion = (idynamic_ffn_expansion_factor, restormer_ffn_expansion_factor)
        self.idynamic = idynamic

        # buildblock body
        body = []
        if buildblock_type == 'edge':
            for _ in range(blocks):
                body.append(IDynamicLayerBlock(dim, window_size, idynamic_group_channels, idynamic_ffn_type, idynamic_ffn_expansion_factor, idynamic))
                body.append(RestormerLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor))

        elif buildblock_type == 'sparse-edge':
            for _ in range(blocks):
                body.append(IDynamicLayerBlock(dim, window_size, idynamic_group_channels, idynamic_ffn_type, idynamic_ffn_expansion_factor, idynamic))
                body.append(SparseAttentionLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, activation))

        elif buildblock_type == 'idynamic':
            for _ in range(blocks):
                body.append(IDynamicLayerBlock(dim, window_size, idynamic_group_channels, idynamic_ffn_type, idynamic_ffn_expansion_factor, idynamic))

        elif buildblock_type == 'restormer':
            for _ in range(blocks):
                body.append(RestormerLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor))

        body.append(nn.Conv2d(dim, dim, 3, 1, 1))   # as like SwinIR, we use one Conv3x3 layer after buildblock
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x


if __name__ == '__main__':
    x = torch.randn(2, 64, 256, 256).cuda()
    model = IDynamicDWConvBlock(64, kernel_size=7, group_channels=6).cuda()
    print(model(x).shape)
