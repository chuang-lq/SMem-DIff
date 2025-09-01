import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from .transformer_layers import *


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()

        assert LayerNorm_type in ['BiasFree', 'WithBias']
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class QueryTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type='BiasFree', ffn_expansion_factor=4, bias=False):
        super(QueryTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.read_from_pixel = Attention(dim, num_heads, bias=bias)
        # self.norm = LayerNorm(dim, LayerNorm_type)
        # self.self_attn = Attention(dim, num_heads, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn1 = FFN(dim, ffn_expansion_factor, bias=True)
        self.ffn1 = SGFN(dim, ffn_expansion_factor, bias=bias)

        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.read_from_query = Attention(dim, num_heads, bias=bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = PixelFFN(dim, ffn_expansion_factor, bias=bias)

    def forward(self, x, pixel):
        x = x + self.read_from_pixel(self.norm1(x), pixel)
        # x = x + self.self_attn(self.norm(x))
        x = x + self.ffn1(self.norm2(x))

        pixel = pixel + self.read_from_query(self.norm3(pixel), x)
        pixel = pixel + self.ffn2(self.norm4(pixel))

        return x, pixel


class QueryTransformerBlock1(torch.nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type='BiasFree', ffn_expansion_factor=4, bias=False):
        super(QueryTransformerBlock1, self).__init__()

        self.cafm1 = CAFM(dim)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.read_from_pixel = Attention(dim, num_heads, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = GDFN(dim, ffn_expansion_factor, bias=bias)

        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.read_from_query = Attention(dim, num_heads, bias=bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = PixelFFN(dim, ffn_expansion_factor, bias=bias)

    def forward(self, x, pixel):
        x, pixel = self.cafm1(x, pixel)
        x = x + self.read_from_pixel(self.norm1(x), pixel)
        x = x + self.ffn1(self.norm2(x))

        pixel = pixel + self.read_from_query(self.norm3(pixel), x)
        pixel = pixel + self.ffn2(self.norm4(pixel))

        return x, pixel


class QueryTransformer(nn.Module):
    def __init__(self, dim, num_heads, num_blocks, LayerNorm_type='BiasFree', ffn_expansion_factor=4, bias=False):
        super().__init__()

        # self.query_init = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.pixel_init = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.query_init = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=bias)
        self.pixel_init = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=bias)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            # QueryTransformerBlock(dim, num_heads, LayerNorm_type=LayerNorm_type,
            #                       ffn_expansion_factor=ffn_expansion_factor, bias=bias) for _ in range(self.num_blocks)
            QueryTransformerBlock1(dim // 2, num_heads, LayerNorm_type=LayerNorm_type,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias) for _ in range(self.num_blocks)
        )

    def forward(self, x, pixel):
        x = self.query_init(x)
        pixel = self.pixel_init(pixel)

        for i in range(self.num_blocks):
            x, pixel = self.blocks[i](x, pixel)

        return x, pixel


class SCABlock(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type='BiasFree', ffn_expansion_factor=4, bias=False):
        super(SCABlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.p2sp = P2SP(dim, num_heads, bias=bias)
        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.self_attn1 = Attention(dim, num_heads, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = GDFN(dim, ffn_expansion_factor, bias=bias)

        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.sp2p = SP2P(dim, num_heads, bias=bias)
        self.norm3_1 = LayerNorm(dim, LayerNorm_type)
        self.self_attn2 = Attention(dim, num_heads, bias=bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = PixelFFN(dim, ffn_expansion_factor, bias=bias)

    def forward(self, x, sp):
        sp = sp + self.p2sp(x, self.norm1(sp))
        sp = sp + self.self_attn1(self.norm1_1(sp))
        sp = sp + self.ffn1(self.norm2(sp))

        x = x + self.sp2p(self.norm3(x), sp)
        x = x + self.self_attn2(self.norm3_1(x))
        x = x + self.ffn2(self.norm4(x))

        return x, sp


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),

            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),

            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PixelShuffleBlock(nn.Module):
    def __init__(self, dim, bias):
        super(PixelShuffleBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1, stride=1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.shuffle(x)

        return x


class SCA(nn.Module):
    def __init__(self, dim, num_heads, num_blocks, LayerNorm_type='BiasFree', ffn_expansion_factor=4, bias=False):
        super(SCA, self).__init__()

        self.p_conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.sp_conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        # self.patch_embed = PatchEmbed(dim * 2, dim * 2)

        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(
            SCABlock(dim * 2, num_heads, LayerNorm_type=LayerNorm_type,
                     ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(self.num_blocks)
        )

        self.downsampling = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, bias=True)
        self.upsampling = PixelShuffleBlock(dim * 2, bias=bias)

    def forward(self, x):
        x = self.p_conv(x)

        h, w = x.shape[-2:]
        sp = self.sp_conv(x)
        # sp = self.patch_embed(sp)
        sp = F.adaptive_avg_pool2d(sp, (h // 4, w // 4))

        for i in range(self.num_blocks):
            x, sp = self.blocks[i](x, sp)

        x = self.downsampling(x)
        sp = self.upsampling(sp)
        x = torch.cat([x, sp], dim=1)

        return x


if __name__ == '__main__':
    x = torch.randn(3, 64, 256, 256).cuda()
    pixel = torch.randn(3, 64, 256, 256).cuda()
    model = QueryTransformer(dim=64, num_heads=8, num_blocks=2).cuda()
    sca = SCA(dim=64, num_heads=8, num_blocks=2, LayerNorm_type='WithBias', ffn_expansion_factor=2).cuda()
    x, pixel = model(x, pixel)

    x1 = sca(x)
    print(x1.shape)
    print(x.shape, pixel.shape)
