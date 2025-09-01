import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from .gshift_arch import Encoder_shift_block
from .HWDownsample import HaarDownsampling


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


# Layer Norm
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


# w/o shape
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)


class DynamicDWConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block1 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, groups=channels)
                  for _ in range(3)]
        Block2 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, groups=channels)
                  for _ in range(3)]
        self.tokernel = nn.Conv2d(channels, kernel_size ** 2 * self.channels, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.Block1 = nn.Sequential(*Block1)
        self.Block2 = nn.Sequential(*Block2)

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.tokernel(self.pool(self.Block2(self.maxpool(self.Block1(self.avgpool(x))))))
        weight = weight.view(b * self.channels, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride,
                     padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden_features * 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                                    bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                                     bias=bias)
        self.kerner_conv_channel = DynamicDWConv(hidden_features, 3, 1, hidden_features)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        b = x1.shape[0]
        x1 = rearrange(self.kerner_conv_channel(rearrange(x1, 'b c t h w -> (b t) c h w')), '(b t) c h w -> b c t h w',
                       b=b)
        x = x1 * x2
        x = self.project_out(x)
        return x


class CWGDN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type):
        super(CWGDN, self).__init__()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        b = x.shape[0]
        identity = x
        x = rearrange(self.norm2(rearrange(x, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b t c h w', b=b)
        x = rearrange(self.ffn(rearrange(x, 'b t c h w -> b c t h w')), 'b c t h w -> b t c h w')
        return x + identity


class CWGDN1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type='BiasFree'):
        super(CWGDN1, self).__init__()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.wave = HaarDownsampling(dim)
        self.x_wave_conv1 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.x_wave_conv2 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)
        self.encoder_level1 = Encoder_shift_block(dim, 3, reduction=4, bias=False)
        self.encoder_level1_1 = Encoder_shift_block(dim, 3, reduction=4, bias=False)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        b = x.shape[0]
        identity = x
        x = self.norm2(rearrange(x, 'b t c h w -> (b t) c h w'))
        tf_wave_l, tf_wave_h = self.wave(x)
        tf_wave_h = self.x_wave_conv2(self.lrelu(self.x_wave_conv1(tf_wave_h)))
        tf_wave_l = self.encoder_level1_1(self.encoder_level1(tf_wave_l), reverse=1)
        x = rearrange(self.wave(torch.cat([tf_wave_l, tf_wave_h], dim=1), rev=True),
                      '(b t) c h w -> b t c h w', b=b)  # 上采样
        x = rearrange(self.ffn(rearrange(x, 'b t c h w -> b c t h w')),
                      'b c t h w -> b t c h w')

        return x + identity


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, embed_dim, group):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        self.group = group

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # prior
        if self.group == 1:
            self.kernel = nn.Linear(embed_dim * 4, dim * 2, bias=bias)

    def forward(self, x, prior=None):
        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            x = x * kv1 + kv2

        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FSAS(nn.Module):
    def __init__(self, dim, bias, embed_dim, group):
        super(FSAS, self).__init__()

        self.patch_size = 8
        self.group = group

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        # prior
        if self.group == 1:
            self.kernel = nn.Linear(embed_dim * 4, dim * 2, bias=bias)

    def local_partition(self, x, h_step, w_step, dh, dw):
        b, c, h, w = x.shape
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
                local_x.append(x[:, :, top:down, left:right])
        local_x = torch.stack(local_x, dim=1)  # b, num_patches, c, dh, dw
        return local_x

    def local_reverse(self, x, local_x, h_step, w_step, dh, dw):
        b, c, h, w = x.shape
        x_output = torch.zeros_like(x).to(x.device)
        count = torch.zeros((b, h, w), device=x.device)

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
                x_output[:, :, top:down, left:right] += local_x[:, index]
                count[:, top:down, left:right] += 1
                index += 1
        x_output = x_output / count.unsqueeze(1)  # b, c, h, w
        return x_output

    def forward(self, x, prior=None):
        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            x = x * kv1 + kv2

        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


# Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, embed_dim, group):
        super(GDFN, self).__init__()

        self.group = group
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # prior
        if self.group == 1:
            self.kernel = nn.Linear(embed_dim * 4, dim * 2, bias=bias)

    def forward(self, x, prior=None):
        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            x = x * kv1 + kv2

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, group):
        super(MDTA, self).__init__()

        self.group = group
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # prior
        if self.group == 1:
            self.kernel = nn.Linear(embed_dim * 4, dim * 2, bias=bias)

    def forward(self, x, prior=None):
        b, c, h, w = x.shape
        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            x = x * kv1 + kv2

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = MDTA(dim, num_heads, bias, embed_dim, group)
        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = GDFN(dim, ffn_expansion_factor, bias, embed_dim, group)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn2 = FSAS(dim, bias, embed_dim, group)
        self.norm2_1 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = DFFN(dim, ffn_expansion_factor, bias, embed_dim, group)

    def forward(self, x, prior=None):
        x = x + self.attn1(self.norm1(x), prior)
        x = x + self.ffn1(self.norm1_1(x), prior)

        x = x + self.attn2(self.norm2(x), prior)
        x = x + self.ffn2(self.norm2_1(x), prior)

        return x


# Hierarchical Integration Module
class HIM(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type):
        super(HIM, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2 = LayerNorm_Without_Shape(embed_dim * 4, LayerNorm_type)

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


# 论文：DAU-Net: Dual attention-aided U-Net for segmenting tumor in breast ultrasound images
class SWA(nn.Module):
    def __init__(self, in_channels, n_heads=8, window_size=8):
        super(SWA, self).__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.window_size = window_size

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        padded_x = F.pad(x,
                         [self.window_size // 2, self.window_size // 2, self.window_size // 2, self.window_size // 2],
                         mode='reflect')

        proj_query = self.query_conv(x).view(batch_size, self.n_heads, C // self.n_heads, height * width)
        proj_key = self.key_conv(padded_x).unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        proj_key = proj_key.permute(0, 1, 4, 5, 2, 3).contiguous().view(batch_size, self.n_heads, C // self.n_heads, -1)
        proj_value = self.value_conv(padded_x).unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        proj_value = proj_value.permute(0, 1, 4, 5, 2, 3).contiguous().view(batch_size, self.n_heads, C // self.n_heads,
                                                                            -1)

        energy = torch.matmul(proj_query.permute(0, 1, 3, 2), proj_key)
        attention = self.softmax(energy)

        out_window = torch.matmul(attention, proj_value.permute(0, 1, 3, 2))
        out_window = out_window.permute(0, 1, 3, 2).contiguous().view(batch_size, C, height, width)

        out = self.gamma * out_window + x
        return out


class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = x_1 * x_2

        x = self.linear2(x)

        return x
