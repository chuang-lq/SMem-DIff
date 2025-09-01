import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from thop import profile
from .modules import *


def to_3d(x):
    if len(x.shape) == 5:
        return rearrange(x, 'b c t h w -> (b t) (h w) c')
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def to_5d(x, b, h, w):
    return rearrange(x, '(b t) (h w) c -> b c t h w', b=b, h=h, w=w)


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

        if len(x.shape) == 5:
            b = x.shape[0]
            return to_5d(self.body(to_3d(x)), b, h, w)

        return to_4d(self.body(to_3d(x)), h, w)


def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer,
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    c = feat_in.size(1)
    b, _, h, w = kernel.size()

    pad = (ksize - 1) // 2
    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(b, h, w, c, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(b, h, w, c, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(b, h, w, c, -1)
    kernel = F.softmax(kernel, dim=-1)

    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()

    return feat_out


class DynamicFiltering(torch.nn.Module):
    # dynamic filtering for alignment
    def __init__(self, dim, kernel_size, bias):
        super(DynamicFiltering, self).__init__()

        self.kernel_size = kernel_size
        self.kernel_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, kernel_size ** 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1, bias=bias)
        )

    def kernel_normalize(self, kernel):
        # kernel: [B, C, H, W, T*k*k]
        K = kernel.shape[-1]
        kernel = kernel - torch.mean(kernel, dim=-1, keepdim=True)
        kernel = kernel + 1.0 / K

        return kernel

    def forward(self, x):
        # x: [B, C, T, H, W]
        # kernel: [B, k*k, T, H, W]
        # return: [B, C, H, W]

        b, c, t, h, w = x.shape
        kernel = self.kernel_conv(x)
        kernel = rearrange(kernel, 'b (k1 k2) t h w -> b h w (t k1 k2)',
                           k1=self.kernel_size, k2=self.kernel_size)
        kernel = kernel.unsqueeze(dim=1)  # [B, 1, H, W, (T*k*k)]
        kernel = self.kernel_normalize(kernel)

        num_pad = (self.kernel_size - 1) // 2
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = F.pad(x, (num_pad, num_pad, num_pad, num_pad), mode="replicate")
        x = F.unfold(x, [self.kernel_size, self.kernel_size], padding=0)
        x = rearrange(x, '(b t) (c k1 k2) (h w) -> b c h w (t k1 k2)', b=b, t=t, c=c, k1=self.kernel_size,
                      k2=self.kernel_size, h=h, w=w)  # [B, C, H, W, (T*k*k)]

        x = torch.sum(x * kernel, dim=-1)  # [B, C, H, W]

        return x


class Xvo_shift(nn.Module):
    def __init__(self, dim, shift=4):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.shift = Shift(dim, shift=shift)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = self.act(self.conv1(x))
        res = self.shift(res) * res
        res = self.conv2(res)
        x = x + res

        return x


class DynamicDownsampling(torch.nn.Module):
    # dynamic downsampling for alignment
    def __init__(self, dim, kernel_size, stride, bias, shift_size=4):
        super(DynamicDownsampling, self).__init__()
        # stride = downsampling scale factor
        self.kernel_size = kernel_size
        self.stride = stride

        self.xvo_shift = Xvo_shift(dim=dim, shift=shift_size)
        self.kernel_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, padding=0, stride=1, bias=bias),
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, kernel_size ** 2, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                      stride=(1, stride, stride), bias=bias)
        )

    def kernel_normalize(self, kernel):
        # kernel: [B, H, W, T*k*k]
        return F.softmax(kernel, dim=-1)

    def forward(self, x):
        # x: [B, C, T, H*stride, W*stride]
        # kernel: [B, k*k, T, H, W]
        # return: [B, C, H, W]

        b = x.shape[0]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.xvo_shift(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', b=b)
        kernel = self.kernel_conv(x)

        _, _, t, h, w = kernel.shape
        kernel = kernel.permute(0, 3, 4, 2, 1).contiguous()  # [B, H, W, T, k*k]
        kernel = kernel.view(b, h, w, t * self.kernel_size * self.kernel_size)  # [B, H, W, T*k*k]
        kernel = self.kernel_normalize(kernel)

        kernel = kernel.unsqueeze(dim=1)  # [B, 1, H, W, T*k*k]

        num_pad = (self.kernel_size - self.stride + 1) // 2
        x = F.pad(x, (num_pad, num_pad, num_pad, num_pad, 0, 0), mode="replicate")
        x = x.unfold(3, self.kernel_size, self.stride)
        x = x.unfold(4, self.kernel_size, self.stride)  # [B, C, T, H, W, k, k]
        x = x.permute(0, 1, 3, 4, 2, 5, 6).contiguous()  # [B, C, H, W, T, k, k]
        x = x.view(b, -1, h, w, t * self.kernel_size * self.kernel_size)  # [B, C, H, W, T*k*k]

        x = x * kernel
        x = torch.sum(x, -1)  # [B, C, H, W]

        return x


class GDFN(nn.Module):
    # Gated-Dconv Feed-Forward Network (GDFN)
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(GDFN, self).__init__()

        hidden_dim = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GDFN3D(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(GDFN3D, self).__init__()

        hidden_dim = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_dim * 2, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dwconv = nn.Conv3d(hidden_dim * 2, hidden_dim * 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                groups=hidden_dim * 2, bias=bias)
        self.project_out = nn.Conv3d(hidden_dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        # b c t h w
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


def local_partition(x, h_step, w_step, dh, dw):
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
    local_x = torch.stack(local_x, dim=0)
    local_x = local_x.permute(1, 0, 2, 3, 4).contiguous()  # b, num_patches, c, dh, dw

    return local_x


def local_reverse(local_x, x, h_step, w_step, dh, dw):
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

    x_output = torch.nan_to_num(x_output, nan=1e-5, posinf=1e-5, neginf=1e-5)
    return x_output


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()

        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

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

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)

        out = self.project_out(out)
        return out


class SlidingChannelAttn(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.q_dwconv = nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), groups=dim, bias=bias)
        self.kv = nn.Conv3d(dim, dim * 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
        self.kv_dwconv = nn.Conv3d(dim * 2, dim * 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), groups=dim * 2,
                                   bias=bias)

        self.proj = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)

    def forward(self, x):
        # b c t h w
        b, c, t, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)

        q, k, v = map(lambda x: rearrange(x, 'b (head c) t h w -> (b t) head c (h w)', head=self.num_heads), (q, k, v))

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, '(b t) head c (h w) -> b (head c) t h w', b=b, h=h, w=w)

        out = self.proj(out)

        return out


class SlidingWindowAttn(nn.Module):
    def __init__(self, dim, num_heads, bias, patch_size, filter_size=5):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.patch_size = (patch_size, patch_size)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.avgFilter = DynamicFiltering(dim, kernel_size=filter_size, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, k_cached=None, v_cached=None):
        # b c h w
        b, c, h, w = x.shape
        ph, pw = self.patch_size

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        k_, v_ = k.unsqueeze(2), v.unsqueeze(2)
        if k_cached is not None and v_cached is not None:
            k_ = torch.cat([k_cached, k.unsqueeze(2)], dim=2)  # [b, c, t, h, w]
            v_ = torch.cat([v_cached, k.unsqueeze(2)], dim=2)
            k = self.avgFilter(k_)
            v = self.avgFilter(v_)

        q = local_partition(q, h - ph // 4, w - pw // 4, ph, pw)  # [b, num_patches, c, ph, pw]
        k = local_partition(k, h - ph // 4, w - pw // 4, ph, pw)
        v = local_partition(v, h - ph // 4, w - pw // 4, ph, pw)
        q = rearrange(q, 'b np (head c) ph pw -> (b np) head (ph pw) c', head=self.num_heads)
        k = rearrange(k, 'b np (head c) ph pw -> (b np) head (ph pw) c', head=self.num_heads)
        v = rearrange(v, 'b np (head c) ph pw -> (b np) head (ph pw) c', head=self.num_heads)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, '(b np) head (ph pw) c -> b np (head c) ph pw',
                        b=b, ph=ph, pw=pw)
        out = local_reverse(out, x, h - ph // 4, w - pw // 4, ph, pw)  # b, c, h, w
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)

        out = self.project_out(out)
        return out, k_[:, :, -2:, :, :], v_[:, :, -2:, :, :]


class FrameMemAttn(nn.Module):
    def __init__(self, dim, num_heads, bias, num_frames_tocache=2):
        """
        Initializes the FrameHistoryRouter module.

        Args:
            dim (int): The input dimension.
            num_heads (int): Number of attention heads.
            bias (bool): Whether to use bias in convolution layers.
            num_frames_tocache (int): Number of frames to cache for attention computation.
        """
        super(FrameMemAttn, self).__init__()
        self.dim = dim
        self.bias = bias

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.num_frames_tocache = num_frames_tocache

    def forward(self, x, k_cached=None, v_cached=None):
        """
        Forward pass of the FrameHistoryRouter.
        Given teh history states, it aggregates critical features for the restoration of the input frame

        Args:
            x (Tensor): Input tensor of shape (batch, channels, height, width).
            k_cached (Tensor, optional): Cached key tensor from previous frames.
            v_cached (Tensor, optional): Cached value tensor from previous frames.

        Returns:
            Tuple: Output tensor, and updated cached key and value tensors.
        """
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Concatenate cached key and value tensors if provided
        # Keys and values are concatenated with historical (cached) frames
        # This allows the attention mechanism to consider both the current and past frames.
        if k_cached is not None and v_cached is not None:
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)

        # Calculating Attention scores
        # Query is from the current frame, while key and value are from both current and cached frames
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        num_cache_to_keep = int(self.num_frames_tocache * c / self.num_heads)
        # k/v: [b, num_heads, c * num_frames_tocache, hw]
        return out, k[:, :, -num_cache_to_keep:, :], v[:, :, -num_cache_to_keep:, :]


class MemAttnBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 bias,
                 patch_size,
                 mlp_ratio=2.66,
                 attn_type='SlidingWindowAttn',
                 layerNorm_type='WithBias',
                 filter_size=5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = LayerNorm(dim, layerNorm_type)
        if attn_type == 'SlidingWindowAttn':
            self.attn = SlidingWindowAttn(dim, num_heads=num_heads, bias=bias,
                                          patch_size=patch_size, filter_size=filter_size)
        elif attn_type == 'FrameMemAttn':
            self.attn = FrameMemAttn(dim, num_heads=num_heads, bias=bias, num_frames_tocache=2)
        else:
            print(attn_type, " Not defined")
            exit()

        self.norm2 = LayerNorm(dim, layerNorm_type)
        self.mlp = GDFN(dim, ffn_expansion_factor=mlp_ratio, bias=bias)

    def forward(self, x, k_cached=None, v_cached=None):
        shortcut = x

        x, k_to_cache, v_to_cache = self.attn(self.norm1(x), k_cached=k_cached, v_cached=v_cached)
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))

        return x, k_to_cache, v_to_cache


class AttnBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 bias,
                 mlp_ratio=2.66,
                 layerNorm_type='WithBias'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = LayerNorm(dim, layerNorm_type)
        self.attn = ChannelAttention(dim, num_heads=num_heads, bias=bias)

        self.norm2 = LayerNorm(dim, layerNorm_type)
        self.mlp = GDFN(dim, ffn_expansion_factor=mlp_ratio, bias=bias)

    def forward(self, x):
        # b c h w
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class Attn3DBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 bias,
                 mlp_ratio=2.66,
                 layerNorm_type='WithBias'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = LayerNorm(dim, layerNorm_type)
        self.attn = SlidingChannelAttn(dim, num_heads=num_heads, bias=bias)

        self.norm2 = LayerNorm(dim, layerNorm_type)
        self.mlp = GDFN3D(dim, ffn_expansion_factor=mlp_ratio, bias=bias)

    def forward(self, x):
        # b c t h w
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class LevelBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, num_blocks, layerNorm_type='WithBias'):
        super(LevelBlock, self).__init__()

        self.num_blocks = num_blocks

        Block_list = []
        for _ in range(num_blocks):
            Block_list.append(Attn3DBlock(dim=dim, num_heads=num_heads,
                                          bias=bias, mlp_ratio=ffn_expansion_factor,
                                          layerNorm_type=layerNorm_type))

        self.transformer_blocks = nn.ModuleList(Block_list)

    def forward(self, x):
        # b c t h w
        for i in range(self.num_blocks):
            x = self.transformer_blocks[i](x)

        return x


class MemBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, num_blocks, layerNorm_type='WithBias',
                 attn_type='SlidingWindowAttn', patch_size=8, filter_size=5):
        super(MemBlock, self).__init__()

        self.num_blocks = num_blocks

        Block_list = []
        for _ in range(num_blocks):
            Block_list.append(AttnBlock(dim=dim, num_heads=num_heads,
                                        bias=bias, mlp_ratio=ffn_expansion_factor,
                                        layerNorm_type=layerNorm_type))

        self.transformer_blocks = nn.ModuleList(Block_list)

        self.memattn = MemAttnBlock(dim=dim, num_heads=num_heads,
                                    bias=bias, patch_size=patch_size,
                                    mlp_ratio=ffn_expansion_factor,
                                    attn_type=attn_type,
                                    layerNorm_type=layerNorm_type,
                                    filter_size=filter_size)

    def forward(self, x, k_cached=None, v_cached=None):
        # b c h w
        for i in range(self.num_blocks):
            x = self.transformer_blocks[i](x)

        x, k_to_cache, v_to_cache = self.memattn(x, k_cached=k_cached, v_cached=v_cached)

        return x, k_to_cache, v_to_cache


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, num_blocks, layerNorm_type='WithBias',
                 attn_type='SlidingWindowAttn', patch_size=8, filter_size=5):
        super(TransformerBlock, self).__init__()

        self.num_blocks = num_blocks

        Block_list = []
        for _ in range(num_blocks):
            Block_list.append(AttnBlock(dim=dim, num_heads=num_heads,
                                        bias=bias, mlp_ratio=ffn_expansion_factor,
                                        layerNorm_type=layerNorm_type))

        self.transformer_blocks = nn.ModuleList(Block_list)

        self.memattn = MemAttnBlock(dim=dim, num_heads=num_heads,
                                    bias=bias, patch_size=patch_size,
                                    mlp_ratio=ffn_expansion_factor,
                                    attn_type=attn_type,
                                    layerNorm_type=layerNorm_type,
                                    filter_size=filter_size)

    def forward(self, x, k_cached=None, v_cached=None, reverse=True):
        # b c t h w
        t = x.size(2)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        for i in range(self.num_blocks):
            x = self.transformer_blocks[i](x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        xi_list = []
        if reverse:
            for i in range(t - 1, -1, -1):
                xi = x[:, :, i, :, :]
                xi, k_cached, k_cached = self.memattn(xi, k_cached=k_cached, v_cached=v_cached)
                xi_list.append(xi)
            xi_list = xi_list[::-1]
        else:
            for i in range(t):
                xi = x[:, :, i, :, :]
                xi, k_cached, k_cached = self.memattn(xi, k_cached=k_cached, v_cached=v_cached)
                xi_list.append(xi)

        x = torch.stack(xi_list, dim=2)  # b c t h w

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Encoder(nn.Module):
    def __init__(self, dim, enc_blocks=[2, 7, 3], num_heads=[2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False, layerNorm_type='WithBias', attn_type1='SlidingWindowAttn',
                 attn_type2='FrameMemAttn', patch_size=[32, 16, 8], filter_size=5):
        super(Encoder, self).__init__()

        self.encoder_level1 = LevelBlock(dim=dim, num_heads=num_heads[0],
                                         ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, num_blocks=enc_blocks[0],
                                         layerNorm_type=layerNorm_type)

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = TransformerBlock(dim=int(dim * 2 ** 1), num_heads=num_heads[1],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, num_blocks=enc_blocks[1],
                                               layerNorm_type=layerNorm_type,
                                               attn_type=attn_type1,
                                               patch_size=patch_size[1],
                                               filter_size=filter_size)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  # From Level 2 to Level 3
        self.encoder_level3 = TransformerBlock(dim=int(dim * 2 ** 2), num_heads=num_heads[2],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, num_blocks=enc_blocks[2],
                                               layerNorm_type=layerNorm_type,
                                               attn_type=attn_type2,
                                               patch_size=patch_size[2],
                                               filter_size=filter_size)

    def forward(self, x):
        # b c t h w
        b, c, t, h, w = x.shape
        enc1 = self.encoder_level1(x)

        x = self.down1_2(rearrange(enc1, 'b c t h w -> (b t) c h w'))
        x = rearrange(x, '(b t) c h w -> b c t h w', b=b)
        enc2 = self.encoder_level2(x)

        x = self.down2_3(rearrange(enc2, 'b c t h w -> (b t) c h w'))
        x = rearrange(x, '(b t) c h w -> b c t h w', b=b)
        enc3 = self.encoder_level3(x)

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, dim, dec_blocks=[2, 7, 3], num_heads=[2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False, layerNorm_type='WithBias', attn_type1='SlidingWindowAttn',
                 attn_type2='FrameMemAttn', patch_size=[32, 16, 8], filter_size=5):
        super(Decoder, self).__init__()

        self.up2_1 = Upsample(int(dim * 2 ** 1))  # From Level 2 to Level 1
        self.reduce_chan_level1 = nn.Conv3d(int(dim * 2 ** 1), int(dim * 1), kernel_size=1,
                                            stride=1, padding=0, bias=bias)
        self.decoder_level1 = LevelBlock(dim=dim, num_heads=num_heads[0],
                                         ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, num_blocks=dec_blocks[0],
                                         layerNorm_type=layerNorm_type)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1,
                                            stride=1, padding=0, bias=bias)
        self.decoder_level2 = TransformerBlock(dim=int(dim * 2 ** 1), num_heads=num_heads[1],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, num_blocks=dec_blocks[1],
                                               layerNorm_type=layerNorm_type,
                                               attn_type=attn_type1,
                                               patch_size=patch_size[1],
                                               filter_size=filter_size)

        self.decoder_level3 = TransformerBlock(dim=int(dim * 2 ** 2), num_heads=num_heads[2],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, num_blocks=dec_blocks[2],
                                               layerNorm_type=layerNorm_type,
                                               attn_type=attn_type2,
                                               patch_size=patch_size[2],
                                               filter_size=filter_size)

    def forward(self, encs):
        enc1, enc2, enc3 = encs
        b, c, t, h, w = enc1.shape

        dec3 = self.decoder_level3(enc3, reverse=False)

        dec2 = self.up3_2(rearrange(dec3, 'b c t h w -> (b t) c h w'))
        dec2 = rearrange(dec2, '(b t) c h w -> b c t h w', b=b)
        dec2 = torch.cat([dec2, enc2], 1)
        dec2 = self.reduce_chan_level2(dec2)
        dec2 = self.decoder_level2(dec2, reverse=False)

        dec1 = self.up2_1(rearrange(dec2, 'b c t h w -> (b t) c h w'))
        dec1 = rearrange(dec1, '(b t) c h w -> b c t h w', b=b)
        dec1 = torch.cat([dec1, enc1], 1)
        dec1 = self.reduce_chan_level1(dec1)
        dec1 = self.decoder_level1(dec1)

        return dec1


class Model(nn.Module):

    def __init__(self, para):

        super(Model, self).__init__()
        self.n_feats = para.mid_channels
        self.num_blocks = para.num_blocks
        self.bias = para.bias

        # feature extraction module
        self.feat_extract = nn.Sequential(
            nn.Conv3d(3, self.n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.n_feats, self.n_feats, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                      padding=(0, 1, 1), bias=True)
        )

        # propagation branches
        self.ddf = nn.ModuleDict()
        self.ddf_enc = nn.ModuleDict()
        modules = ['backward_', 'forward_']
        for i, module in enumerate(modules):
            self.ddf[module] = DynamicDownsampling(self.n_feats, kernel_size=5, stride=1, bias=True)
            self.ddf_enc[module] = ResidualBlocksWithInputConv(
                (2 + i) * self.n_feats, self.n_feats, self.num_blocks)

        # backbone
        self.encoder_level1 = LevelBlock(dim=self.n_feats, num_heads=2,
                                         ffn_expansion_factor=2.66,
                                         bias=False, num_blocks=2,
                                         layerNorm_type='WithBias')

        self.down1_2 = Downsample(self.n_feats)  # From Level 1 to Level 2

        self.encoder_level2 = MemBlock(dim=int(self.n_feats * 2 ** 1), num_heads=4,
                                       ffn_expansion_factor=2.66,
                                       bias=False, num_blocks=5,
                                       layerNorm_type='WithBias',
                                       attn_type='SlidingWindowAttn',
                                       patch_size=16,
                                       filter_size=5)

        self.down2_3 = Downsample(int(self.n_feats * 2 ** 1))  # From Level 2 to Level 3

        self.encoder_level3 = MemBlock(dim=int(self.n_feats * 2 ** 2), num_heads=8,
                                       ffn_expansion_factor=2.66,
                                       bias=False, num_blocks=3,
                                       layerNorm_type='WithBias',
                                       attn_type='FrameMemAttn',
                                       patch_size=8,
                                       filter_size=5)

        self.decoder_level1 = LevelBlock(dim=self.n_feats, num_heads=2,
                                         ffn_expansion_factor=2.66,
                                         bias=False, num_blocks=2,
                                         layerNorm_type='WithBias')

        self.up2_1 = Upsample(int(self.n_feats * 2 ** 1))  # From Level 2 to Level 1
        self.reduce_chan_level1 = nn.Conv2d(int(self.n_feats * 2 ** 1), int(self.n_feats * 1), kernel_size=1,
                                            stride=1, padding=0, bias=False)

        self.decoder_level2 = MemBlock(dim=int(self.n_feats * 2 ** 1), num_heads=4,
                                       ffn_expansion_factor=2.66,
                                       bias=False, num_blocks=5,
                                       layerNorm_type='WithBias',
                                       attn_type='SlidingWindowAttn',
                                       patch_size=16,
                                       filter_size=5)

        self.up3_2 = Upsample(int(self.n_feats * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(self.n_feats * 2 ** 2), int(self.n_feats * 2 ** 1), kernel_size=1,
                                            stride=1, padding=0, bias=False)

        self.decoder_level3 = MemBlock(dim=int(self.n_feats * 2 ** 2), num_heads=8,
                                       ffn_expansion_factor=2.66,
                                       bias=False, num_blocks=3,
                                       layerNorm_type='WithBias',
                                       attn_type='FrameMemAttn',
                                       patch_size=8,
                                       filter_size=5)

        # reconstruction
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Converse2D(self.n_feats, self.n_feats, 3, 2, 2, 'replicate', 1e-5),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.n_feats, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, lqs, profile_flag=False):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output high quality (HR) sequence with shape (n, t, c, h, w).
        """

        if profile_flag:
            return self.profile_forward(lqs)

        b, t, c, h, w = lqs.size()

        # compute spatial features
        feats = self.feat_extract(rearrange(lqs, 'b t c h w -> b c t h w'))

        feat_prop = None
        feat_list = []

        for i in range(t - 1, -1, -1):
            feat_current = feats[:, :, i]

            # second-order backward alignment
            if i < t - 1:
                if i < t - 2:  # second-order features
                    # current + 2, current + 1 -> current
                    feat_prop = torch.stack([feat_current, feat_prop, feat_list[-2]], dim=2)
                else:  # current + 1 -> current
                    feat_prop = torch.stack([feat_current, feat_prop], dim=2)
            else:
                feat_prop = feat_current.unsqueeze(2)

            # shift-guided dynamic filtering
            feat_prop = self.ddf['backward_'](feat_prop)

            feat_prop = torch.cat([feat_current, feat_prop], dim=1)
            feat_prop = self.ddf_enc['backward_'](feat_prop)

            feat_list.append(feat_prop)

        feat_list = feat_list[::-1]
        for i in range(0, t):
            feat_current = feats[:, :, i]
            feati = feat_list[i]

            # second-order forward alignment
            if i > 0:
                if i > 1:  # second-order features
                    # current - 2, current - 1 -> current
                    feat_prop = torch.stack([feat_list[i - 2], feat_prop, feati], dim=2)
                else:  # current - 1 -> current
                    feat_prop = torch.stack([feat_prop, feati], dim=2)
            else:
                feat_prop = feati.unsqueeze(2)

            # shift-guided dynamic filtering
            feat_prop = self.ddf['forward_'](feat_prop)

            feat_prop = torch.cat([feat_current, feati, feat_prop], dim=1)
            feat_prop = self.ddf_enc['forward_'](feat_prop)

            feat_list[i] = feat_prop

        feats = torch.stack(feat_list, dim=2)  # [b, c, t, h, w]

        k_cached = [None] * 4
        v_cached = [None] * 4
        enc2 = []
        enc3 = []
        decs_list = []

        enc1 = self.encoder_level1(feats)

        for i in range(t - 1, -1, -1):
            feati = self.down1_2(enc1[:, :, i, :, :])
            feati, k_to_cache, v_to_cache = self.encoder_level2(feati, k_cached=k_cached[0], v_cached=v_cached[0])
            enc2.append(feati)
            k_cached[0] = k_to_cache
            v_cached[0] = v_to_cache
            feati = self.down2_3(feati)
            feati, k_to_cache, v_to_cache = self.encoder_level3(feati, k_cached=k_cached[1], v_cached=v_cached[1])
            enc3.append(feati)
            k_cached[1] = k_to_cache
            v_cached[1] = v_to_cache
        enc2 = enc2[::-1]
        enc3 = enc3[::-1]

        for i in range(t):
            feati = enc3[i]
            feati, k_to_cache, v_to_cache = self.decoder_level3(feati, k_cached=k_cached[2], v_cached=v_cached[2])
            k_cached[2] = k_to_cache
            v_cached[2] = v_to_cache
            feati = self.up3_2(feati)
            feati = torch.cat([enc2[i], feati], dim=1)
            feati = self.reduce_chan_level2(feati)
            feati, k_to_cache, v_to_cache = self.decoder_level2(feati, k_cached=k_cached[3], v_cached=v_cached[3])
            k_cached[3] = k_to_cache
            v_cached[3] = v_to_cache
            feati = self.up2_1(feati)
            feati = torch.cat([enc1[:, :, i, :, :], feati], dim=1)
            feati = self.reduce_chan_level1(feati)
            decs_list.append(feati)

        feats = torch.stack(decs_list, dim=2)
        feats = self.decoder_level1(feats)

        outs = []
        for i in range(0, t):
            feati = feats[:, :, i, :, :]
            feati = self.upsampler(feati)
            feati += lqs[:, i]
            outs.append(feati)

        outs = torch.stack(outs, dim=1)
        return outs

    def profile_forward(self, inputs):
        return self.forward(inputs)


def cost_profile(model, H, W, seq_length=5):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=15):
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
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


if __name__ == '__main__':
    x = torch.randn(1, 3, 64, 64)

    x_ = local_partition(x, 16 - 16//4, 16 - 16//4, 16, 16)
    x = local_reverse(x, x_, 16 - 16//4, 16 - 16//4, 16, 16)
    print(x.shape)
