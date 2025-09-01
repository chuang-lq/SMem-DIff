# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def conv(in_channels, out_channels, kernel_size, bias=True, groups=1, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=(stride, stride),
                     padding=(kernel_size // 2), bias=bias, groups=groups)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 use_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.use_conv = use_conv
        if use_conv:
            self.conv_layer = nn.Sequential(
                conv(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, groups=hidden_features),
                nn.GELU()
            )

    def forward(self, x, x_size):
        # x: B, H * W, C
        # x_size: H, W
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.use_conv:
            x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W).contiguous()
            x = self.conv_layer(x)
            x = rearrange(x, 'B C H W -> B (H W) C').contiguous()
        x = self.fc2(x)
        x = self.drop(x)
        # assert not torch.isnan(x).any(), "mlp contains NaN!"
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_conv=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 避免softmax的两端化
        self.use_conv = use_conv

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw], broadcast广播机制
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 求相对位置索引: relative_position_index
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if self.use_conv:
            self.qkv = conv(in_channels=dim, out_channels=dim * 3, kernel_size=3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_conv:
            self.proj = conv(in_channels=dim, out_channels=dim, kernel_size=3)
        else:
            self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_size, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        H, W = x_size
        if self.use_conv:
            assert self.window_size[0] == self.window_size[1]
            x = window_reverse(x, self.window_size[0], H, W)  # (B, H, W, C)
            x = rearrange(x, 'B H W C -> B C H W').contiguous()
            x = self.qkv(x)
            x = rearrange(x, 'B C H W -> B H W C').contiguous()
            x = window_partition(x, self.window_size[0])  # num_windows*B, w, w, C
            qkv = rearrange(x, 'B w1 w2 C -> B (w1 w2) C').contiguous()  # num_windows*B, w*w, C
        else:
            qkv = self.qkv(x)
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.use_conv:
            x = window_reverse(x, self.window_size[0], H, W)  # (B, H, W, C)
            x = rearrange(x, 'B H W C -> B C H W').contiguous()
            x = self.proj(x)
            x = rearrange(x, 'B C H W -> B H W C').contiguous()
            x = window_partition(x, self.window_size[0])  # num_windows*B, w, w, C
            x = rearrange(x, 'B w1 w2 C -> B (w1 w2) C').contiguous()  # num_windows*B, w*w, C
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_conv = use_conv
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_conv=use_conv)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       use_conv=use_conv)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0  # 给每个连续区域编号
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask, x_size=x_size)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device), x_size=x_size)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), x_size=x_size))
        # assert not torch.isnan(x).any(), "stb contains NaN!"
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # B H/2*W/2 2*C

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_conv=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_conv=use_conv)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        # x: B, H*W, C
        # x_size: H, W
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x, x_size)
        return x


class MRSTB(nn.Module):
    """Modified Residual Swin Transformer Block (MRSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_conv=False, ms=False):
        super(MRSTB, self).__init__()

        self.window_size = window_size

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         use_conv=use_conv)

        self.ms = ms
        if ms:
            self.fuse_layer = conv(in_channels=3 * dim, out_channels=dim, kernel_size=1)
        self.last_layer = conv(in_channels=dim, out_channels=dim, kernel_size=3)

    def forward(self, x, x_size):
        # x: B, H*W, C
        # x_size: H, W
        if self.ms:  # 多尺度
            H1, W1 = x_size
            # H2, W2 should be divisible by window_size
            H2, W2 = H1 // 2, W1 // 2
            H2, W2 = H2 - (H2 % self.window_size), W2 - (W2 % self.window_size)
            # H3, W3 should be divisible by window_size
            H3, W3 = H1 // 4, W1 // 4
            H3, W3 = H3 - (H3 % self.window_size), W3 - (W3 % self.window_size)

            x1 = rearrange(x, 'B (H W) C -> B C H W', H=H1, W=W1)
            x2 = F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x1, size=(H3, W3), mode='bilinear', align_corners=False)

            x1 = rearrange(x1, 'B C H W -> B (H W) C')
            res1 = self.residual_group(x1, x_size=(H1, W1))  # B, H1*W1, C
            res1 = rearrange(res1, 'B (H W) C -> B C H W', H=H1, W=W1)

            x2 = rearrange(x2, 'B C H W -> B (H W) C')
            res2 = self.residual_group(x2, x_size=(H2, W2))  # B, H2*W2, C
            res2 = rearrange(res2, 'B (H W) C -> B C H W', H=H2, W=W2)
            res2 = F.interpolate(res2, size=(H1, W1), mode='bilinear', align_corners=False)

            x3 = rearrange(x3, 'B C H W -> B (H W) C')
            res3 = self.residual_group(x3, x_size=(H3, W3))  # B, H3*W3, C
            res3 = rearrange(res3, 'B (H W) C -> B C H W', H=H3, W=W3)
            res3 = F.interpolate(res3, size=(H1, W1), mode='bilinear', align_corners=False)

            res = torch.cat([res1, res2, res3], dim=1)
            res = self.last_layer(self.fuse_layer(res))
            res = rearrange(res, 'B C H W -> B (H W) C')

            return x + res
        else:
            H, W = x_size
            res = self.residual_group(x, x_size)  # B, H*W, C
            res = rearrange(res, 'B (H W) C -> B C H W', H=H, W=W)
            res = self.last_layer(res)
            res = rearrange(res, 'B C H W -> B (H W) C')

            return x + res


class MRSTBs(nn.Module):

    # img_size: (list/tuple[int])
    def __init__(self, img_size, num_layers, dim, depth=4, num_heads=4, window_size=8,
                 mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, use_conv=False, ms=False):
        super(MRSTBs, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            MRSTB(dim, input_resolution=img_size,
                  depth=depth, num_heads=num_heads,
                  window_size=window_size, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, norm_layer=norm_layer,
                  use_checkpoint=False, use_conv=use_conv,
                  ms=ms)
            for _ in range(num_layers))

    def forward(self, x):
        # x: B, C, H, W
        H, W = x.shape[-2:]
        x_size = (H, W)
        x = rearrange(x, 'B C H W -> B (H W) C', H=H, W=W)
        for i in range(self.num_layers):
            x = self.layers[i](x, x_size)
        x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_conv=False):
        super(RSTB, self).__init__()
        self.window_size = window_size
        Ph = input_resolution[0] - input_resolution[0] % window_size
        Pw = input_resolution[1] - input_resolution[1] % window_size

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=(Ph, Pw),
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         use_conv=use_conv)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1)
        else:
            self.downsample = None
            self.last_layer = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)

    def forward(self, x, x_size):
        # x: B, H*W, C
        # x_size: H, W
        H, W = x_size

        # H2, W2 should be divisible by window_size
        H2, W2 = H - (H % self.window_size), W - (W % self.window_size)
        res = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)
        res = F.interpolate(res, size=(H2, W2), mode='bilinear', align_corners=False)
        res = rearrange(res, 'B C H W -> B (H W) C')

        res = self.residual_group(res, x_size=(H2, W2))  # B, H2*W2, C
        res = rearrange(res, 'B (H W) C -> B C H W', H=H2//2, W=W2//2)
        res = F.interpolate(res, size=(H//2, W//2), mode='bilinear', align_corners=False)
        res = self.last_layer(res)
        res = rearrange(res, 'B C H W -> B (H W) C')
        if self.downsample is not None:
            x = self.downsample(x, x_size)

        return x + res


class RSTB_DS(nn.Module):

    # img_size: (list/tuple[int])
    def __init__(self, img_size, dim, depth=4, window_size=8, qkv_bias=True,
                 norm_layer=nn.LayerNorm, use_conv=False):
        super(RSTB_DS, self).__init__()

        self.rstb1 = RSTB(dim=dim, input_resolution=(img_size[0] // 2, img_size[1] // 2),
                          depth=depth, num_heads=4,
                          window_size=window_size, mlp_ratio=4,
                          qkv_bias=qkv_bias, norm_layer=norm_layer,
                          downsample=PatchMerging, use_checkpoint=False,
                          use_conv=use_conv)

        self.rstb2 = RSTB(dim=dim * 2, input_resolution=(img_size[0] // 4, img_size[1] // 4),
                          depth=depth, num_heads=8,
                          window_size=window_size, mlp_ratio=2,
                          qkv_bias=qkv_bias, norm_layer=norm_layer,
                          downsample=PatchMerging, use_checkpoint=False,
                          use_conv=use_conv)

    def forward(self, x):
        # x: B, C, H, W
        H, W = x.shape[-2:]
        x = rearrange(x, 'B C H W -> B (H W) C', H=H, W=W)
        x = self.rstb1(x, x_size=(H, W))
        x = self.rstb2(x, x_size=(H // 2, W // 2))
        x = rearrange(x, 'B (H W) C -> B C H W', H=H // 4, W=W // 4)

        return x


if __name__ == '__main__':
    x = torch.randn(2, 64, 256, 256)
    y = torch.randn(2, 64, 320, 180)
    model1 = MRSTBs(img_size=[256, 256], num_layers=3, dim=64, depth=4, num_heads=4, window_size=8, mlp_ratio=4.,
                    norm_layer=nn.LayerNorm, ms=False)
    model2 = RSTB_DS(img_size=[320, 180], dim=64, depth=2, norm_layer=nn.LayerNorm, use_conv=True)
    print(model1(x).shape)
    print(model2(y).shape)
