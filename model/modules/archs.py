import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.modules.batchnorm import _BatchNorm
import math


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


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


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


class TFR(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(TFR, self).__init__()

        self.orb1 = CABs(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = CABs(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = CABs(n_feat, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = Upsample(int(n_feat*2**1))
        self.up_dec1 = Upsample(int(n_feat*2**1))

        self.up_enc2 = nn.Sequential(Upsample(int(n_feat*2**2)),
                                     Upsample(int(n_feat*2**1)))
        self.up_dec2 = nn.Sequential(Upsample(int(n_feat*2**2)),
                                     Upsample(int(n_feat*2**1)))

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


def clipped_softmax(tensor, dim=-1):
    # Create a mask for zero elements
    zero_mask = tensor == 0

    # Apply the mask to ignore zero elements in the softmax computation
    # Set zero elements to `-inf` so that they become 0 after softmax
    masked_tensor = tensor.masked_fill(zero_mask, float('-inf'))

    # Compute softmax on the modified tensor
    softmaxed = F.softmax(masked_tensor, dim=dim)

    # Zero out `-inf` elements (which are now 0 due to softmax) if any original zeros existed
    softmaxed = softmaxed.masked_fill(zero_mask, 0)

    non_zero_softmaxed_sum = softmaxed.sum(dim=dim, keepdim=True)
    normalized_softmaxed = softmaxed / non_zero_softmaxed_sum

    return normalized_softmaxed


class FrameHistoryRouter(nn.Module):
    def __init__(self, dim, num_heads, bias, num_frames_tocache=1, group=4):
        """
        Initializes the FrameHistoryRouter module.

        Args:
            dim (int): The input dimension.
            num_heads (int): Number of attention heads.
            bias (bool): Whether to use bias in convolution layers.
            num_frames_tocache (int): Number of frames to cache for attention computation.
        """
        super(FrameHistoryRouter, self).__init__()

        self.group = group
        self.dim = dim
        self.bias = bias

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.num_frames_tocache = num_frames_tocache

        if self.group == 1:
            self.kernel = nn.Linear(256, dim * 2, bias=False)

    def forward(self, x, prior=None, k_cached=None, v_cached=None):
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


class StateAlignBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, num_frames_tocache, Scale_patchsize=1, plot_attn=False):
        """
        Initializes the StateAlignBlock module.

        Args:
            dim (int): The input dimension.
            num_heads (int): Number of attention heads.
            bias (bool): Whether to use bias in convolution layers.
            num_frames_tocache (int): Number of frames to cache for attention computation.
            Scale_patchsize (int): Scale patch size for windowing.
            plot_attn (bool): Whether to plot attention (used for visualization/debugging).
        """
        super(StateAlignBlock, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim * 1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.num_frames_tocache = num_frames_tocache
        self.window_size = 2 * Scale_patchsize

        self.k2 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.k2_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=self.window_size, stride=self.window_size, padding=1,
                                   groups=dim * 2, bias=bias)
        self.q2 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q2_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=self.window_size, stride=self.window_size, padding=1,
                                   groups=dim * 2, bias=bias)

    def zero_out_non_top_k(self, attn_matrix, k):
        """
        Zero out all but the top-k values in the attention matrix.

        Args:
            attn_matrix (Tensor): The attention matrix.
            k (int): Number of top elements to keep.

        Returns:
            Tensor: The modified attention matrix with only top-k values.
        """

        # Step 1: Get the top-k values and their indices for the last dimension
        a, n, b, c, c = attn_matrix.shape
        _, topk_indices = torch.topk(attn_matrix, k=k, dim=-1)

        # Step 2: Create a mask of zeros
        mask = torch.zeros_like(attn_matrix)

        # Use these indices with scatter_ to update the mask. This time correctly broadcasting
        mask.scatter_(dim=-1, index=topk_indices, value=1)

        return attn_matrix * mask

    def positionalencoding2d(self, d_model, height, width):
        """
        Generates a 2D positional encoding for attention mechanism.

        Args:
            d_model (int): The dimension of the model.
            height (int): Height of the position encoding grid.
            width (int): Width of the position encoding grid.

        Returns:
            Tensor: A 2D positional encoding matrix of shape (d_model, height, width).
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe

    def create_local_attention_mask(self, h, w, n):
        """
        Creates a local attention mask So that every patch can only attend to neighboring patches.

        Args:
            h (int): Height of the attention mask.
            w (int): Width of the attention mask.
            n (int): Local attention range.

        Returns:
            Tensor: A binary mask tensor that determines the local attention scope.
        """
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2)  # Shape: (hw, 2)
        distances = torch.cdist(coords.float(), coords.float(), p=1)  # Using L1 distance
        mask = distances <= n
        return mask

    # def forward(self, x, k_cached=None, v_cached=None):
    #     """
    #     Forward pass of the StateAlignBlock, which aligns the History states, Cached K,V of previous frames,
    #     with the current frame.

    #     Args:
    #         x (Tensor): Input tensor of shape (batch, channels, height, width).
    #         k_cached (Tensor, optional): Cached key tensor.
    #         v_cached (Tensor, optional): Cached value tensor.

    #     Returns:
    #         Tuple: Output tensor and cached key, value tensors.
    #     """
    #     b, c, h, w = x.shape
    #     head_dim = c//self.num_heads

    #     pos = self.positionalencoding2d(c, h, w)
    #     x_qk = x + pos.to(x.device)

    #     qk = self.qk_dwconv(self.qk(x_qk))
    #     q, k = qk.chunk(2, dim=1)
    #     v = self.v_dwconv(self.v(x))

    #     # Rearrange inputs into windows and split into multiple heads in one step
    #     q = rearrange(q, 'b (h_head d) (p1 h) (p2 w) -> b 1 h_head (h w) d p1 p2',
    #                   h_head=self.num_heads, p1=self.window_size, p2=self.window_size, d=head_dim)
    #     k = rearrange(k, 'b (h_head d) (p1 h) (p2 w) -> b 1 h_head (h w) d p1 p2',
    #                   h_head=self.num_heads, p1=self.window_size, p2=self.window_size, d=head_dim)
    #     v = rearrange(v, 'b (h_head d) (p1 h) (p2 w) -> b 1 h_head (h w) (p1 p2 d)',
    #                   h_head=self.num_heads, p1=self.window_size, p2=self.window_size, d=head_dim)

    #     n = q.shape[3]
    #     q = rearrange(q, 'b 1 h_head n d p1 p2 -> (b h_head n) d p1 p2')
    #     q = self.q2_dwconv(self.q2(q))
    #     q = rearrange(q, '(b h_head n) d 1 1 -> b 1 h_head n d', b=b, h_head=self.num_heads, n=n)

    #     k = rearrange(k, 'b 1 h_head n d p1 p2 -> (b h_head n) d p1 p2')
    #     k = self.k2_dwconv(self.k2(k))
    #     k = rearrange(k, '(b h_head n) d 1 1 -> b 1 h_head n d', b=b, h_head=self.num_heads, n=n)

    #     H, W = q.shape[2], q.shape[3]

    #     q = torch.nn.functional.normalize(q, dim=-1)
    #     k = torch.nn.functional.normalize(k, dim=-1)
    #     # Concatenate cached k and v if they exist
    #     if k_cached is not None and v_cached is not None:
    #         k = torch.cat([k_cached, k], dim=1)
    #         v = torch.cat([v_cached, v], dim=1)

    #     curr_num_frames = k.shape[1]
    #     attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature

    #     # Create local attention mask
    #     mask = self.create_local_attention_mask(H, W, 4)
    #     mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    #     mask = mask.to(attn.device)
    #     attn1 = mask * attn

    #     # Zero out non-top-k attention scores
    #     attn2 = self.zero_out_non_top_k(attn, 5)

    #     # Combine the two attention mechanisms and apply softmax
    #     attn = (attn1 + attn2)/2
    #     attn = clipped_softmax(attn)

    #     del q
    #     out = torch.matmul(attn, v)
    #     del attn

    #     out = rearrange(out, 'b curr_num_frames h_head (h w) (p1 p2 d) -> (b curr_num_frames) (h_head d) (p1 h) (p2 w)',
    #             h_head=self.num_heads, p1=self.window_size, p2=self.window_size,
    #             h=h//self.window_size, w=w//self.window_size, d=head_dim)

    #     out = self.project_out(out)
    #     out = rearrange(out, '(b curr_num_frames) c h w -> b curr_num_frames c h w',
    #                     b=b, curr_num_frames=curr_num_frames)

    #     return out, k[:, -self.num_frames_tocache:, :, :, :], v[:, -self.num_frames_tocache:, :, :, :]

    def forward(self, x, k_cached=None, v_cached=None):
        b, c, h, w = x.shape
        head_dim = c // self.num_heads

        # pos = self.positionalencoding2d(c, h, w)
        # x_qk = x + pos.to(x.device)

        qk = self.qk_dwconv(self.qk(x))
        q, k = qk.chunk(2, dim=1)
        v = self.v_dwconv(self.v(x))

        k = self.k2_dwconv(self.k2(k))  # b 2*c h//window_size w//window_size
        q = self.q2_dwconv(self.q2(q))
        H, W = q.shape[2], q.shape[3]

        # pos = self.positionalencoding2d(c*2, q.shape[2], q.shape[3])
        # q = q + pos.to(x.device)
        # k = k + pos.to(x.device)

        # Rearrange inputs into windows and split into multiple heads in one step
        q = rearrange(q, 'b (h_head d) h w -> b 1 h_head (h w) d',
                      h_head=self.num_heads, d=head_dim * 2)
        k = rearrange(k, 'b (h_head d) h w -> b 1 h_head (h w) d',
                      h_head=self.num_heads, d=head_dim * 2)
        v = rearrange(v, 'b (h_head d) (p1 h) (p2 w) -> b 1 h_head (h w) (p1 p2 d)',
                      h_head=self.num_heads, p1=self.window_size, p2=self.window_size, d=head_dim)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # Concatenate cached k and v if they exist
        if k_cached is not None and v_cached is not None:
            k = torch.cat([k_cached, k], dim=1)
            v = torch.cat([v_cached, v], dim=1)

        curr_num_frames = k.shape[1]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature

        # Zero out non-top-k attention scores
        attn1 = self.zero_out_non_top_k(attn, 5)

        # Create local attention mask
        mask = self.create_local_attention_mask(H, W, 4)
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask = mask.to(attn.device)
        attn2 = mask * attn

        # Combine the two attention mechanisms and apply softmax
        attn = attn1 + attn2
        attn = clipped_softmax(attn)

        del q
        out = torch.matmul(attn, v)
        del attn

        out = rearrange(out, 'b curr_num_frames h_head (h w) (p1 p2 d) -> (b curr_num_frames) (h_head d) (p1 h) (p2 w)',
                        h_head=self.num_heads, p1=self.window_size, p2=self.window_size,
                        h=h // self.window_size, w=w // self.window_size, d=head_dim)

        out = self.project_out(out)
        out = rearrange(out, '(b curr_num_frames) c h w -> b curr_num_frames c h w',
                        b=b, curr_num_frames=curr_num_frames)
        # k: [b, num_frames_tocache, num_heads, h//window_size * w//window_size, head_dim*2]
        # v: [b, num_frames_tocache, num_heads, np, (p1 p2 head_dim)]
        return out, k[:, -self.num_frames_tocache:, :, :, :], v[:, -self.num_frames_tocache:, :, :, :]


class CausalHistoryModel(nn.Module):
    def __init__(self, dim, num_heads, bias, scale_patchsize, num_frames_tocache=1, group=4):
        super(CausalHistoryModel, self).__init__()

        self.group = group

        # SAB
        self.spatial_aligner = StateAlignBlock(dim, num_heads, bias, num_frames_tocache,
                                               Scale_patchsize=scale_patchsize)

        # FHR
        self.ChanAttn = FrameHistoryRouter(dim, num_heads, bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.num_heads = num_heads

        if self.group == 1:
            self.kernel = nn.Linear(256, dim * 2, bias=False)

    def forward(self, x, prior=None, k_cached=None, v_cached=None):
        """
        Forward pass of the CausalHistoryModel.
        MotionCompensatedHistory = concat(SAB(History, Input), Input)
        Output = FHR(MotionCompensatedHistory, Input) + Input

        Args:
            x (Tensor): Input tensor of shape (batch, channels, height, width).
            k_cached (Tensor, optional): Cached key tensor from previous frames.
            v_cached (Tensor, optional): Cached value tensor from previous frames.

        Returns:
            Tuple: Output tensor from channel attention, and updated cached key and value tensors.
        """

        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            x = x * kv1 + kv2

        # Perform spatial alignment and get the updated cached key-value tensors
        x_spatial, k_tocache, v_tocache = self.spatial_aligner(x, k_cached, v_cached)

        cached_num_frames = x_spatial.shape[1]
        x_spatial = rearrange(x_spatial, 'b cached_num_frames c h w -> (b cached_num_frames) c h w')

        # Compute key and value embeddings of aligned history
        kv = self.kv_dwconv(self.kv(x_spatial))
        k, v = kv.chunk(2, dim=1)

        k = rearrange(k, '(b cached_num_frames) (head c) h w -> b head (cached_num_frames c) (h w)',
                      head=self.num_heads, cached_num_frames=cached_num_frames)
        v = rearrange(v, '(b cached_num_frames) (head c) h w -> b head (cached_num_frames c) (h w)',
                      head=self.num_heads, cached_num_frames=cached_num_frames)

        k = torch.nn.functional.normalize(k, dim=-1)

        # pass the input frame(x) and aligned history(k,v) to the Frame History router
        X_channel, _, _ = self.ChanAttn(x, k, v)

        return X_channel, k_tocache, v_tocache


# Gated Feed-Forward Network
class GatedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, group=4):
        super(GatedFeedForward, self).__init__()

        self.group = group
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, stride=1,
                                padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        if self.group == 1:
            self.kernel = nn.Linear(256, dim * 2, bias=bias)

    def forward(self, x, prior):
        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            x = x * kv1 + kv2

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# Feed_Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, FFN_Expand=2, drop_out_rate=0., group=4):
        super(FeedForward, self).__init__()

        self.group = group
        ffn_channel = FFN_Expand * dim
        self.conv4 = nn.Conv2d(in_channels=dim,
                               out_channels=ffn_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel,
                               out_channels=dim,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias=True)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        if self.group == 1:
            self.kernel = nn.Linear(256, dim * 2, bias=False)

    def forward(self, x, prior):
        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            x = x * kv1 + kv2

        x = self.conv4(x)
        x = F.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return x * self.gamma


# Transposed Self-Attention
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, group=4):
        super(ChannelAttention, self).__init__()

        self.group = group
        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        if self.group == 1:
            self.kernel = nn.Linear(256, dim * 2, bias=False)

    def forward(self, x, prior, k_cached=None, v_cached=None):
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
        return out, None, None


class ReducedAttn(nn.Module):
    def __init__(self, dim, DW_Expand=2.0, drop_out_rate=0., group=4):
        super().__init__()

        self.group = group
        dw_channel = int(dim * DW_Expand)
        self.conv1 = nn.Conv2d(in_channels=dim,
                               out_channels=dw_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias=True)

        self.conv2 = nn.Conv2d(in_channels=dw_channel,
                               out_channels=dw_channel,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=dw_channel,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=dw_channel,
                               out_channels=dim,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        if self.group == 1:
            self.kernel = nn.Linear(256, dim * 2, bias=False)

    def forward(self, x, prior, k_cached=None, v_cached=None):
        if prior is not None and self.group == 1:
            kv = self.kernel(prior).squeeze(dim=1).unsqueeze(-1).unsqueeze(-1)
            kv1, kv2 = kv.chunk(2, dim=1)
            x = x * kv1 + kv2

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        return x * self.beta, None, None


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


class AttnBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_heads=1, Scale_patchsize=1,
                 attention_type='channel', FFW_type="GFFW", num_frames_tocache=1, plot_attn=False, group=4):
        super(AttnBlock, self).__init__()
        """
        This class gives you the freedom to design custom transfomer blocks with teh abilty to 
        specify the attention type and Feed forward fucntion.

        Args:
            dim (int): Dimension of the input feature space.
            ffn_expansion_factor (int): Expansion factor for the Feed Forward Network (FFN).
            bias (bool): Whether to use bias in the layers.
            LayerNorm_type (str): Type of Layer Normalization to use.
            num_blocks (int): Number of TurtleAttnBlocks to be used in the LevelBlock.
            attn_type1 (str): Attention type for the initial blocks. Default is "Channel".
            attn_type2 (str): Attention type for the last block. Default is "CHM".
            FFW_type (str): Type of Feed Forward Network. Default is "GFFW".
            num_frames_tocache (int): Number of frames to cache for the attention mechanism. Default is 1.
            num_heads (int): Number of attention heads in each TurtleAttnBlock. Default is 1.
            Scale_patchsize (int): patch size for the CHM's spatial alignemnt. Default is 1.

        Supported Attention types:
            - `FHR`: Frame History Router for utilizing past information without spatial alignment.
            - `CHM`: Causal History Model for utilizing past information.
            - `Channel`: Channel Attention for feature refinement.
            - `ReducedAttn`: Uses convolutions and gating, replacing Channel Attention to reduce computational complexity.
            - `NoAttn`: Only applies feed-forward layers without any attention mechanism.

        Supported FeedForward Types:
            - `FFW`: Feed Forward.
            - `GFFW`: Gated Feed Forward.

        """
        self.norm1 = LayerNorm(dim, LayerNorm_type)

        if attention_type == "Channel":
            self.attn = ChannelAttention(dim, num_heads, bias, group=group)
        elif attention_type == "ReducedAttn":
            self.attn = ReducedAttn(dim, group=group)
        elif attention_type == "FHR":  # Caches num_frames_tocache
            self.attn = FrameHistoryRouter(dim, num_heads, bias, num_frames_tocache, group=group)
        elif attention_type == "CHM":  # Best march14
            self.attn = CausalHistoryModel(dim, num_heads, bias, Scale_patchsize, num_frames_tocache, group=group)
        elif attention_type == "NoAttn":
            self.attn = None
        else:
            print(attention_type, " Not defined")
            exit()

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        if FFW_type == "GFFW":
            self.ffn = GatedFeedForward(dim, ffn_expansion_factor, bias, group=group)
        elif FFW_type == "FFW":
            self.ffn = FeedForward(dim, group=group)
        else:
            print(FFW_type, " Not defined")
            exit()

    def forward(self, x, prior=None, k_cached=None, v_cached=None):
        if self.attn is None:
            return x + self.ffn(self.norm2(x), prior), None, None
        else:
            attn_out, k_tocahe, v_tocahe = self.attn(self.norm1(x), prior, k_cached, v_cached)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x), prior)
            return x, k_tocahe, v_tocahe


class LevelBlock(nn.Module):
    def __init__(self, dim, embed_dim, ffn_expansion_factor, bias, LayerNorm_type, num_blocks,
                 attn_type1="Channel", attn_type2="CHM", FFW_type="GFFW", num_frames_tocache=1, num_heads=1,
                 Scale_patchsize=1, group=4):
        super(LevelBlock, self).__init__()
        """
        Initializes multiple `TurtleAttnBlock` layers where the last blocks use `CHM` or 'FHR' 
        attention to handle historical dependencies, while the initial layers use a different
        attention type like `Channel` or `ReducedAttn`.

        Args:
            dim (int): Dimension of the input feature space.
            ffn_expansion_factor (int): Expansion factor for the Feed Forward Network (FFN).
            bias (bool): Whether to use bias in the layers.
            LayerNorm_type (str): Type of Layer Normalization to use.
            num_blocks (int): Number of TurtleAttnBlocks to be used in the LevelBlock.
            attn_type1 (str): Attention type for the initial blocks. Default is "Channel".
            attn_type2 (str): Attention type for the last block. Default is "CHM".
            FFW_type (str): Type of Feed Forward Network. Default is "GFFW".
            num_frames_tocache (int): Number of frames to cache for the attention mechanism. Default is 1.
            num_heads (int): Number of attention heads in each TurtleAttnBlock. Default is 1.
            Scale_patchsize (int): patch size for the CHM's spatial alignemnt. Default is 1.

        Attention type options used:
        - `FHR`: Frame History Router for utilizing past information without spatial alignment.
        - `CHM`: Causal History Model for utilizing past information.
        - `Channel`: Channel Attention for feature refinement.
        - `ReducedAttn`: Uses convolutions and gating, replacing Channel Attention to reduce computational complexity.
        - `NoAttn`: Only applies feed-forward layers without any attention mechanism.
        """

        self.group = group
        self.num_blocks = num_blocks
        Block_list = []

        for _ in range(num_blocks - 1):
            Block_list.append(AttnBlock(dim=dim, num_heads=num_heads,
                                        ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                        LayerNorm_type=LayerNorm_type, attention_type=attn_type1,
                                        FFW_type=FFW_type, num_frames_tocache=num_frames_tocache,
                                        Scale_patchsize=Scale_patchsize, group=group))

        Block_list.append(AttnBlock(dim=dim, num_heads=num_heads,
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                    LayerNorm_type=LayerNorm_type, attention_type=attn_type2, FFW_type=FFW_type,
                                    num_frames_tocache=num_frames_tocache,
                                    Scale_patchsize=Scale_patchsize, group=group))

        self.transformer_blocks = nn.ModuleList(Block_list)

        if self.group > 1:
            self.him = HIM(dim=dim, num_heads=num_heads, bias=bias, embed_dim=embed_dim, LayerNorm_type=LayerNorm_type)

    def forward(self, x, prior, k_cached=None, v_cached=None):

        if prior is not None and self.group > 1:
            x = self.him(x, prior)

        for i in range(self.num_blocks - 1):
            x, _, _ = self.transformer_blocks[i](x, prior)

        # Pass k_cached and v_cached to the last block
        out1, k_tocahe, v_tocahe = self.transformer_blocks[-1](x, prior, k_cached, v_cached)
        if k_tocahe != None:
            return out1, k_tocahe, v_tocahe
        else:
            return out1, None, None


class LatentCacheBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_blocks,
                 attn_type1="FHR", attn_type2="Channel", attn_type3="FHR", FFW_type="GFFW", num_frames_tocache=1,
                 num_heads=1, group=1):
        super(LatentCacheBlock, self).__init__()
        """
        Initializes multiple `TurtleAttnBlock` layers where the first and last blocks use `CHM` or 'FHR' 
        attention to handle historical dependencies, while the intermediate layers use a different
        attention type like `Channel` or `ReducedAttn`.

        Args:
            dim (int): Dimension of the input feature space.
            ffn_expansion_factor (int): Expansion factor for the Feed Forward Network (FFN).
            bias (bool): Whether to use bias in the layers.
            LayerNorm_type (str): Type of Layer Normalization to use.
            num_blocks (int): Number of TurtleAttnBlocks to be used in the LevelBlock.
            attn_type1 (str): Attention type for the latent middle blocks. Default is "Channel".
            attn_type2 (str): Attention type for the first block. Default is "CHM".
            attn_type2 (str): Attention type for the last block. Default is "CHM".
            FFW_type (str): Type of Feed Forward Network. Default is "GFFW".
            num_frames_tocache (int): Number of frames to cache for the attention mechanism. Default is 1.
            num_heads (int): Number of attention heads in each TurtleAttnBlock. Default is 1.
            Scale_patchsize (int): patch size for the CHM's spatial alignemnt. Default is 1.

        Attention type options used:
        - `FHR`: Frame History Router for utilizing past information without spatial alignment.
        - `CHM`: Causal History Model for utilizing past information.
        - `Channel`: Channel Attention for feature refinement.
        - `ReducedAttn`: Uses convolutions and gating, replacing Channel Attention to reduce computational complexity.
        - `NoAttn`: Only applies feed-forward layers without any attention mechanism.
        """
        self.num_blocks = num_blocks
        Block_list = []
        if self.num_blocks < 2:
            print("LatentCacheBlock should have more than 2 layers")
            exit()

        Block_list.append(AttnBlock(dim=dim, num_heads=num_heads,
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                    LayerNorm_type=LayerNorm_type, attention_type=attn_type1, FFW_type=FFW_type,
                                    num_frames_tocache=num_frames_tocache, plot_attn=True, group=group))

        if self.num_blocks > 2:
            for _ in range(self.num_blocks - 2):
                Block_list.append(AttnBlock(dim=dim, num_heads=num_heads,
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                            LayerNorm_type=LayerNorm_type, attention_type=attn_type2,
                                            FFW_type=FFW_type, num_frames_tocache=num_frames_tocache, group=group))

        Block_list.append(AttnBlock(dim=dim, num_heads=num_heads,
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                    LayerNorm_type=LayerNorm_type, attention_type=attn_type3, FFW_type=FFW_type,
                                    num_frames_tocache=num_frames_tocache, plot_attn=True, group=group))

        self.transformer_blocks = nn.ModuleList(Block_list)

    def forward(self, x, prior=None, k1_cached=None, v1_cached=None, k2_cached=None, v2_cached=None):

        out, k1_tocahe, v1_tocahe = self.transformer_blocks[0](x, prior, k1_cached, v1_cached)

        if self.num_blocks > 2:
            for i in range(1, self.num_blocks - 1):
                out, _, _ = self.transformer_blocks[i](out, prior)

        out, k2_tocahe, v2_tocahe = self.transformer_blocks[-1](out, prior, k2_cached, v2_cached)

        return out, k1_tocahe, v1_tocahe, k2_tocahe, v2_tocahe


class Encoder(nn.Module):
    def __init__(self, dim, embed_dim=64, enc_blocks=[2, 4, 6], middle_blocks=7, num_heads=[2, 4, 8, 16],
                 ffn_expansion_factor=2.5, bias=False, LayerNorm_type='WithBias', num_frames_tocache=3, group=4):
        super(Encoder, self).__init__()

        self.encoder_level1 = LevelBlock(dim=dim, embed_dim=embed_dim, bias=bias,
                                         ffn_expansion_factor=ffn_expansion_factor,
                                         LayerNorm_type=LayerNorm_type, num_blocks=enc_blocks[0],
                                         attn_type1="ReducedAttn", attn_type2="ReducedAttn",
                                         FFW_type="FFW", num_frames_tocache=num_frames_tocache,
                                         num_heads=num_heads[0], group=group)

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = LevelBlock(dim=int(dim * 2 ** 1), embed_dim=embed_dim, bias=bias,
                                         ffn_expansion_factor=ffn_expansion_factor,
                                         LayerNorm_type=LayerNorm_type, num_blocks=enc_blocks[1],
                                         attn_type1="ReducedAttn", attn_type2="ReducedAttn",
                                         FFW_type="FFW", num_frames_tocache=num_frames_tocache,
                                         num_heads=num_heads[1], group=group // 2)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  # From Level 2 to Level 3
        self.encoder_level3 = LevelBlock(dim=int(dim * 2 ** 2), embed_dim=embed_dim, bias=bias,
                                         ffn_expansion_factor=ffn_expansion_factor,
                                         LayerNorm_type=LayerNorm_type, num_blocks=enc_blocks[2],
                                         attn_type1="Channel", attn_type2="Channel",
                                         FFW_type="GFFW", num_frames_tocache=num_frames_tocache,
                                         num_heads=num_heads[2], group=1)

        # Middle block
        self.down3_4 = Downsample(int(dim * 2 ** 2))  # From Level 3 to Level 4
        self.latent = LatentCacheBlock(dim=int(dim * 2 ** 3), ffn_expansion_factor=ffn_expansion_factor,
                                       bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=middle_blocks,
                                       attn_type1="FHR", attn_type2="Channel",
                                       attn_type3="FHR", FFW_type="GFFW",
                                       num_frames_tocache=num_frames_tocache,
                                       num_heads=num_heads[3], group=1)

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

    def forward(self, x, diff_prior, k_cached=None, v_cached=None):
        prior1 = diff_prior
        enc1, k1_tocahe, v1_tocahe = self.encoder_level1(x, prior1, k_cached[0], v_cached[0])

        prior2 = self.down_prior12(prior1)
        x = self.down1_2(enc1)
        enc2, k2_tocahe, v2_tocahe = self.encoder_level2(x, prior2, k_cached[1], v_cached[1])

        prior3 = self.down_prior23(prior2)
        x = self.down2_3(enc2)
        enc3, k3_tocahe, v3_tocahe = self.encoder_level3(x, prior3, k_cached[2], v_cached[2])

        x = self.down3_4(enc3)
        latent, k4_tocahe, v4_tocahe, k5_tocahe, v5_tocahe = self.latent(x, prior=None,
                                                                         k1_cached=k_cached[3], v1_cached=v_cached[3],
                                                                         k2_cached=k_cached[4], v2_cached=v_cached[4])

        return ([enc1, enc2, enc3], latent, [prior1, prior2, prior3],
                [k1_tocahe, k2_tocahe, k3_tocahe, k4_tocahe, k5_tocahe],
                [v1_tocahe, v2_tocahe, v3_tocahe, v4_tocahe, v5_tocahe])


class Decoder(nn.Module):
    def __init__(self, dim, embed_dim=64, dec_blocks=[2, 4, 6], num_heads=[2, 4, 8], ffn_expansion_factor=2.5,
                 bias=False, LayerNorm_type='WithBias', num_frames_tocache=3, group=4):
        super(Decoder, self).__init__()

        self.up2_1 = Upsample(int(dim * 2 ** 1))  # From Level 2 to Level 1
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1), kernel_size=1, bias=bias)
        self.decoder_level1 = LevelBlock(dim=int(dim * 1), embed_dim=embed_dim,
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, num_blocks=dec_blocks[0],
                                         attn_type1="Channel", attn_type2="CHM",
                                         FFW_type="GFFW", num_frames_tocache=num_frames_tocache,
                                         num_heads=num_heads[0], Scale_patchsize=8, group=group)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = LevelBlock(dim=int(dim * 2 ** 1), embed_dim=embed_dim,
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, num_blocks=dec_blocks[1],
                                         attn_type1="Channel", attn_type2="CHM",
                                         FFW_type="GFFW", num_frames_tocache=num_frames_tocache,
                                         num_heads=num_heads[1], Scale_patchsize=4, group=group // 2)

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = LevelBlock(dim=int(dim * 2 ** 2), embed_dim=embed_dim,
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, num_blocks=dec_blocks[2],
                                         attn_type1="Channel", attn_type2="CHM",
                                         FFW_type="GFFW", num_frames_tocache=num_frames_tocache,
                                         num_heads=num_heads[2], Scale_patchsize=2, group=1)

    def forward(self, encs, latent, diff_prior, k_cached=None, v_cached=None):
        enc1, enc2, enc3 = encs
        prior1, prior2, prior3 = diff_prior

        dec3 = self.up4_3(latent)
        dec3 = torch.cat([dec3, enc3], 1)
        dec3 = self.reduce_chan_level3(dec3)
        dec3, k6_tocahe, v6_tocahe = self.decoder_level3(dec3, prior3, k_cached[5], v_cached[5])

        dec2 = self.up3_2(dec3)
        dec2 = torch.cat([dec2, enc2], 1)
        dec2 = self.reduce_chan_level2(dec2)
        dec2, k7_tocahe, v7_tocahe = self.decoder_level2(dec2, prior2, k_cached[6], v_cached[6])

        dec1 = self.up2_1(dec2)
        dec1 = torch.cat([dec1, enc1], 1)
        dec1 = self.reduce_chan_level1(dec1)
        dec1, k8_tocahe, v8_tocahe = self.decoder_level1(dec1, prior1, k_cached[7], v_cached[7])

        return [dec1, dec2, dec3], [k6_tocahe, k7_tocahe, k8_tocahe], [v6_tocahe, v7_tocahe, v8_tocahe]


class Trans_UNet(nn.Module):
    def __init__(self,
                 inp_channels,
                 out_channels,
                 dim,
                 Enc_blocks,
                 Middle_blocks,
                 Dec_blocks,
                 num_heads,
                 num_refinement_blocks,
                 ffn_expansion_factor,
                 bias,
                 LayerNorm_type,
                 num_heads_blks,

                 # Encoder attention types
                 encoder1_attn_type1, encoder1_attn_type2,
                 encoder2_attn_type1, encoder2_attn_type2,
                 encoder3_attn_type1, encoder3_attn_type2,

                 # Decoder attention types
                 decoder1_attn_type1, decoder1_attn_type2,
                 decoder2_attn_type1, decoder2_attn_type2,
                 decoder3_attn_type1, decoder3_attn_type2,

                 # FFW types for each encoder and decoder level
                 encoder1_ffw_type, encoder2_ffw_type, encoder3_ffw_type,
                 decoder1_ffw_type, decoder2_ffw_type, decoder3_ffw_type,

                 # Latent
                 latent_attn_type1, latent_attn_type2, latent_attn_type3, latent_ffw_type,

                 # Refinement
                 refinement_attn_type1, refinement_attn_type2, refinement_ffw_type,

                 use_both_input,
                 num_frames_tocache):
        super(Trans_UNet, self).__init__()
        if use_both_input:
            inp_channels *= 2
        self.use_both_input = use_both_input
        self.num_heads = num_heads
        self.input_projection = nn.Conv2d(inp_channels,
                                          dim, kernel_size=3,
                                          stride=1, padding=1,
                                          bias=bias)

        # Encoder Levels
        self.encoder_level1 = LevelBlock(dim=dim, bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                         LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[0],
                                         attn_type1=encoder1_attn_type1, attn_type2=encoder1_attn_type2,
                                         FFW_type=encoder1_ffw_type, num_frames_tocache=num_frames_tocache,
                                         num_heads=self.num_heads[0])

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = LevelBlock(dim=int(dim * 2 ** 1), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                         LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[1],
                                         attn_type1=encoder2_attn_type1, attn_type2=encoder2_attn_type2,
                                         FFW_type=encoder2_ffw_type, num_frames_tocache=num_frames_tocache,
                                         num_heads=self.num_heads[1])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  # From Level 2 to Level 3
        self.encoder_level3 = LevelBlock(dim=int(dim * 2 ** 2), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                         LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[2],
                                         attn_type1=encoder3_attn_type1, attn_type2=encoder3_attn_type2,
                                         FFW_type=encoder3_ffw_type, num_frames_tocache=num_frames_tocache,
                                         num_heads=self.num_heads[2])

        # Middle block
        self.down3_4 = Downsample(int(dim * 2 ** 2))  # From Level 3 to Level 4
        self.latent = LatentCacheBlock(dim=int(dim * 2 ** 3), ffn_expansion_factor=ffn_expansion_factor,
                                       bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Middle_blocks,
                                       attn_type1=latent_attn_type1, attn_type2=latent_attn_type2,
                                       attn_type3=latent_attn_type3, FFW_type=latent_ffw_type,
                                       num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[3])

        # Decoder Levels
        self.up4_3 = Upsample(int(dim * 2 ** 3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = LevelBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[0],
                                         attn_type1=decoder1_attn_type1, attn_type2=decoder1_attn_type2,
                                         FFW_type=decoder1_ffw_type, num_frames_tocache=num_frames_tocache,
                                         num_heads=self.num_heads[2], Scale_patchsize=2)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = LevelBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[1],
                                         attn_type1=decoder2_attn_type1, attn_type2=decoder2_attn_type2,
                                         FFW_type=decoder2_ffw_type, num_frames_tocache=num_frames_tocache,
                                         num_heads=self.num_heads[1], Scale_patchsize=4)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  # From Level 2 to Level 1
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1), kernel_size=1, bias=bias)
        self.decoder_level1 = LevelBlock(dim=int(dim * 1), ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[2],
                                         attn_type1=decoder3_attn_type1, attn_type2=decoder3_attn_type2,
                                         FFW_type=decoder3_ffw_type, num_frames_tocache=2, num_heads=self.num_heads[0],
                                         Scale_patchsize=8)

        # Refinement Block
        self.refinement = LevelBlock(dim=int(dim * 1), ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_refinement_blocks,
                                     attn_type1=refinement_attn_type1, attn_type2=refinement_attn_type2,
                                     FFW_type=refinement_ffw_type, num_frames_tocache=num_frames_tocache,
                                     num_heads=self.num_heads[0])

        self.ending = nn.Conv2d(in_channels=int(dim * 1),
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1,
                                stride=1,
                                groups=1,
                                bias=True)

        self.padder_size = (2 ** 3) * 4

    def forward(self, x, k_cached=None, v_cached=None):
        B, _, C, H, W = x.shape
        x = self.check_image_size(x)
        if k_cached == None:
            k_cached = [None] * 8
            v_cached = [None] * 8

        k_to_cache = []
        v_to_cache = []

        if self.use_both_input:
            previous, current = x[:, 0, :, :, :], x[:, 1, :, :, :]
            inp_img = torch.cat([previous,
                                 current], dim=1)
        else:
            inp_img = x[:, 1, :, :, :]
            current = inp_img

        inp_enc_level1 = self.input_projection(inp_img.float())
        out_enc_level1, k1_tocahe, v1_tocahe = self.encoder_level1(inp_enc_level1,
                                                                   k_cached[0],
                                                                   v_cached[0])

        k_to_cache.append(k1_tocahe)
        v_to_cache.append(v1_tocahe)

        inp_enc_level2 = self.down1_2(out_enc_level1)  # h/w下采样2倍, 通道扩大2倍
        out_enc_level2, k2_tocahe, v2_tocahe = self.encoder_level2(inp_enc_level2,
                                                                   k_cached[1],
                                                                   v_cached[1])

        k_to_cache.append(k2_tocahe)
        v_to_cache.append(v2_tocahe)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, k3_tocahe, v3_tocahe = self.encoder_level3(inp_enc_level3,
                                                                   k_cached[2],
                                                                   v_cached[2])

        k_to_cache.append(k3_tocahe)
        v_to_cache.append(v3_tocahe)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, k4_tocahe, v4_tocahe, k5_tocahe, v5_tocahe = self.latent(inp_enc_level4,
                                                                         k_cached[3], v_cached[3],
                                                                         k_cached[4], v_cached[4])

        # k4/v4、k5/v5缓存
        k_to_cache.append(k4_tocahe)
        k_to_cache.append(k5_tocahe)

        v_to_cache.append(v4_tocahe)
        v_to_cache.append(v5_tocahe)

        inp_dec_level3 = self.up4_3(latent)  # h/w上采样2倍, 通道缩小2倍, 与同级的encoder输出相对应
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, k6_tocahe, v6_tocahe = self.decoder_level3(inp_dec_level3,
                                                                   k_cached[5],
                                                                   v_cached[5])

        # k6/v6、k7/v7、k8/v8缓存
        k_to_cache.append(k6_tocahe)
        v_to_cache.append(v6_tocahe)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, k7_tocahe, v7_tocahe = self.decoder_level2(inp_dec_level2,
                                                                   k_cached[6],
                                                                   v_cached[6])

        k_to_cache.append(k7_tocahe)
        v_to_cache.append(v7_tocahe)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1, k8_tocahe, v8_tocahe = self.decoder_level1(inp_dec_level1,
                                                                   k_cached[7],
                                                                   v_cached[7])

        k_to_cache.append(k8_tocahe)
        v_to_cache.append(v8_tocahe)

        out_dec_level1, _, _ = self.refinement(out_dec_level1)

        ending = self.ending(out_dec_level1) + current

        return (ending[:, :, :H, :W], k_to_cache, v_to_cache)

    def check_image_size(self, x):
        _, _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
