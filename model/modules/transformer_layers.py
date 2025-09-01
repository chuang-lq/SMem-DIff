
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # b, h, w, c
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class SpatialGate(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm2d(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * self.dwconv(self.norm(x2))


class SGFN(nn.Module):
    # Spatial-Gate Feed-Forward Network
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(SGFN, self).__init__()

        hidden_dim = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.GELU()
        self.sg = SpatialGate(hidden_dim // 2)
        self.project_out = nn.Conv2d(hidden_dim // 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.act(x)
        x = self.sg(x)
        x = self.project_out(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FFN, self).__init__()

        hidden_dim = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.project_out(x)
        return x


class PixelFFN(nn.Module):
    """
    Args:
        dim (int): Base channels.
        hidden_dim (int): Channels of hidden mlp.
    """

    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super().__init__()

        hidden_dim = int(dim * ffn_expansion_factor)

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1, bias=bias),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, dim, 1, bias=bias),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=bias),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, bias=bias),
            nn.Sigmoid()
        )

        self.fc1 = nn.Conv2d(dim * 2, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        x = torch.cat([self.ca(x) * x, self.pa(x) * x], dim=1)
        return self.fc2(self.act(self.fc1(x)))


# class Attention(nn.Module):
#     # 计算空间上的注意力
#
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.kv_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
#
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x, f=None):
#         h, w = x.shape[-2:]
#
#         q = self.q_dwconv(self.q(x))
#         if f is None:
#             kv = self.kv_dwconv(self.kv_conv(x))
#         else:
#             kv = self.kv_dwconv(self.kv_conv(f))
#         k, v = kv.chunk(2, dim=1)
#
#         q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#         out = rearrange(out, 'b head (h w) c -> b (head c) h w', h=h)
#         out = self.project_out(out)
#
#         return out


class Attention(nn.Module):
    # Restormer (CVPR 2022) transposed-attention block
    # 计算通道上的自注意力而不是空间上，通过计算通道上的注意力来隐式编码全局上下文信息
    # original source code: https://github.com/swz30/Restormer
    def __init__(self, dim, num_heads, bias=False):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_conv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        # self.act = nn.ReLU()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def grids(self, x, kernel_size):
        b, c, h, w = x.shape

        k1 = k2 = kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.stack(parts, dim=1)  # b, num_patches, c, k1, k2

        return parts, idxes

    def grids_inverse(self, x, outs, idxes, kernel_size):
        b, c, h, w = x.shape
        preds = torch.zeros(x.shape).to(x.device)
        count_mt = torch.zeros((b, 1, h, w)).to(x.device)

        k1 = k2 = kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[:, :, i:i + k1, j:j + k2] += outs[:, cnt]
            count_mt[:, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def forward(self, x, f=None):
        h, w = x.shape[-2:]

        q = self.q_dwconv(self.q(x))
        if f is None:
            kv = self.kv_dwconv(self.kv_conv(x))
        else:
            kv = self.kv_dwconv(self.kv_conv(f))
        k, v = kv.chunk(2, dim=1)

        q, k, v = map(lambda x: rearrange(x, 'b (head c) h w -> b head c (h w)', head=self.num_heads), (q, k, v))

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = self.act(attn)  # Sparse Attention due to ReLU's property # attn = self.act(attn)**2
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


class P2SP(nn.Module):
    def __init__(self, dim, num_heads, patch_kernel=32, bias=False):
        super().__init__()

        self.patch_kernel = patch_kernel
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

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

    def forward(self, x, sp):
        b, c, h, w = sp.shape
        p1, p2 = self.patch_kernel, self.patch_kernel // 4

        x = x + self.pos_embed(x)
        sp = sp + self.pos_embed(sp)

        q = self.q_dwconv(self.q_conv(sp))
        kv = self.kv_dwconv(self.kv_conv(x))

        q = self.local_partition(q, p2 - p2 // 4, p2 - p2 // 4, p2, p2)  # b n c dh dw
        kv = self.local_partition(kv, p1 - p1 // 4, p1 - p1 // 4, p1, p1)
        k, v = kv.chunk(2, dim=2)
        q, k, v = map(lambda x: rearrange(x, 'b n (head c) dh dw -> (b n) head (dh dw) c',
                                          head=self.num_heads), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, '(b n) head (dh dw) c -> b n (head c) dh dw', b=b, dh=p2)
        out = self.local_reverse(sp, out, p2 - p2 // 4, p2 - p2 // 4, p2, p2)  # b, c, h, w

        out = self.project_out(out)

        return out


class SP2P(nn.Module):
    def __init__(self, dim, num_heads, patch_kernel=32, bias=False):
        super().__init__()

        self.patch_kernel = patch_kernel
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

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

    def forward(self, x, sp):
        b, c, h, w = x.shape
        p1, p2 = self.patch_kernel, self.patch_kernel // 4

        x = x + self.pos_embed(x)
        sp = sp + self.pos_embed(sp)

        q = self.q_dwconv(self.q_conv(x))
        kv = self.kv_dwconv(self.kv_conv(sp))

        q = self.local_partition(q, p1 - p1 // 4, p1 - p1 // 4, p1, p1)  # b n c dh dw
        kv = self.local_partition(kv, p2 - p2 // 4, p2 - p2 // 4, p2, p2)
        k, v = kv.chunk(2, dim=2)
        # q, k, v = map(lambda x: rearrange(x, 'b (head c) h w -> b head (h w) c', head=self.num_heads), (q, k, v))
        q, k, v = map(lambda x: rearrange(x, 'b n (head c) dh dw -> (b n) head (dh dw) c',
                                          head=self.num_heads), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, '(b n) head (dh dw) c -> b n (head c) dh dw', b=b, dh=p1)
        out = self.local_reverse(x, out, p1 - p1 // 4, p1 - p1 // 4, p1, p1)  # b, c, h, w
        # out = rearrange(out, 'b head (h w) c -> b (head c) h w', h=h)
        out = self.project_out(out)

        return out


class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self, dim):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(dim, dim//2, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(dim, dim//2, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(dim, dim//2, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(dim, dim//2, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(dim//2, dim, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(dim//2, dim, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(dim//2, dim, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(dim//2, dim, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        # ca
        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        # sa
        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2

        return f1.view(b, c, h, w), f2.view(b, c, h, w)


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.network(x)


class AdditiveTokenMixer(nn.Module):
    """ Conv Additive Self-Attention
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


# Top-K稀疏注意力(TKSA)模块
class TKSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TKSA, self).__init__()
        self.num_heads = num_heads

        # 温度参数，用于缩放注意力得分
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 定义q, k, v的投影层，1x1卷积和深度卷积3x3
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        # 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 注意力的dropout
        self.attn_drop = nn.Dropout(0.)

        # 每个注意力掩码的可学习权重
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        # 计算q, k, v投影，并应用深度卷积
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 将q, k, v重排列为多头注意力的形式
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对q和k进行归一化，确保稳定的注意力计算
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape  # C表示每个头的通道数

        # 初始化不同top-k稀疏程度的掩码
        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        # 计算缩放后的点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 对注意力进行top-k稀疏处理，创建不同的注意力掩码
        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        # 对每个掩码中的注意力权重进行softmax归一化
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        # 对每个掩码计算注意力输出
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        # 使用学习到的注意力权重结合多个输出
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        # 将输出重排列回原始的维度
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 最后投影输出匹配输入维度
        out = self.project_out(out)
        return out


if __name__ == '__main__':
    x = torch.randn(2, 64, 320, 180).cuda()
    sp = torch.randn(2, 64, 80, 45).cuda()
    p2sp = P2SP(64, 8).cuda()
    x0 = p2sp.local_partition(x, 32 - 8, 32 - 8, 32, 32)
    sp0 = p2sp.local_partition(sp, 8 - 2, 8 - 2, 8, 8)
    print(x0.shape)
    print(sp0.shape)
    out1 = p2sp(x, sp)
    print(out1.shape)

    sp2p = SP2P(64, 8).cuda()
    out2 = sp2p(x, sp)
    print(out2.shape)
