import torch
import torch.nn as nn
import torch.nn.functional as F
from .aff import AFNO2D
from mmcv.cnn.bricks import DropPath


class Shift(nn.Module):  # 空间移位
    def __init__(self, dim, shift=4):
        super(Shift, self).__init__()
        self.number = dim // 8
        self.shift = shift

    def forward(self, hw):
        n2 = 2
        n1 = self.number - 2 * n2

        s = self.shift  # 基本空间移位长度
        _, _, H, W = hw.shape
        s_out = torch.zeros_like(hw)
        # 16组*n2
        s_out[:, 0 * n2:1 * n2, s * 2:, s * 2:] = hw[:, 0 * n2:1 * n2, :-s * 2, :-s * 2]
        s_out[:, 1 * n2:2 * n2, s * 2:, s:] = hw[:, 1 * n2:2 * n2, :-s * 2, :-s]
        s_out[:, 2 * n2:3 * n2, s * 2:, 0:] = hw[:, 2 * n2:3 * n2, :-s * 2, :]
        s_out[:, 3 * n2:4 * n2, s * 2:, 0:-s] = hw[:, 3 * n2:4 * n2, :-s * 2, s:]
        s_out[:, 4 * n2:5 * n2, s * 2:, 0:-s * 2] = hw[:, 4 * n2:5 * n2, :-s * 2, s * 2:]
        s_out[:, 5 * n2:6 * n2, 0:-s * 2, s * 2:] = hw[:, 5 * n2:6 * n2, s * 2:, :-s * 2]
        s_out[:, 6 * n2:7 * n2, 0:-s * 2, s:] = hw[:, 6 * n2:7 * n2, s * 2:, :-s]
        s_out[:, 7 * n2:8 * n2, 0:-s * 2, 0:] = hw[:, 7 * n2:8 * n2, s * 2:, :]
        s_out[:, 8 * n2:9 * n2, 0:-s * 2, 0:-s] = hw[:, 8 * n2:9 * n2, s * 2:, s:]
        s_out[:, 9 * n2:10 * n2, 0:-s * 2, 0:-s * 2] = hw[:, 9 * n2:10 * n2, s * 2:, s * 2:]
        s_out[:, 10 * n2:11 * n2, s:, s * 2:] = hw[:, 10 * n2:11 * n2, :-s, :-s * 2]
        s_out[:, 11 * n2:12 * n2, s:, 0:-s * 2] = hw[:, 11 * n2:12 * n2, :-s, s * 2:]
        s_out[:, 12 * n2:13 * n2, :, s * 2:] = hw[:, 12 * n2:13 * n2, :, :-s * 2]
        s_out[:, 13 * n2:14 * n2, :, 0:-s * 2] = hw[:, 13 * n2:14 * n2, :, s * 2:]
        s_out[:, 14 * n2:15 * n2, 0:-s, s * 2:] = hw[:, 14 * n2:15 * n2, s:, :-s * 2]
        s_out[:, 15 * n2:16 * n2, 0:-s, 0:-s * 2] = hw[:, 15 * n2:16 * n2, s:, s * 2:]

        # 8组*n1
        s_out[:, 16 * n2 + 0 * n1:16 * n2 + 1 * n1, s:, s:] = hw[:, 16 * n2 + 0 * n1:16 * n2 + 1 * n1, :-s, :-s]
        s_out[:, 16 * n2 + 1 * n1:16 * n2 + 2 * n1, s:, 0:] = hw[:, 16 * n2 + 1 * n1:16 * n2 + 2 * n1, :-s, :]
        s_out[:, 16 * n2 + 2 * n1:16 * n2 + 3 * n1, s:, 0:-s] = hw[:, 16 * n2 + 2 * n1:16 * n2 + 3 * n1, :-s, s:]
        s_out[:, 16 * n2 + 3 * n1:16 * n2 + 4 * n1, :, s:] = hw[:, 16 * n2 + 3 * n1:16 * n2 + 4 * n1, :, :-s]
        s_out[:, 16 * n2 + 4 * n1:16 * n2 + 5 * n1, :, 0:-s] = hw[:, 16 * n2 + 4 * n1:16 * n2 + 5 * n1, :, s:]
        s_out[:, 16 * n2 + 5 * n1:16 * n2 + 6 * n1, 0:-s, s:] = hw[:, 16 * n2 + 5 * n1:16 * n2 + 6 * n1, s:, :-s]
        s_out[:, 16 * n2 + 6 * n1:16 * n2 + 7 * n1, 0:-s, 0:] = hw[:, 16 * n2 + 6 * n1:16 * n2 + 7 * n1, s:, :]
        s_out[:, 16 * n2 + 7 * n1:16 * n2 + 8 * n1, 0:-s, 0:-s] = hw[:, 16 * n2 + 7 * n1:16 * n2 + 8 * n1, s:, s:]

        return s_out


def generate_kernels(h=11, l=80, n=10):
    kernels = torch.zeros(l, 1, h, h).to(torch.device('cuda'))
    n2 = 2
    n1 = n-2*n2
    kernels[0*n2:1*n2,:,0,0] = 1
    kernels[1*n2:2*n2,:,0,h//4] = 1
    kernels[2*n2:3*n2,:,0,h//2] = 1
    kernels[3*n2:4*n2,:,0,3*h//4] = 1
    kernels[4*n2:5*n2,:,0,h-1] = 1
    kernels[5*n2:6*n2,:,h-1,0] = 1
    kernels[6*n2:7*n2,:,h-1,h//4] = 1
    kernels[7*n2:8*n2,:,h-1,h//2] = 1
    kernels[8*n2:9*n2,:,h-1,3*h//4] = 1
    kernels[9*n2:10*n2,:,h-1,h-1] = 1
    kernels[10*n2:11*n2,:,h//4,0] = 1
    kernels[11*n2:12*n2,:,h//4,h-1] = 1
    kernels[12*n2:13*n2,:,h//2,0] = 1
    kernels[13*n2:14*n2,:,h//2,h-1] = 1
    kernels[14*n2:15*n2,:,3*h//4,0] = 1
    kernels[15*n2:16*n2,:,3*h//4,h-1] = 1
    kernels[16*n2+0*n1:16*n2+1*n1,:,h//4,h//4] = 1
    kernels[16*n2+1*n1:16*n2+2*n1,:,h//4,h//2] = 1
    kernels[16*n2+2*n1:16*n2+3*n1,:,h//4,3*h//4] = 1
    kernels[16*n2+3*n1:16*n2+4*n1,:,h//2,h//4] = 1
    kernels[16*n2+4*n1:16*n2+5*n1,:,h//2,3*h//4] = 1
    kernels[16*n2+5*n1:16*n2+6*n1,:,3*h//4,h//4] = 1
    kernels[16*n2+6*n1:16*n2+7*n1,:,3*h//4,h//2] = 1
    kernels[16*n2+7*n1:16*n2+8*n1,:,3*h//4,3*h//4] = 1

    return kernels


class ResidualBlockShift(nn.Module):

    def __init__(self, dim):
        super(ResidualBlockShift, self).__init__()

        med_channel = int(dim * 4)
        self.number = med_channel // 8
        self.conv1 = nn.Conv2d(dim, med_channel, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(med_channel, dim, kernel_size=1)

    def forward(self, x):

        out = self.conv2(self.act(self.spatial_shift(self.conv1(x))))
        out = torch.cat([x, out], dim=1)
        # out += x

        return out

    def spatial_shift(self, hw):  # 切片索引
        n2 = (self.number - 1) // 2  # number = n_feat // 8
        n1 = self.number - 2 * n2

        s = 4  # 基本空间移位长度
        _, _, H, W = hw.shape
        s_out = torch.zeros_like(hw)
        # 16组*n2
        s_out[:, 0 * n2:1 * n2, s * 2:, s * 2:] = hw[:, 0 * n2:1 * n2, :-s * 2, :-s * 2]
        s_out[:, 1 * n2:2 * n2, s * 2:, s:] = hw[:, 1 * n2:2 * n2, :-s * 2, :-s]
        s_out[:, 2 * n2:3 * n2, s * 2:, 0:] = hw[:, 2 * n2:3 * n2, :-s * 2, :]
        s_out[:, 3 * n2:4 * n2, s * 2:, 0:-s] = hw[:, 3 * n2:4 * n2, :-s * 2, s:]
        s_out[:, 4 * n2:5 * n2, s * 2:, 0:-s * 2] = hw[:, 4 * n2:5 * n2, :-s * 2, s * 2:]
        s_out[:, 5 * n2:6 * n2, 0:-s * 2, s * 2:] = hw[:, 5 * n2:6 * n2, s * 2:, :-s * 2]
        s_out[:, 6 * n2:7 * n2, 0:-s * 2, s:] = hw[:, 6 * n2:7 * n2, s * 2:, :-s]
        s_out[:, 7 * n2:8 * n2, 0:-s * 2, 0:] = hw[:, 7 * n2:8 * n2, s * 2:, :]
        s_out[:, 8 * n2:9 * n2, 0:-s * 2, 0:-s] = hw[:, 8 * n2:9 * n2, s * 2:, s:]
        s_out[:, 9 * n2:10 * n2, 0:-s * 2, 0:-s * 2] = hw[:, 9 * n2:10 * n2, s * 2:, s * 2:]
        s_out[:, 10 * n2:11 * n2, s:, s * 2:] = hw[:, 10 * n2:11 * n2, :-s, :-s * 2]
        s_out[:, 11 * n2:12 * n2, s:, 0:-s * 2] = hw[:, 11 * n2:12 * n2, :-s, s * 2:]
        s_out[:, 12 * n2:13 * n2, :, s * 2:] = hw[:, 12 * n2:13 * n2, :, :-s * 2]
        s_out[:, 13 * n2:14 * n2, :, 0:-s * 2] = hw[:, 13 * n2:14 * n2, :, s * 2:]
        s_out[:, 14 * n2:15 * n2, 0:-s, s * 2:] = hw[:, 14 * n2:15 * n2, s:, :-s * 2]
        s_out[:, 15 * n2:16 * n2, 0:-s, 0:-s * 2] = hw[:, 15 * n2:16 * n2, s:, s * 2:]

        # 8组*n1
        s_out[:, 16 * n2 + 0 * n1:16 * n2 + 1 * n1, s:, s:] = hw[:, 16 * n2 + 0 * n1:16 * n2 + 1 * n1, :-s, :-s]
        s_out[:, 16 * n2 + 1 * n1:16 * n2 + 2 * n1, s:, 0:] = hw[:, 16 * n2 + 1 * n1:16 * n2 + 2 * n1, :-s, :]
        s_out[:, 16 * n2 + 2 * n1:16 * n2 + 3 * n1, s:, 0:-s] = hw[:, 16 * n2 + 2 * n1:16 * n2 + 3 * n1, :-s, s:]
        s_out[:, 16 * n2 + 3 * n1:16 * n2 + 4 * n1, :, s:] = hw[:, 16 * n2 + 3 * n1:16 * n2 + 4 * n1, :, :-s]
        s_out[:, 16 * n2 + 4 * n1:16 * n2 + 5 * n1, :, 0:-s] = hw[:, 16 * n2 + 4 * n1:16 * n2 + 5 * n1, :, s:]
        s_out[:, 16 * n2 + 5 * n1:16 * n2 + 6 * n1, 0:-s, s:] = hw[:, 16 * n2 + 5 * n1:16 * n2 + 6 * n1, s:, :-s]
        s_out[:, 16 * n2 + 6 * n1:16 * n2 + 7 * n1, 0:-s, 0:] = hw[:, 16 * n2 + 6 * n1:16 * n2 + 7 * n1, s:, :]
        s_out[:, 16 * n2 + 7 * n1:16 * n2 + 8 * n1, 0:-s, 0:-s] = hw[:, 16 * n2 + 7 * n1:16 * n2 + 8 * n1, s:, s:]

        return s_out


class GDFN(nn.Module):
    def __init__(self, dim, mlp_ratio=2, bias=False):
        super(GDFN, self).__init__()

        hidden_features = int(dim * mlp_ratio)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TokenMixer(nn.Module):
    def __init__(self, dim=64):
        super(TokenMixer, self).__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.aff = AFNO2D(hidden_size=dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.aff(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class Xvo_Attention(nn.Module):
    def __init__(self, dim, shift=3):
        super(Xvo_Attention, self).__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.shift = Shift(dim, shift=shift)
        self.aff = AFNO2D(hidden_size=dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x_shift = self.shift(x) * x
        x_aff = self.aff(x)
        x = x_shift + x_aff
        # x = torch.cat([x_shift, x_aff], dim=1)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class Xvo_block(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0., bias=False):
        super(Xvo_block, self).__init__()
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        # self.attn = TokenMixer(dim)
        self.attn = Xvo_Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = GDFN(dim, mlp_ratio=mlp_ratio, bias=bias)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.ffn(self.norm2(x)))
        return x


if __name__ == '__main__':
    x1 = torch.randn(3, 16, 256, 256).cuda()
    x2 = torch.randn(3, 64, 256, 256).cuda()
    res_shift = ResidualBlockShift(16).cuda()
    Xvo_block = Xvo_block(64).cuda()
    x1 = res_shift(x1)
    print(x1.shape)
    x2 = Xvo_block(x2)
    print(x2.shape)
