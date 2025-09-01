import torch
from torch import nn
from .detailconv import DEConv
from .cga import SpatialAttention, ChannelAttention, PixelAttention


class DEABlock(nn.Module):
    def __init__(self, dim, kernel_size, reduction, bias, act):
        super(DEABlock, self).__init__()
        self.conv1 = DEConv(dim)
        self.act = act
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.sa = SpatialAttention(bias=bias)
        self.ca = ChannelAttention(dim, reduction, bias=bias)
        self.pa = PixelAttention(dim, bias=bias)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res


class DEBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(DEBlock, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res


# class DEABlock(nn.Module):
#     def __init__(self, dim, kernel_size, reduction=4, bias=False, act=nn.PReLU()):
#         super(DEABlock, self).__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, bias=bias)
#         self.act = act
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, bias=bias)
#         self.sa = SpatialAttention(bias=bias)
#         self.ca = ChannelAttention(dim, reduction, bias=bias)
#         self.pa = PixelAttention(dim, bias=bias)
#
#     def forward(self, x):
#         res = self.conv1(x)
#         res = self.act(res)
#         res = res + x
#         res = self.conv2(res)
#         cattn = self.ca(res)
#         sattn = self.sa(res)
#         pattn1 = sattn + cattn
#         pattn2 = self.pa(res, pattn1)
#         res = res * pattn2
#         res = res + x
#         return res


# class DEBlock(nn.Module):
#     def __init__(self, dim, kernel_size, bias=False):
#         super(DEBlock, self).__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, bias=bias)
#         self.act1 = nn.LeakyReLU(0.1, inplace=True)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, bias=bias)
#
#     def forward(self, x):
#         res = self.conv1(x)
#         res = self.act1(res)
#         res = res + x
#         res = self.conv2(res)
#         res = res + x
#         return res


if __name__ == '__main__':
    x = torch.randn(1, 64, 64, 64)
    deab = DEABlock(64, 3, reduction=4, bias=False, act=nn.PReLU())
    x = deab(x)
    print(x.shape)