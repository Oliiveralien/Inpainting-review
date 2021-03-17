import torch
from torch import nn
from torch.nn import functional as F

from modules.normalization import PCN


class ResBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False):
        super(ResBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.BatchNorm2d(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.conv2(out)
        # out = self.n2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.elu2(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.n1 = nn.BatchNorm2d(channels_out)
        self.elu1 = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.n1(out)
        out = self.elu1(out)
        return out


class PCBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False):
        super(PCBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.elu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.pcn = PCN(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.elu2 = nn.ELU(inplace=True)

    def forward(self, x, m):
        residual = x
        out = self.conv1(x)
        out = self.elu1(out)
        out = self.conv2(out)
        _, _, h, w = out.size()
        out = self.pcn(out, F.interpolate(m, (h, w), mode="nearest"))
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.elu2(out)
        return out


if __name__ == '__main__':
    pcb = PCBlock(channels_in=3, channels_out=32, kernel_size=5, stride=1, padding=2)
    inp = torch.rand((4, 3, 256, 256))
    mask = torch.rand((4, 1, 256, 256))
    out = pcb(inp, mask)
    print(out.size())
