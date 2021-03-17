import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from modeling.base import BaseNetwork
from modules.blocks import ResBlock, ConvBlock, PCBlock


class MPN(BaseNetwork):
    def __init__(self, base_n_channels, neck_n_channels):
        super(MPN, self).__init__()
        assert base_n_channels >= 4, "Base num channels should be at least 4"
        assert neck_n_channels >= 16, "Neck num channels should be at least 16"
        self.rb1 = ResBlock(channels_in=3, channels_out=base_n_channels, kernel_size=5, stride=2, padding=2, dilation=1)
        self.rb2 = ResBlock(channels_in=base_n_channels, channels_out=base_n_channels * 2, kernel_size=3, stride=2)
        self.rb3 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.rb4 = ResBlock(channels_in=base_n_channels * 2, channels_out=neck_n_channels, kernel_size=3, stride=1, padding=4, dilation=4)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

        self.rb5 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1)
        self.rb6 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels, kernel_size=3, stride=1)
        self.rb7 = ResBlock(channels_in=base_n_channels, channels_out=base_n_channels // 2, kernel_size=3, stride=1)

        self.cb1 = ConvBlock(channels_in=base_n_channels // 2, channels_out=base_n_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(base_n_channels // 4, 1, kernel_size=3, stride=1, padding=1)

        self.init_weights(init_type="normal", gain=0.02)

    def forward(self, x):
        out = self.rb1(x)
        out = self.rb2(out)
        out = self.rb3(out)
        neck = self.rb4(out)
        # bottleneck here

        out = self.rb5(neck)
        out = self.upsample(out)
        out = self.rb6(out)
        out = self.upsample(out)
        out = self.rb7(out)

        out = self.cb1(out)
        out = self.conv1(out)

        return torch.sigmoid(out), neck


class RIN(BaseNetwork):
    def __init__(self, base_n_channels, neck_n_channels):
        super(RIN, self).__init__()
        assert base_n_channels >= 8, "Base num channels should be at least 8"
        assert neck_n_channels >= 32, "Neck num channels should be at least 32"
        self.pc1 = PCBlock(channels_in=3, channels_out=base_n_channels, kernel_size=5, stride=1, padding=2)
        self.pc2 = PCBlock(channels_in=base_n_channels, channels_out=base_n_channels * 2, kernel_size=3, stride=2, padding=1)
        self.pc3 = PCBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.pc4 = PCBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 4, kernel_size=3, stride=2, padding=1)
        self.pc5 = PCBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.pc6 = PCBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=2, dilation=2)
        self.pc7 = PCBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=2, dilation=2)
        self.pc8 = PCBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=4, dilation=4)
        self.pc9 = PCBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=4, dilation=4)
        self.pc10 = PCBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

        self.pc11 = PCBlock(channels_in=base_n_channels * 4 + neck_n_channels, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.pc12 = PCBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.pc13 = PCBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels, kernel_size=3, stride=1, padding=1)
        self.pc14 = PCBlock(channels_in=base_n_channels, channels_out=base_n_channels, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(base_n_channels, 3, kernel_size=3, stride=1, padding=1)
        self.init_weights(init_type="normal", gain=0.02)

    def forward(self, x, m, n):
        out = self.pc1(x, m)
        out = self.pc2(out, m)
        out = self.pc3(out, m)
        out = self.pc4(out, m)
        out = self.pc5(out, m)
        out = self.pc6(out, m)
        out = self.pc7(out, m)
        out = self.pc8(out, m)
        out = self.pc9(out, m)
        out = self.pc10(out, m)
        # bottleneck here

        out = torch.cat([out, n], dim=1)
        out = self.upsample(out)
        out = self.pc11(out, m)
        out = self.pc12(out, m)
        out = self.upsample(out)
        out = self.pc13(out, m)
        out = self.pc14(out, m)
        out = self.conv1(out)

        return torch.tanh(out)


# class Discriminator(BaseNetwork):
#     def __init__(self, base_n_channels):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(3, base_n_channels, kernel_size=5, stride=2, padding=2, bias=False)
#         self.conv2 = spectral_norm(nn.Conv2d(base_n_channels, base_n_channels * 2, kernel_size=5, stride=2, padding=2, bias=False))
#         self.conv3 = spectral_norm(nn.Conv2d(base_n_channels * 2, base_n_channels * 2, kernel_size=5, stride=2, padding=2, bias=False))
#         self.conv4 = spectral_norm(nn.Conv2d(base_n_channels * 2, base_n_channels * 4, kernel_size=5, stride=2, padding=2, bias=False))
#         self.conv5 = nn.Conv2d(base_n_channels * 4, base_n_channels * 8, kernel_size=5, stride=2, padding=2, bias=False)
#
#         self.init_weights()
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
#         out = F.leaky_relu(self.conv3(out), negative_slope=0.2)
#         out = F.leaky_relu(self.conv4(out), negative_slope=0.2)
#         out = self.conv5(out)
#
#         return out


class Discriminator(BaseNetwork):
    def __init__(self, base_n_channels):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.image_to_features = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 4 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(4 * base_n_channels, 8 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * base_n_channels * 8 * 8
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1)
        )

        self.init_weights(init_type="normal", gain=0.02)

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)


class PatchDiscriminator(Discriminator):
    def __init__(self, base_n_channels):
        super(PatchDiscriminator, self).__init__(base_n_channels)

        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        )

    def forward(self, input_data, mask=None):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        m = self.downsample(mask)
        x = x * m
        x = x.view(batch_size, -1)
        m = m.view(batch_size, -1)
        x = torch.sum(x, dim=-1, keepdim=True) / torch.sum(m, dim=-1, keepdim=True)
        return x


if __name__ == '__main__':
    mpn = MPN(64, 128)
    inp = torch.rand((2, 3, 256, 256))
    _, neck = mpn(inp)
    print(neck.size())
    rin = RIN(32, 128)
    inp = torch.rand((2, 3, 256, 256))
    mask = torch.rand((2, 1, 256, 256))
    out = rin(inp, mask, neck)
    print(out.size())
    disc = Discriminator(64)
    d_out = disc(out)
    print(d_out.size())
    patch_disc = PatchDiscriminator(64)
    p_d_out = patch_disc(out, mask)
    print(p_d_out.size())
