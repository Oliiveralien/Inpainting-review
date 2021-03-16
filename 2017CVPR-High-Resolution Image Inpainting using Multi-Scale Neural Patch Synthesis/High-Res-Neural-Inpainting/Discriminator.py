import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # input 3 x 64 x 64
            nn.Conv2d(opt.channal, opt.DFilter, 4, 2, 1, bias=False),
            nn.ELU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(opt.DFilter, opt.DFilter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.DFilter * 2),
            nn.ELU(0.2, inplace=True),
            # state size. 128 x 16 x 16
            nn.Conv2d(opt.DFilter * 2, opt.DFilter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.DFilter * 4),
            nn.ELU(0.2, inplace=True),
            # state size. 256 x 8 x 8
            nn.Conv2d(opt.DFilter * 4, opt.DFilter * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.DFilter * 8),
            nn.ELU(0.2, inplace=True),
            # state size. 512 x 4 x 4
            nn.Conv2d(opt.DFilter * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.net(input)
        return output.view(-1, 1)
