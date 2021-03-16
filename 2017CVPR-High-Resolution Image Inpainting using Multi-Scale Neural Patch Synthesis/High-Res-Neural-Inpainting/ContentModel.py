import torch.nn as nn


class ContentNet(nn.Module):
    def __init__(self, option):
        super(ContentNet, self).__init__()
        self.net = nn.Sequential(
            # input:3*128*128?
            # first layer:input->64*64*64
            nn.Conv2d(option.channal, option.filter1, 4, 2, 1, bias=False),
            nn.ELU(alpha=0.2, inplace=True),
            # second layer:64*64*64->64*32*32
            nn.Conv2d(option.filter1, option.filter1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(option.filter1),
            nn.ELU(alpha=0.2, inplace=True),
            # third layer
            nn.Conv2d(option.filter1, option.filter1 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(option.filter1 * 2),
            nn.ELU(alpha=0.2, inplace=True),
            # fourth layer
            nn.Conv2d(option.filter1 * 2, option.filter1 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(option.filter1 * 4),
            nn.ELU(alpha=0.2, inplace=True),
            # fifth layer
            nn.Conv2d(option.filter1 * 4, option.filter1 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(option.filter1 * 8),
            nn.ELU(alpha=0.2, inplace=True),
            # bottleneck of encoder
            nn.Conv2d(option.filter1 * 8, option.bottleneck, 4, bias=False),
            nn.BatchNorm2d(option.bottleneck),
            nn.ELU(alpha=0.2, inplace=True),
            # decoder first layer
            nn.ConvTranspose2d(option.bottleneck, option.deFilter * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(option.deFilter * 8),
            nn.ELU(alpha=0.2, inplace=True),
            # decoder second layer
            nn.ConvTranspose2d(option.deFilter * 8, option.deFilter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(option.deFilter * 4),
            nn.ELU(alpha=0.2, inplace=True),
            # thrid layer
            nn.ConvTranspose2d(option.deFilter * 4, option.deFilter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(option.deFilter * 2),
            nn.ELU(alpha=0.2, inplace=True),
            # fourth layer
            nn.ConvTranspose2d(option.deFilter * 2, option.deFilter, 4, 2, 1, bias=False),
            nn.BatchNorm2d(option.deFilter),
            nn.ELU(alpha=0.2, inplace=True),
            # output layer
            nn.ConvTranspose2d(option.deFilter, option.channal, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.net(input)
        return output
