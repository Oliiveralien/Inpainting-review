import torch
import torch.nn as nn
import torch.nn.functional as F

class TVLoss(nn.Module):
    def __init__(self,weight=1):
        super(TVLoss,self).__init__()
        self.weight=weight
        self.loss=None

    def forward(self, input):
        image = input.squeeze().permute([1, 2, 0])
        r = (image[:, :, 0] + 2.12) / 4.37
        g = (image[:, :, 1] + 2.04) / 4.46
        b = (image[:, :, 2] + 1.80) / 4.44

        temp = torch.cat([r.unsqueeze(2), g.unsqueeze(2), b.unsqueeze(2)], dim=2)
        gx = torch.cat((temp[1:, :, :], temp[-1, :, :].unsqueeze(0)), dim=0)
        gx = gx - temp

        gy = torch.cat((temp[:, 1:, :], temp[:, -1, :].unsqueeze(1)), dim=1)
        gy = gy - temp

        self.loss = torch.mean(torch.pow(gx, 2)) + torch.mean(torch.pow(gy, 2))
        return input