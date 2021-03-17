import torch
from torch import nn


class WeightedBCELoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        super(WeightedBCELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, out, target, weights=None):
        out = out.clamp(self.epsilon, 1 - self.epsilon)
        if weights is not None:
            assert len(weights) == 2
            loss = weights[1] * (target * torch.log(out)) + weights[0] * ((1 - target) * torch.log(1 - out))
        else:
            loss = target * torch.log(out) + (1 - target) * torch.log(1 - out)
        return torch.neg(torch.mean(loss))
