import torch
from torch import nn
from torch.nn import functional as F


class PCN(nn.Module):
    def __init__(self, n_features, epsilon=1e-5):
        super(PCN, self).__init__()
        self.epsilon = epsilon
        self.sqex = SqEx(n_features)

    def forward(self, x, m):
        _t = self._compute_T(x, m)
        _beta = self.sqex(x)
        context_feat = _beta * (_t * m) + (1. - _beta) * (x * m)
        preserved_feat = x * (1. - m)
        return context_feat + preserved_feat

    def _compute_T(self, x, m):
        X_p = x * m
        X_q = x * (1. - m)
        X_p_mean = self._compute_weighted_mean(X_p, m).unsqueeze(-1).unsqueeze(-1)
        X_p_std = self._compute_weighted_std(X_p, m).unsqueeze(-1).unsqueeze(-1)
        X_q_mean = self._compute_weighted_mean(X_q, m).unsqueeze(-1).unsqueeze(-1)
        X_q_std = self._compute_weighted_std(X_q, m).unsqueeze(-1).unsqueeze(-1)
        return ((X_p - X_p_mean) / X_p_std) * X_q_std + X_q_mean

    def _compute_weighted_mean(self, x, m):
        return torch.sum(x * m, dim=(2, 3)) / (torch.sum(m) + self.epsilon)

    def _compute_weighted_std(self, x, m):
        _mean = self._compute_weighted_mean(x, m).unsqueeze(-1).unsqueeze(-1)
        return torch.sqrt((torch.sum(torch.pow(x * m - _mean, 2), dim=(2, 3)) /
                          (torch.sum(m) + self.epsilon)) + self.epsilon)


class SqEx(nn.Module):
    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


if __name__ == '__main__':
    # sqex = SqEx(32)
    # inp = torch.rand((1, 32, 256, 256))
    # out = sqex(inp)
    # print(out.size())

    pcn = PCN(16)
    inp = torch.rand((2, 16, 256, 256))
    mask = torch.rand((2, 1, 256, 256))
    out = pcn(inp, mask)
    print(out.size())
