import torch
import torch.nn as nn
import torch.nn.functional as F


class RBNModule(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True):
        super(RBNModule, self).__init__()
        self.num_features = num_features
        self.track_running_stats = track_running_stats
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, mask_t):
        input_m = input * mask_t
        if self.training:
            mask_mean = torch.mean(mask_t, (0, 2, 3), True)
            x_mean = torch.mean(input_m, (0, 2, 3), True) / mask_mean
            x_var = torch.mean(((input_m - x_mean) * mask_t) ** 2, (0, 2, 3), True) / mask_mean

            x_out = self.weight * (input_m - x_mean) / torch.sqrt(x_var + self.eps) + self.bias

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * x_mean.data)
            self.running_var.mul_(self.momentum)
            self.running_var.add_((1 - self.momentum) * x_var.data)
        else:
            x_out = self.weight * (input_m - self.running_mean) / torch.sqrt(self.running_var + self.eps) + self.bias
        return x_out * mask_t + input * (1 - mask_t)


class RCNModule(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True):
        super(RCNModule, self).__init__()
        self.num_features = num_features
        self.track_running_stats = track_running_stats
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, mask_t):
        input_m = input * mask_t

        if self.training:
            mask_mean_bn = torch.mean(mask_t, (0, 2, 3), True)
            mean_bn = torch.mean(input_m, (0, 2, 3), True) / mask_mean_bn
            var_bn = torch.mean(((input_m - mean_bn) * mask_t) ** 2, (0, 2, 3), True) / mask_mean_bn

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean_bn.data)
            self.running_var.mul_(self.momentum)
            self.running_var.add_((1 - self.momentum) * var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        mask_mean_in = torch.mean(mask_t, (2, 3), True)
        mean_in = torch.mean(input_m, (2, 3), True) / mask_mean_in
        var_in = torch.mean(((input_m - mean_in) * mask_t) ** 2, (2, 3), True) / mask_mean_in

        mask_mean_ln = torch.mean(mask_t, (1, 2, 3), True)
        mean_ln = torch.mean(input_m, (1, 2, 3), True) / mask_mean_ln
        var_ln = torch.mean(((input_m - mean_ln) * mask_t) ** 2, (1, 2, 3), True) / mask_mean_ln

        mean_weight = F.softmax(self.mean_weight)
        var_weight = F.softmax(self.var_weight)

        x_mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
        x_var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

        x_out = self.weight * (input_m - x_mean) / torch.sqrt(x_var + self.eps) + self.bias
        return x_out * mask_t + input * (1 - mask_t)

