import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """
    content loss layer
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = None

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

    def update(self, target):
        """
        update target of content loss
        :param target:
        :return:
        """
        self.target = target.detach()

