from __future__ import print_function

import torch.nn as nn
import torchvision.models as models

from Mylib import StyleLoss, TV


class CNNMRF(nn.Module):
    def __init__(self, style_image, device, style_weight, tv_weight, gpu_chunck_size=256,
                 mrf_style_stride=2, mrf_synthesis_stride=2):
        super(CNNMRF, self).__init__()
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.patch_size = 3
        self.device = device
        self.gpu_chunck_size = gpu_chunck_size
        self.mrf_style_stride = mrf_style_stride
        self.mrf_synthesis_stride = mrf_synthesis_stride
        self.style_layers = [12, 21]
        self.model, self.style_losses, self.tv_loss = self.get_model_and_losses(
            style_image=style_image)

    def forward(self, input):
        # input is synthesis picture
        self.model(input)
        style_score = 0
        tv_score = self.tv_loss.loss

        # calculate style loss
        for sl in self.style_losses:
            style_score += sl.loss

        loss = self.style_weight * style_score + self.tv_weight + tv_score
        return loss

    def get_model_and_losses(self, style_image):
        vgg = models.vgg19(pretrained=True).to(self.device)
        model = nn.Sequential()

        style_losses = []
        # add tv loss layer
        tv_loss = TV.TVLoss()
        model.add_module('tv_loss', tv_loss)

        next_style_idx = 0

        for i in range(len(vgg.features)):
            if next_style_idx >= len(self.style_layers):
                break
            # add layer of vgg19
            layer = vgg.features[i]
            name = str(i)
            model.add_module(name, layer)

            # add style loss layer
            if i in self.style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss.StyleLoss(target_feature, patch_size=self.patch_size,
                                                 mrf_style_stride=self.mrf_style_stride,
                                                 mrf_synthesis_stride=self.mrf_synthesis_stride,
                                                 gpu_chunck_size=self.gpu_chunck_size, device=self.device)

                model.add_module("style_loss_{}".format(next_style_idx), style_loss)
                style_losses.append(style_loss)
                next_style_idx += 1

        return model, style_losses, tv_loss

    def update_style_and_content_image(self, style_image):
        x = style_image.clone()
        next_style_idx = 0
        i = 0
        for layer in self.model:
            if isinstance(layer, TV.TVLoss) or isinstance(layer, StyleLoss.StyleLoss):
                continue
            if next_style_idx >= len(self.style_losses):
                break
            x = layer(x)
            if i in self.style_layers:
                # extract feature of style image in vgg19 as style loss target
                self.style_losses[next_style_idx].update(x)
                next_style_idx += 1
            i += 1
