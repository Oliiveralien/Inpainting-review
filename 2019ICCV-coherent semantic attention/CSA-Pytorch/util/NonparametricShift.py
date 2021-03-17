import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class NonparametricShift(object):
    def buildAutoencoder(self, target_img, normalize, interpolate,  nonmask_point_idx, mask_point_idx, patch_size=1, stride=1):
        nDim = 3
        assert target_img.dim() == nDim, 'target image must be of dimension 3.'
        C = target_img.size(0)

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor

        patches_all, patches_part, patches_mask= self._extract_patches(target_img, patch_size, stride, nonmask_point_idx,mask_point_idx)

        #相当于个数 patch的个数
        npatches_part = patches_part.size(0)
        npatches_all = patches_all.size(0)
        npatches_mask=patches_mask.size(0)

        conv_enc_non_mask, conv_dec_non_mask = self._build(patch_size, stride, C, patches_part, npatches_part, normalize, interpolate)

        conv_enc_all, conv_dec_all = self._build(patch_size, stride, C, patches_all, npatches_all, normalize, interpolate)



        return conv_enc_all, conv_enc_non_mask, conv_dec_all, conv_dec_non_mask ,patches_part, patches_mask

    def _build(self, patch_size, stride, C, target_patches, npatches, normalize, interpolate):
        # for each patch, divide by its L2 norm.
        enc_patches = target_patches.clone()
        for i in range(npatches):
            enc_patches[i] = enc_patches[i]*(1/(enc_patches[i].norm(2)+1e-8))

        conv_enc = nn.Conv2d(C, npatches, kernel_size=patch_size, stride=stride, bias=False)
        conv_enc.weight.data = enc_patches

        # normalize is not needed, it doesn't change the result!
        if normalize:
            raise NotImplementedError

        if interpolate:
            raise NotImplementedError

        conv_dec = nn.ConvTranspose2d(npatches, C, kernel_size=patch_size, stride=stride, bias=False)
        conv_dec.weight.data = target_patches

        return conv_enc, conv_dec

    def _extract_patches(self, img, patch_size, stride, nonmask_point_idx, mask_point_idx):
        n_dim = 3
        assert img.dim() == n_dim, 'image must be of dimension 3.'
#kH=1, kW=1，dH, dW=1
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        #其中i_1, i_2, i_3, i_4, i_5    ------1：通道数  2：横轴上的patch个数   3：纵轴上patch个数    4和5分别是patch的长宽
        i_1, i_2, i_3, i_4, i_5 = input_windows.size(0), input_windows.size(1), input_windows.size(2), input_windows.size(3), input_windows.size(4)
        input_windows = input_windows.permute(1,2,0,3,4).contiguous().view(i_2*i_3, i_1, i_4, i_5)

        patches_all = input_windows
        patches = input_windows.index_select(0, nonmask_point_idx) #It returns a new tensor, representing patches extracted from non-masked region!
        maskpatches = input_windows.index_select(0,mask_point_idx)
        return patches_all, patches,maskpatches
    def _extract_patches_mask(self, img, patch_size, stride, nonmask_point_idx, mask_point_idx):
        n_dim = 3
        assert img.dim() == n_dim, 'image must be of dimension 3.'
#kH=1, kW=1，dH, dW=1
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        #其中i_1, i_2, i_3, i_4, i_5    ------1：通道数  2：横轴上的patch个数   3：纵轴上patch个数    4和5分别是patch的长宽
        i_1, i_2, i_3, i_4, i_5 = input_windows.size(0), input_windows.size(1), input_windows.size(2), input_windows.size(3), input_windows.size(4)
        input_windows = input_windows.permute(1,2,0,3,4).contiguous().view(i_2*i_3, i_1, i_4, i_5)
        maskpatches = input_windows.index_select(0,mask_point_idx)
        return maskpatches


