import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Attention import PixelContextualAttention
from torchvision import models
from modules.ValidMigration import ConvOffset2D
from modules.RegionNorm import RBNModule, RCNModule

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class DSModule(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, rn=True, sample='none-3', activ='relu',
                 conv_bias=False, defor=True):
        super().__init__()
        if sample == 'down-5':
            self.conv = nn.Conv2d(in_ch+1, out_ch, 5, 2, 2, bias=conv_bias)
            self.updatemask = nn.MaxPool2d(5,2,2)
            if defor:
                self.offset = ConvOffset2D(in_ch+1)
        elif sample == 'down-7':
            self.conv = nn.Conv2d(in_ch+1, out_ch, 7, 2, 3, bias=conv_bias)
            self.updatemask = nn.MaxPool2d(7, 2, 3)
            if defor:
                self.offset = ConvOffset2D(in_ch+1)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(in_ch+1, out_ch, 3, 2, 1, bias=conv_bias)
            self.updatemask = nn.MaxPool2d(3, 2, 1)
            if defor:
                self.offset = ConvOffset2D(in_ch+1)
        else:
            self.conv = nn.Conv2d(in_ch+2, out_ch, 3, 1, 1, bias=conv_bias)
            self.updatemask = nn.MaxPool2d(3,1,1)
            if defor:
                self.offset0 = ConvOffset2D(in_ch-out_ch+1)
                self.offset1 = ConvOffset2D(out_ch+1)
        self.in_ch = in_ch
        self.out_ch = out_ch

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if rn:
            # Regional Composite Normalization
            self.rn = RCNModule(out_ch)

            # Regional Batch Normalization
            # self.rn = RBNModule(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace = True)

    def forward(self, input, input_mask):
        if hasattr(self, 'offset'):
            input = torch.cat([input, input_mask[:,:1,:,:]], dim = 1)
            h = self.offset(input)
            h = input*input_mask[:,:1,:,:] + (1-input_mask[:,:1,:,:])*h
            h = self.conv(h)
            h_mask = self.updatemask(input_mask[:,:1,:,:])
            h = h*h_mask
            h = self.rn(h, h_mask)
        elif hasattr(self, 'offset0'):
            h1_in = torch.cat([input[:,self.in_ch-self.out_ch:,:,:], input_mask[:,1:,:,:]], dim = 1)
            m1_in = input_mask[:,1:,:,:]
            h0 = torch.cat([input[:,:self.in_ch-self.out_ch,:,:], input_mask[:,:1,:,:]], dim = 1)
            h1 = self.offset1(h1_in)
            h1 = m1_in*h1_in + (1-m1_in)*h1
            h = self.conv(torch.cat([h0,h1], dim = 1))
            h = self.rn(h, input_mask[:,:1,:,:])
            h_mask = F.interpolate(input_mask[:,:1,:,:], scale_factor=2, mode='nearest')
        else:
            h = self.conv(torch.cat([input, input_mask[:,:,:,:]], dim = 1))
            h_mask = self.updatemask(input_mask[:,:1,:,:])
            h = h*h_mask

        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

class DSNet(nn.Module):
    def __init__(self, layer_size=8, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = DSModule(input_channels, 64, rn=False, sample='down-7', defor = False)
        self.enc_2 = DSModule(64, 128, sample='down-5')
        self.enc_3 = DSModule(128, 256, sample='down-5')
        self.enc_4 = DSModule(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, DSModule(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, DSModule(512 + 512, 512, activ='leaky'))
        self.dec_4 = DSModule(512 + 256, 256, activ='leaky')
        self.dec_3 = DSModule(256 + 128, 128, activ='leaky')
        self.dec_2 = DSModule(128 + 64, 64, activ='leaky')
        self.dec_1 = DSModule(64 + input_channels, input_channels,
                              rn=False, activ=None, defor = False)
        self.att = PixelContextualAttention(128)
    def forward(self, input, input_mask):
        input_mask = input_mask[:,0:1,:,:]
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]
        h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            if i == 3:
                h = self.att(h, input_mask[:,:1,:,:])
        return h, h_mask

    def train(self, mode=True, finetune = False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, RCNModule):
                    module.eval()
