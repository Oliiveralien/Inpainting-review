import torch
import torch.nn as nn
import torch.nn.functional as F


from models.shift_net.InnerShiftTriple import InnerShiftTriple
from models.shift_net.InnerCos import InnerCos
from models.soft_shift_net.innerSoftShiftTriple import InnerSoftShiftTriple
from .unet import UnetSkipConnectionBlock
from .modules import *
from .ContextualAttention import *


################################### ***************************  #####################################
################################### Shift_net  #####################################
################################### ***************************  #####################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGeneratorShiftTriple(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list, shift_list, mask_global, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, dilation = True)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout, dilation = True)

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)

        unet_CA_block = UnetSkipConnectionCATriple(ngf * 2, ngf * 4, mask_global, CA_type = opt.CA_type, input_nc=None,
                                                                    submodule=unet_block,
                                                                    norm_layer=norm_layer)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf , ngf * 2, input_nc=None, submodule = unet_CA_block,
                                             norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class UnetSkipConnectionCATriple(nn.Module):
    def __init__(self, outer_nc, inner_nc, mask, input_nc, patch_size = 3, prop_size = 3, CA_type = "single",
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, dilation = False):
        super(UnetSkipConnectionCATriple, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc
            
        if dilation:
            dila_conv = [nn.LeakyReLU(0.2, True), nn.Conv2d(inner_nc, inner_nc, kernel_size = 3, stride = 1, padding = 2, dilation = 2)]
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                                 stride=1, padding=1)
            print("dilation")
        else:
            dila_conv = []
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1)
        self.mask = mask.float()
        if CA_type == "single":
            self.contextual_attention = ContextualAttentionModule(patch_size = patch_size, propagate_size = prop_size)
            print("single CA")
        elif CA_type == "parallel":
            self.contextual_attention = ParallelContextualAttention(outer_nc)
            print("parallel CA")
        self.combine_layer = nn.Conv2d(outer_nc * 2, outer_nc, kernel_size = 3, stride = 1, padding = 1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # Different position only has differences in `upconv`
            # for the outermost, the special is `tanh`
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv] + dila_conv
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv] + dila_conv # for the innermost, no submodule, and delete the bn
            if dilation:
                up = [uprelu, nn.Conv2d(inner_nc, outer_nc, kernel_size = 3, stride = 1, padding = 1), upnorm]
            else:
                up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm] + dila_conv
            if dilation:
                up = [uprelu, nn.Conv2d(inner_nc * 2, outer_nc, kernel_size = 3, stride = 1, padding = 1), upnorm]
            else:
                up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            mask = F.interpolate(self.mask, size = x_latter.size()[2:])
            x_CA = self.contextual_attention(x_latter, mask)
            x_latter = torch.cat([x_latter, x_CA], 1)
            x_latter = self.combine_layer(x_latter)
            return torch.cat([x_latter, x], 1)  # cat in the C channel