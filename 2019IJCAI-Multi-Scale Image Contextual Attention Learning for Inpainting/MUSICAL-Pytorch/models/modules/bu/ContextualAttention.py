import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, num_channel, squeeze_ratio = 1.0):
        super(SEModule, self).__init__()
        self.sequeeze_mod = nn.AdaptiveAvgPool2d(1)
        self.num_channel = num_channel
        
        blocks = [ nn.Linear(num_channel, int(num_channel*squeeze_ratio)),
                   nn.ReLU(),
                   nn.Linear(int(num_channel*squeeze_ratio), num_channel),
                   nn.Sigmoid()]
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        ori = x
        x = self.sequeeze_mod(x)
        x = x.view(x.size(0), 1, self.num_channel)
        x = self.blocks(x)
        x = x.view(x.size(0), self.num_channel, 1, 1)
        x = ori * x
        return x
    

class ContextualAttentionModule(nn.Module):
    
    def __init__(self, patch_size = 3, propagate_size = 3, stride = 1):
        super(ContextualAttentionModule, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        
    def forward(self, foreground, mask, background = "same"):
        ###assume the masked area has value 1
        bz, nc, w, h = foreground.size()
        if background == "same":
            background = foreground.clone()
        background = background * (1 - mask)
        background = F.pad(background, [self.patch_size//2, self.patch_size//2, self.patch_size//2, self.patch_size//2])
        conv_kernels_all = background.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride).contiguous().view(bz, nc, -1, self.patch_size, self.patch_size)
        conv_kernels_all = conv_kernels_all.transpose(2, 1)
        output_tensor = []
        for i in range(bz):
            feature_map = foreground[i:i+1]
            
            #form convolutional kernels
            conv_kernels = conv_kernels_all[i] + 0.0000001
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3], keepdim = True)**0.5
            conv_kernels = conv_kernels/norm_factor
            
            conv_result = F.conv2d(feature_map, conv_kernels, padding = self.patch_size//2)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.conv2d(conv_result, self.prop_kernels, stride = 1, padding = 1, groups = conv_result.size(1))
            attention_scores = F.softmax(conv_result, dim = 1)
            ##propagate the scores
            recovered_foreground = F.conv_transpose2d(attention_scores, conv_kernels, stride = 1, padding = self.patch_size//2)
            #average the recovered value, at the same time make non-masked area 0
            recovered_foreground = (recovered_foreground * mask)/(self.patch_size ** 2)
            #recover the image
            final_output = recovered_foreground + feature_map * (1 - mask)
            output_tensor.append(final_output)
        return torch.cat(output_tensor, dim = 0)
                
                
                
class ParallelContextualAttention(nn.Module):
    
    def __init__(self, inchannel, patch_size_list = [1, 3], propagate_size_list = [3, 3], stride_list = [1 ,1]):
        assert isinstance(patch_size_list, list), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(propagate_size_list) == len(stride_list), "the input_lists should have same lengths"
        super(ParallelContextualAttention, self).__init__()
        for i in range(len(patch_size_list)):
            name = "CA_{:d}".format(i)
            setattr(self, name, ContextualAttentionModule(patch_size_list[i], propagate_size_list[i], stride_list[i]))
        self.num_of_modules = len(patch_size_list)
        self.SqueezeExc = SEModule(inchannel * self.num_of_modules)
        self.combiner = nn.Conv2d(inchannel * self.num_of_modules, inchannel, kernel_size = 1)
        
    def forward(self, foreground, mask, background = "same"):
        outputs = []
        for i in range(self.num_of_modules):
            name = "CA_{:d}".format(i)
            CA_module = getattr(self, name)
            outputs.append(CA_module(foreground, mask, background))
        outputs = torch.cat(outputs, dim = 1)
        outputs = self.SqueezeExc(outputs)
        outputs = self.combiner(outputs)
        return outputs
    
class SerialContextualAttention(nn.Module):
    
    def __init__(self, inchannel, patch_size_list = [5, 3, 1], propagate_size_list = [3, 3, 3], stride_list = [1, 1, 1]):
        assert isinstance(patch_size_list, list), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(propagate_size_list) == len(stride_list), "the input_lists should have same lengths"
        super(SerialContextualAttention, self).__init__()
        modules = []
        for i in range(len(patch_size_list)):
            modules.append(ContextualAttentionModule(patch_size_list[i], propagate_size_list[i], stride_list[i]))
        self.blocks = modules
        self.SqueezeExc = SEModule(inchannel * len(patch_size_list))
        
    def forward(self, foreground, mask, background = "same"):
        outputs = [foreground]
        for block in self.blocks:
            outputs = [block(outputs[0], mask, background)] + outputs
        outputs = torch.cat(outputs[0:-1], dim = 1)
        outputs = self.SqueezeExc(outputs)
        return outputs