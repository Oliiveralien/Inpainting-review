from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F 
import functools 
from torch.autograd import Variable 
import numpy as np 


############################################################
### Functions
############################################################
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in') 
        m.weight.data *= 0.1
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('ConvTranspose2d') != -1: 
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_() 
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_() 

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type) 
    return norm_layer 

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params) 
    print('--------------------------------------------------------------') 
    return num_params 

def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance', gpu_ids=[]): 
    netG = ImageTinker2(input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=4, norm_layer=nn.BatchNorm2d, pad_type='replicate') 

    num_params = print_network(netG) 

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0]) 
    netG.apply(weights_init) 

    return netG, num_params 

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]): 
    norm_layer = get_norm_layer(norm_type=norm) 
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat) 
    num_params = print_network(netD) 

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0]) 
    netD.apply(weights_init) 

    return netD, num_params 

class ImageTinker2(nn.Module): 
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=4, norm_layer=nn.InstanceNorm2d, pad_type='replicate', activation=nn.LeakyReLU(0.2, True)): 
        assert(n_blocks >= 0)
        super(ImageTinker2, self).__init__() 

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d 
        elif pad_type == 'zero': 
            self.pad = nn.ZeroPad2d 
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d 

        # LR coarse tinker (encoder) 
        lr_coarse_tinker = [self.pad(3), nn.Conv2d(input_nc, ngf // 2, kernel_size=7, stride=1, padding=0), activation] 
        lr_coarse_tinker += [self.pad(1), nn.Conv2d(ngf // 2, ngf, kernel_size=4, stride=2, padding=0), activation] 
        lr_coarse_tinker += [self.pad(1), nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=0), activation] 
        lr_coarse_tinker += [self.pad(1), nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=0), activation] 
        # bottle neck
        lr_coarse_tinker += [MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None)] 
        lr_coarse_tinker += [MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None)] 
        lr_coarse_tinker += [MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None)] 
        lr_coarse_tinker += [MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None)] 
        lr_coarse_tinker += [MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None)] 
        lr_coarse_tinker += [MultiDilationResnetBlock(ngf * 4, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None)] 
        # decoder 
        lr_coarse_tinker += [nn.UpsamplingBilinear2d(scale_factor=2), self.pad(1), nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=0), activation] 
        lr_coarse_tinker += [nn.UpsamplingBilinear2d(scale_factor=2), self.pad(1), nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=0), activation] 
        lr_coarse_tinker += [nn.UpsamplingBilinear2d(scale_factor=2), self.pad(1), nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=0), activation] 
        lr_coarse_tinker += [self.pad(3), nn.Conv2d(ngf // 2, output_nc, kernel_size=7, stride=1, padding=0)] 
        ### get a coarse (256x256x3) 
        self.lr_coarse_tinker = nn.Sequential(*lr_coarse_tinker) 

        self.r_en_padd1 = self.pad(3) 
        self.r_en_conv1 = nn.Conv2d(input_nc, ngf // 2, kernel_size=7, stride=1, padding=0) 
        self.r_en_acti1 = activation 

        self.r_en_padd2 = self.pad(1) 
        self.r_en_conv2 = nn.Conv2d(ngf // 2, ngf, kernel_size=4, stride=2, padding=0) 
        self.r_en_acti2 = activation 

        self.r_en_padd3 = self.pad(1) 
        self.r_en_conv3 = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=0) 
        self.r_en_acti3 = activation 

        self.r_en_skp_padd3 = self.pad(1) 
        self.r_en_skp_conv3 = nn.Conv2d(ngf * 2, ngf * 2 // 2, kernel_size=3, stride=1, padding=0) 
        self.r_en_skp_acti3 = activation 

        self.r_en_padd4 = self.pad(1) 
        self.r_en_conv4 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=0) 
        self.r_en_acti4 = activation 

        self.r_en_skp_padd4 = self.pad(1) 
        self.r_en_skp_conv4 = nn.Conv2d(ngf * 4, ngf * 4 // 2, kernel_size=3, stride=1, padding=0) 
        self.r_en_skp_acti4 = activation 

        self.r_en_padd5 = self.pad(1) 
        self.r_en_conv5 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=0) 
        self.r_en_acti5 = activation 

        self.r_md_mres1 = MultiDilationResnetBlock_v3(ngf * 8, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None) 
        self.r_md_mres2 = MultiDilationResnetBlock_v3(ngf * 8, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None) 
        self.r_md_mres5 = MultiDilationResnetBlock_v3(ngf * 8, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None) 
        self.r_md_satn1 = NonLocalBlock(ngf * 8, sub_sample=False, bn_layer=False) 
        self.r_md_mres3 = MultiDilationResnetBlock_v3(ngf * 8, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None) 
        self.r_md_mres4 = MultiDilationResnetBlock_v3(ngf * 8, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None) 
        self.r_md_mres6 = MultiDilationResnetBlock_v3(ngf * 8, kernel_size=3, stride=1, padding=1, pad_type='replicate', norm=None) 

        self.r_de_upbi1 = nn.UpsamplingBilinear2d(scale_factor=2) 
        self.r_de_padd1 = self.pad(1) 
        self.r_de_conv1 = nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=0) 
        self.r_de_acti1 = activation 

        self.r_de_satn2 = NonLocalBlock(ngf * 4 // 2, sub_sample=False, bn_layer=False) 
        self.r_de_satn3 = NonLocalBlock(ngf * 2 // 2, sub_sample=False, bn_layer=False) 

        self.r_de_mix_padd1 = self.pad(1) 
        self.r_de_mix_conv1 = nn.Conv2d(ngf * 4 + ngf * 4 // 2, ngf * 4, kernel_size=3, stride=1, padding=0) 
        self.r_de_mix_acti1 = activation 

        self.r_de_upbi2 = nn.UpsamplingBilinear2d(scale_factor=2) 
        self.r_de_padd2 = self.pad(1) 
        self.r_de_conv2 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=0) 
        self.r_de_acti2 = activation 

        self.r_de_mix_padd2 = self.pad(1) 
        self.r_de_mix_conv2 = nn.Conv2d(ngf * 2 + ngf * 2 // 2, ngf * 2, kernel_size=3, stride=1, padding=0) 
        self.r_de_mix_acti2 = activation 

        self.r_de_padd2_lr = self.pad(1) 
        self.r_de_conv2_lr = nn.Conv2d(ngf * 2, ngf // 2, kernel_size=3, stride=1, padding=0) 
        self.r_de_acti2_lr = activation 

        self.r_de_padd3_lr = self.pad(1) 
        self.r_de_conv3_lr = nn.Conv2d(ngf // 2, output_nc, kernel_size=3, stride=1, padding=0) 

        self.r_de_upbi3 = nn.UpsamplingBilinear2d(scale_factor=2) 
        self.r_de_padd3 = self.pad(1) 
        self.r_de_conv3 = nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=0) 
        self.r_de_acti3 = activation 

        self.r_de_upbi4 = nn.UpsamplingBilinear2d(scale_factor=2) 
        self.r_de_padd4 = self.pad(1) 
        self.r_de_conv4 = nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=0) 
        self.r_de_acti4 = activation 

        self.r_de_padd5 = self.pad(3) 
        self.r_de_conv5 = nn.Conv2d(ngf // 2, output_nc, kernel_size=7, stride=1, padding=0) 

        self.r_de_padd5_lr_alpha = self.pad(1) 
        self.r_de_conv5_lr_alpha = nn.Conv2d(ngf // 2, 1, kernel_size=3, stride=1, padding=0) 
        self.r_de_acti5_lr_alpha = nn.Sigmoid() 

        self.up   = nn.UpsamplingBilinear2d(scale_factor=4)
        self.down = nn.UpsamplingBilinear2d(scale_factor=0.25) 

    def forward(self, msked_img, msk, real_img=None, real_seg=None): 
        if real_img is not None: 
            rimg = real_img 
            inp = real_img * (1 - msk) + msk 
        else:
            rimg = msked_img 
            inp = msked_img 
        
        x = torch.cat((inp, msk, real_seg), dim=1) 
        lr_x = self.lr_coarse_tinker(x) 
        hr_x = lr_x * msk + rimg * (1 - msk) 

        y = torch.cat((hr_x, msk, real_seg), dim=1) 
        e1 = self.r_en_acti1(self.r_en_conv1(self.r_en_padd1(y))) 
        e2 = self.r_en_acti2(self.r_en_conv2(self.r_en_padd2(e1))) 
        e3 = self.r_en_acti3(self.r_en_conv3(self.r_en_padd3(e2))) 
        e4 = self.r_en_acti4(self.r_en_conv4(self.r_en_padd4(e3))) 
        e5 = self.r_en_acti5(self.r_en_conv5(self.r_en_padd5(e4))) 

        skp_e3 = self.r_en_skp_acti3(self.r_en_skp_conv3(self.r_en_skp_padd3(e3)))
        skp_e4 = self.r_en_skp_acti4(self.r_en_skp_conv4(self.r_en_skp_padd4(e4))) 

        de3 = self.r_de_satn3(skp_e3) 
        de4 = self.r_de_satn2(skp_e4) 

        m1 = self.r_md_mres1(e5)
        m2 = self.r_md_mres2(m1) 
        m5 = self.r_md_mres5(m2)
        a1 = self.r_md_satn1(m5)
        m3 = self.r_md_mres3(a1)
        m4 = self.r_md_mres4(m3) 
        m6 = self.r_md_mres6(m4)

        d1 = self.r_de_acti1(self.r_de_conv1((self.r_de_padd1(self.r_de_upbi1(m6))))) # 32x32x256
        cat1 = torch.cat((d1, de4), dim=1) 
        md1 = self.r_de_mix_acti1(self.r_de_mix_conv1(self.r_de_mix_padd1(cat1))) 

        d2 = self.r_de_acti2(self.r_de_conv2((self.r_de_padd2(self.r_de_upbi2(md1))))) # 64x64x128
        cat2 = torch.cat((d2, de3), dim=1) 
        md2 = self.r_de_mix_acti2(self.r_de_mix_conv2(self.r_de_mix_padd2(cat2))) 

        d2_lr = self.r_de_acti2_lr(self.r_de_conv2_lr(self.r_de_padd2_lr(md2))) 
        d3_lr = self.r_de_conv3_lr(self.r_de_padd3_lr(d2_lr))

        d3 = self.r_de_acti3(self.r_de_conv3((self.r_de_padd3(self.r_de_upbi3(md2))))) # 128x128x64
        d4 = self.r_de_acti4(self.r_de_conv4((self.r_de_padd4(self.r_de_upbi4(d3))))) # 256x256x32

        d5 = self.r_de_conv5(self.r_de_padd5(d4)) 
        d5_lr_alpha = self.r_de_acti5_lr_alpha(self.r_de_conv5_lr_alpha(self.r_de_padd5_lr_alpha(d4))) 

        ###
        # d5: 256x256x3 
        # d5_lr_alpha: 256x256x1
        # d3_lr: 64x64x3
        ###
        lr_img = d3_lr
        
        #reconst_img = d5
        d5 = d5 * msk + rimg * (1 - msk)
        lr_d5 = self.down(d5)
        lr_d5_res = d3_lr - lr_d5 
        hr_d5_res = self.up(lr_d5_res) 
        reconst_img = d5 + hr_d5_res * d5_lr_alpha  
        compltd_img = reconst_img * msk + rimg * (1 - msk) 
        #out = compltd_img + hr_d5_res * d5_lr_alpha 

        return compltd_img, reconst_img, lr_x, lr_img 


############################################################
### Losses
############################################################
class TVLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3] 

class VGGLoss(nn.Module):
    # vgg19 perceptual loss
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.mse_loss = nn.MSELoss() 

        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda() 
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda() 
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def gram_matrix(self, x):
        (b, ch, h, w) = x.size() 
        features = x.view(b, ch, w*h) 
        features_t = features.transpose(1, 2) 
        gram = features.bmm(features_t) / (ch * h * w) 
        return gram 

    def forward(self, x, y):
        x = (x - self.mean) / self.std 
        y = (y - self.mean) / self.std 
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0
        style_loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
            gm_x = self.gram_matrix(x_vgg[i]) 
            gm_y = self.gram_matrix(y_vgg[i]) 
            style_loss += self.weights[i] * self.mse_loss(gm_x, gm_y.detach()) 
        return loss, style_loss 

class GANLoss_D_v2(nn.Module): 
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor): 
        super(GANLoss_D_v2, self).__init__() 
        self.real_label = target_real_label 
        self.fake_label = target_fake_label 
        self.real_label_var = None 
        self.fake_label_var = None 
        self.Tensor = tensor 
        if use_lsgan: 
            self.loss = nn.MSELoss() 
        else:
            def wgan_loss(input, target):
                return torch.mean(F.relu(1.-input)) if target else torch.mean(F.relu(1.+input)) 
            self.loss = wgan_loss 
        
    def get_target_tensor(self, input, target_is_real): 
        target_tensor = None 
        if target_is_real: 
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label) 
                self.real_label_var = Variable(real_tensor, requires_grad=False) 
            target_tensor = self.real_label_var 
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel())) 
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label) 
                self.fake_label_var = Variable(fake_tensor, requires_grad=False) 
            target_tensor = self.fake_label_var 
        return target_tensor 
        
    def __call__(self, input, target_is_real): 
        if isinstance(input[0], list): 
            loss = 0
            for input_i in input:
                pred = input_i[-1] 
                target_tensor = self.get_target_tensor(pred, target_is_real) 
                #loss += self.loss(pred, target_tensor) 
                loss += self.loss(pred, target_is_real) 
            return loss 
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real) 
            return self.loss(input[-1], target_tensor) 

class GANLoss_G_v2(nn.Module): 
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor): 
        super(GANLoss_G_v2, self).__init__() 
        self.real_label = target_real_label 
        self.fake_label = target_fake_label 
        self.real_label_var = None 
        self.fake_label_var = None 
        self.Tensor = tensor 
        if use_lsgan: 
            self.loss = nn.MSELoss() 
        else:
            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean() 
            self.loss = wgan_loss 
        
    def get_target_tensor(self, input, target_is_real): 
        target_tensor = None 
        if target_is_real: 
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label) 
                self.real_label_var = Variable(real_tensor, requires_grad=False) 
            target_tensor = self.real_label_var 
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel())) 
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label) 
                self.fake_label_var = Variable(fake_tensor, requires_grad=False) 
            target_tensor = self.fake_label_var 
        return target_tensor 
        
    def __call__(self, input, target_is_real): 
        if isinstance(input[0], list): 
            loss = 0
            for input_i in input:
                pred = input_i[-1] 
                target_tensor = self.get_target_tensor(pred, target_is_real) 
                #loss += self.loss(pred, target_tensor) 
                loss += self.loss(pred, target_is_real)
            return loss 
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real) 
            return self.loss(input[-1], target_tensor) 


# Define the PatchGAN discriminator with the specified arguments. 
class NLayerDiscriminator(nn.Module): 
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, getIntermFeat=False): 
        super(NLayerDiscriminator, self).__init__() 
        self.getIntermFeat = getIntermFeat 
        self.n_layers = n_layers 
        
        kw = 4
        padw = int(np.ceil((kw-1.0)/2)) 
        sequence = [[SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]] 
        
        nf = ndf 
        for n in range(1, n_layers): 
            nf_prev = nf 
            nf = min(nf * 2, 512) 
            sequence += [[
                SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)), 
                nn.LeakyReLU(0.2, True) 
            ]] 
        
        nf_prev = nf 
        nf = min(nf * 2, 512) 
        sequence += [[
            SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)), 
            nn.LeakyReLU(0.2, True) 
        ]] 
        
        sequence += [[SpectralNorm(nn.Conv2d(nf, nf, kernel_size=kw, stride=1, padding=padw))]]
                
        # if use_sigmoid: 
        #     sequence += [[nn.Sigmoid()]] 
        
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n])) 
        else: 
            sequence_stream = [] 
            for n in range(len(sequence)):
                sequence_stream += sequence[n] 
            self.model = nn.Sequential(*sequence_stream)
        
    def forward(self, input): 
        if self.getIntermFeat:
            res = [input] 
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model'+str(n)) 
                res.append(model(res[-1])) 
            return res[1:]
        else:
            return self.model(input) 


# Define the Multiscale Discriminator. 
class MultiscaleDiscriminator(nn.Module): 
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False): 
        super(MultiscaleDiscriminator, self).__init__() 
        self.num_D = num_D 
        self.n_layers = n_layers 
        self.getIntermFeat = getIntermFeat 
        
        for i in range(num_D): 
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat) 
            if getIntermFeat: 
                for j in range(n_layers+2): 
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j))) 
            else:
                setattr(self, 'layer'+str(i), netD.model) 
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False) 
    
    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input] 
            for i in range(len(model)):
                result.append(model[i](result[-1])) 
            return result[1:]
        else:
            return [model(input)] 
    
    def forward(self, input): 
        num_D = self.num_D 
        result = [] 
        input_downsampled = input 
        for i in range(num_D):
            if self.getIntermFeat: 
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)] 
            else: 
                model = getattr(self, 'layer'+str(num_D-1-i)) 
            result.append(self.singleD_forward(model, input_downsampled)) 
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled) 
        return result 
            
        
### Define Vgg19 for vgg_loss 
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 20):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 29):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # fixed pretrained vgg19 model for feature extraction
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out 

### Multi-Dilation ResnetBlock
class MultiDilationResnetBlock(nn.Module): 
    def __init__(self, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, pad_type='reflect', norm='instance', acti='relu', use_dropout=False): 
        super(MultiDilationResnetBlock, self).__init__() 

        self.branch1 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch2 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=3, dilation=3, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch3 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=4, dilation=4, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch4 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=5, dilation=5, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch5 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=6, dilation=6, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch6 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=8, dilation=8, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch7 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=10, dilation=10, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch8 = ConvBlock(input_nc, input_nc // 8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 

        self.fusion9 = ConvBlock(input_nc, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, pad_type=pad_type, norm=norm, acti=None) 

    def forward(self, x):
        d1 = self.branch1(x) 
        d2 = self.branch2(x) 
        d3 = self.branch3(x) 
        d4 = self.branch4(x) 
        d5 = self.branch5(x) 
        d6 = self.branch6(x) 
        d7 = self.branch7(x) 
        d8 = self.branch8(x) 
        d9 = torch.cat((d1, d2, d3, d4, d5, d6, d7, d8), dim=1) 
        out = x + self.fusion9(d9) 
        return out 

### Multi-Dilation ResnetBlock
class MultiDilationResnetBlock_v3(nn.Module): 
    def __init__(self, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, pad_type='reflect', norm='instance', acti='relu', use_dropout=False): 
        super(MultiDilationResnetBlock_v3, self).__init__() 

        self.branch1 = ConvBlock(input_nc, input_nc // 4, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch2 = ConvBlock(input_nc, input_nc // 4, kernel_size=3, stride=1, padding=3, dilation=3, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch3 = ConvBlock(input_nc, input_nc // 4, kernel_size=3, stride=1, padding=4, dilation=4, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        self.branch4 = ConvBlock(input_nc, input_nc // 4, kernel_size=3, stride=1, padding=5, dilation=5, groups=1, bias=True, pad_type=pad_type, norm=norm, acti='relu') 
        
        self.fusion5 = ConvBlock(input_nc, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, pad_type=pad_type, norm=norm, acti=None) 

    def forward(self, x):
        d1 = self.branch1(x) 
        d2 = self.branch2(x) 
        d3 = self.branch3(x) 
        d4 = self.branch4(x)  
        d5 = torch.cat((d1, d2, d3, d4), dim=1) 
        out = x + self.fusion5(d5) 
        return out 

### ResnetBlock
class ResnetBlock(nn.Module): 
    def __init__(self, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, pad_type='reflect', norm='instance', acti='relu', use_dropout=False): 
        super(ResnetBlock, self).__init__() 
        self.conv_block = self.build_conv_block(input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti, use_dropout)


    def build_conv_block(self, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti, use_dropout):
        conv_block = [] 
        conv_block += [ConvBlock(input_nc, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti='relu')]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)] 
        conv_block += [ConvBlock(input_nc, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti=None)] 

        return nn.Sequential(*conv_block) 

    def forward(self, x):
        out = x + self.conv_block(x) 
        return out 

### ResnetBlock
class ResnetBlock_v2(nn.Module): 
    def __init__(self, input_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, pad_type='reflect', norm='instance', acti='relu', use_dropout=False): 
        super(ResnetBlock_v2, self).__init__() 
        self.conv_block = self.build_conv_block(input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti, use_dropout)


    def build_conv_block(self, input_nc, kernel_size, stride, padding, dilation, groups, bias, pad_type, norm, acti, use_dropout):
        conv_block = [] 
        conv_block += [ConvBlock(input_nc, input_nc, kernel_size=3, stride=1, padding=padding, dilation=dilation, groups=groups, bias=bias, pad_type=pad_type, norm=norm, acti='elu')]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)] 
        conv_block += [ConvBlock(input_nc, input_nc, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, pad_type='reflect', norm='instance', acti=None)] 

        return nn.Sequential(*conv_block) 

    def forward(self, x):
        out = x + self.conv_block(x) 
        return out 

### NonLocalBlock2D 
class NonLocalBlock(nn.Module): 
    def __init__(self, input_nc, inter_nc=None, sub_sample=True, bn_layer=True): 
        super(NonLocalBlock, self).__init__() 
        self.input_nc = input_nc 
        self.inter_nc = inter_nc 

        if inter_nc is None: 
            self.inter_nc = input_nc // 2

        self.g = nn.Conv2d(in_channels=self.input_nc, out_channels=self.inter_nc, kernel_size=1, stride=1, padding=0) 

        if bn_layer: 
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_nc, out_channels=self.input_nc, kernel_size=1, stride=1, padding=0), 
                nn.BatchNorm2d(self.input_nc) 
            )
            self.W[0].weight.data.zero_()
            self.W[0].bias.data.zero_() 
        else:
            self.W = nn.Conv2d(in_channels=self.inter_nc, out_channels=self.input_nc, kernel_size=1, stride=1, padding=0) 
            self.W.weight.data.zero_()
            self.W.bias.data.zero_() 

        self.theta = nn.Conv2d(in_channels=self.input_nc, out_channels=self.inter_nc, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.input_nc, out_channels=self.inter_nc, kernel_size=1, stride=1, padding=0) 

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size(2, 2))) 

    def forward(self, x): 
        batch_size = x.size(0) 

        g_x = self.g(x).view(batch_size, self.inter_nc, -1) 
        g_x = g_x.permute(0, 2, 1) 

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1) 
        theta_x = theta_x.permute(0, 2, 1) 

        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1) 

        f = torch.matmul(theta_x, phi_x) 
        f_div_C = F.softmax(f, dim=-1) 

        y = torch.matmul(f_div_C, g_x) 
        y = y.permute(0, 2, 1).contiguous() 
        y = y.view(batch_size, self.inter_nc, *x.size()[2:]) 
        W_y = self.W(y) 

        z = W_y + x
        return z 

### ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, pad_type='zero', norm=None, acti='lrelu'):
        super(ConvBlock, self).__init__() 
        self.use_bias = bias 

        # initialize padding 
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding) 
        elif pad_type == 'replicate': 
            self.pad = nn.ReplicationPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type) 

        # initialize normalization
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(output_nc) 
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(output_nc) 
        elif norm is None or norm == 'spectral': 
            self.norm = None 
        else: 
            assert 0, "Unsupported normalization: {}".format(norm) 

        # initialize activation
        if acti == 'relu':
            self.acti = nn.ReLU(inplace=True) 
        elif acti == 'lrelu':
            self.acti = nn.LeakyReLU(0.2, inplace=True) 
        elif acti == 'prelu':
            self.acti = nn.PReLU() 
        elif acti == 'elu':
            self.acti = nn.ELU() 
        elif acti == 'tanh':
            self.acti = nn.Tanh() 
        elif acti == 'sigmoid':
            self.acti = nn.Sigmoid() 
        elif acti is None:
            self.acti = None 
        else: 
            assert 0, "Unsupported activation: {}".format(acti) 

        # initialize convolution 
        if norm == 'spectral': 
            self.conv = SpectralNorm(nn.Conv2d(input_nc, output_nc, kernel_size, stride, dilation=dilation, groups=groups, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, dilation=dilation, groups=groups, bias=self.use_bias) 

    def forward(self, x):
        x = self.conv(self.pad(x)) 
        if self.norm:
            x = self.norm(x)
        if self.acti:
            x = self.acti(x)
        return x 

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps) 

### SpectralNorm 
class SpectralNorm(nn.Module):
    """
    Spectral Normalization for Generative Adversarial Networks
    Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan 
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__() 
        self.module = module 
        self.name = name 
        self.power_iterations = power_iterations 
        if not self._made_params():
            self._make_params() 
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u") 
        v = getattr(self.module, self.name + "_v") 
        w = getattr(self.module, self.name + "_bar") 

        height = w.data.shape[0] 
        for _ in range(self.power_iterations): 
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data)) 
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data)) 

        sigma = u.dot(w.view(height, -1).mv(v)) 
        setattr(self.module, self.name, w / sigma.expand_as(w)) 

    def _made_params(self):
        try: 
            u = getattr(self.module, self.name + "_u") 
            v = getattr(self.module, self.name + "_v") 
            w = getattr(self.module, self.name + "_bar") 
            return True 
        except AttributeError:
            return False 

    def _make_params(self): 
        w = getattr(self.module, self.name) 

        height = w.data.shape[0] 
        width = w.view(height, -1).data.shape[1] 

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False) 
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False) 
        u.data = l2normalize(u.data) 
        v.data = l2normalize(v.data) 
        w_bar = nn.Parameter(w.data) 

        del self.module._parameters[self.name] 

        self.module.register_parameter(self.name + "_u", u) 
        self.module.register_parameter(self.name + "_v", v) 
        self.module.register_parameter(self.name + "_bar", w_bar) 

    def forward(self, *args):
        self._update_u_v() 
        return self.module.forward(*args) 

