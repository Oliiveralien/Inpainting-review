import numpy as np 
import torch 
import torch.nn as nn
import os 
from torch.autograd import Variable 
#from util.image_pool import ImagePool 
# BaseModel, save_network & load_network 
from .base_model import BaseModel 
from . import networks 

class OurModel(BaseModel):
    def name(self):
        return 'OurModel' 

    def get_num_params(self): 
        return self.num_params_G, self.num_params_D 
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss): 
        flags = (True, True, True, True, use_gan_feat_loss, use_vgg_loss, True, True, True) 
        def loss_filter(g_gan, g_coarse_l1, g_out_l1, g_style, g_gan_feat, g_vgg, g_tv, d_real, d_fake): 
            return [l for (l, f) in zip((g_gan, g_coarse_l1, g_out_l1, g_style, g_gan_feat, g_vgg, g_tv, d_real, d_fake), flags) if f] 
        return loss_filter 

    def initialize(self, opt): 
        BaseModel.initialize(self, opt) 
        self.isTrain = opt.isTrain 
        input_nc = opt.input_nc ### masked_img + mask + seg (RGB + gray + Seg) 7 channels 

        ### define networks
        # Generator 
        netG_input_nc = input_nc 
        self.netG, self.num_params_G = networks.define_G(netG_input_nc, opt.output_nc, ngf=64, gpu_ids=self.gpu_ids) 

        # Discriminator 
        if self.isTrain: 
            use_sigmoid = opt.no_lsgan 
            netD_input_nc = opt.output_nc * 2 
            self.netD, self.num_params_D = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids) 
            # for param in self.netD.parameters(): 
            #     param.requires_grad = False
        else:
            self.num_params_D = 0

        print('-------------------- Networks initialized --------------------') 

        ### load networks 
        if not self.isTrain or opt.continue_train or opt.load_pretrain: 
            pretrained_path = '' if not self.isTrain else opt.load_pretrain 
            print(pretrained_path) 

            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path) 
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path) 
        

        ### set loss functions and optimizers 
        if self.isTrain: 
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU") 
            # self.fake_pool = ImagePool(opt.pool_size) 
            self.old_lr = opt.lr 

            # define loss functions 
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss) 
 
            self.criterionGAN_D = networks.GANLoss_D_v2(use_lsgan=not opt.no_lsgan, tensor=self.Tensor) 
            self.criterionGAN_G = networks.GANLoss_G_v2(use_lsgan=not opt.no_lsgan, tensor=self.Tensor) 

            self.criterionFeat = torch.nn.L1Loss() 
            if not opt.no_vgg_loss: 
                self.criterionVGG = networks.VGGLoss(self.gpu_ids) 
            self.criterionL1 = torch.nn.L1Loss() 
            self.down = nn.UpsamplingBilinear2d(scale_factor=0.25) 
            self.resize = nn.UpsamplingBilinear2d(size=(224, 224)) 
            self.criterionTV = networks.TVLoss() 

            # Names so we can breakout loss 
            self.loss_names = self.loss_filter('G_GAN', 'G_COARSE_L1', 'G_OUT_L1', 'G_STYLE', 'G_GAN_Feat', 'G_VGG', 'G_TV', 'D_real', 'D_fake') 

            # initialize optimizers
            # optimizer G 
            if opt.niter_fix_global > 0:
                import sys 
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set 
                    finetune_list = Set() 
                
                params_dict = dict(self.netG.named_parameters()) 
                params = [] 
                for key, value in params_dict.items(): 
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value] 
                        finetune_list.add(key.split('.')[0]) 
                print('--------------- Only training the local enhancer network (for %d epochs) ---------------' % opt.niter_fix_global) 
                print('The layers that are finetuned are ', sorted(finetune_list)) 
            else:
                params = list(self.netG.parameters())  
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999)) 

            # optimizer D 
            params = list(self.netD.parameters()) 
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr * 4.0, betas=(opt.beta1, 0.999)) 

    def encode_input(self, masked_image, mask, real_image, real_seg, infer=False): 
        input_msked_img = masked_image.data.cuda() 
        input_msk = mask.data.cuda() 

        input_msked_img = Variable(input_msked_img, volatile=infer) 
        input_msk = Variable(input_msk, volatile=infer) 

        real_image = Variable(real_image.data.cuda(), volatile=infer) 
        real_seg = Variable(real_seg.data.cuda(), volatile=infer)

        return input_msked_img, input_msk, real_image, real_seg 

    def encode_input_test(self, masked_image, mask, real_seg, infer=False): 
        input_msked_img = masked_image.data.cuda()
        input_msk = mask.data.cuda() 
        input_seg = real_seg.data.cuda() 
                
        input_msked_img = Variable(input_msked_img) 
        input_msk = Variable(input_msk) 
        input_seg = Variable(input_seg) 

        return input_msked_img, input_msk, input_seg 
    
    def discriminate(self, input_image, test_image, use_pool=False): 
        input_concat = torch.cat((input_image, test_image.detach()), dim=1) 
        return self.netD.forward(input_concat) 
    
    def forward(self, masked_img, mask, real_img=None, real_seg=None, infer=False, mode=None): 
        # inference 
        if mode == 'inference': 
            compltd_img, reconst_img, lr_x = self.inference(masked_img, mask, real_seg) 
            return compltd_img, reconst_img, lr_x 
    
        # Encode inputs 
        input_msked_img, input_msk, real_image, real_seg = self.encode_input(masked_img, mask, real_img, real_seg) 

        # Fake Generation 
        compltd_img, reconst_img, lr_x, lr_img = self.netG.forward(input_msked_img, input_msk, real_image, real_seg) 
        lr_msk = input_msk
        
        msk025 = self.down(input_msk) 
        img025 = self.down(real_image)

        # Fake Detection and Loss 
        pred_fake = self.discriminate(input_msked_img, compltd_img) 
        loss_D_fake = self.criterionGAN_D(pred_fake, False) 
        
        # Real Detection and Loss 
        pred_real = self.discriminate(input_msked_img, real_image) 
        loss_D_real = self.criterionGAN_D(pred_real, True) 

        # GAN Loss (Fake pass-ability loss) 
        pred_fake = self.netD.forward(torch.cat((input_msked_img, compltd_img), dim=1)) 
        loss_G_GAN = self.criterionGAN_G(pred_fake, True) 
        loss_G_GAN *= self.opt.lambda_gan 

        # GAN L1 Loss 
        loss_G_GAN_L1 = self.criterionL1(reconst_img * input_msk, real_image * input_msk) * self.opt.lambda_l1 
        loss_G_GAN_L1 += self.criterionL1(reconst_img * (1 - input_msk), real_image * (1 - input_msk))  
        loss_G_GAN_L1 += self.criterionL1(lr_img * msk025, img025 * msk025) * self.opt.lambda_l1 
        loss_G_GAN_L1 += self.criterionL1(lr_img * (1 - msk025), img025 * (1 - msk025)) 

        loss_G_COARSE_L1 = self.criterionL1(lr_x * lr_msk, real_image * lr_msk) * self.opt.lambda_l1 
        loss_G_COARSE_L1 += self.criterionL1(lr_x * (1 - lr_msk), real_image * (1 - lr_msk)) 

        loss_G_TV = self.criterionTV(compltd_img) * self.opt.lambda_tv 

        # GAN feature matching loss 
        loss_G_GAN_Feat = 0 
        if not self.opt.no_ganFeat_loss: 
            feat_weights = 4.0 / (self.opt.n_layers_D + 1) 
            D_weights = 1.0 / self.opt.num_D 
            for i in range(self.opt.num_D): 
                for j in range(len(pred_fake[i]) - 1): 
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) 
        loss_G_GAN_Feat *= self.opt.lambda_feat 

        # VGG feature matching loss 
        loss_G_VGG = 0 
        if not self.opt.no_vgg_loss: 
            resized_reconst_img = self.resize(reconst_img)
            resized_compltd_img = self.resize(compltd_img)
            resized_real_img = self.resize(real_image) 
            loss_G_VGG, loss_G_VGGStyle = self.criterionVGG(resized_reconst_img, resized_real_img) 
            loss_G_VGG2, loss_G_VGGStyle2 = self.criterionVGG(resized_compltd_img, resized_real_img) 
        loss_G_VGG += loss_G_VGG2
        loss_G_VGGStyle += loss_G_VGGStyle2
        loss_G_VGG *= self.opt.lambda_vgg 
        loss_G_VGGStyle *= self.opt.lambda_style 
  
        # only return the fake image if necessary 
        return [ self.loss_filter( loss_G_GAN, loss_G_COARSE_L1, loss_G_GAN_L1, loss_G_VGGStyle, loss_G_GAN_Feat, loss_G_VGG, loss_G_TV, loss_D_real, loss_D_fake ), None if not infer else compltd_img, reconst_img, lr_x ] 

    def inference(self, masked_img, mask, real_seg): 
        # Encode inputs 
        input_msked_img, input_msk, input_seg = self.encode_input_test(Variable(masked_img), Variable(mask), Variable(real_seg), infer=True) 

        # Fake Generation 
        if torch.__version__.startswith('0.4'): 
            with torch.no_grad(): 
                compltd_img, reconst_img, lr_x, lr_img = self.netG.forward(input_msked_img, input_msk, real_seg=input_seg) 
        else: 
            compltd_img, reconst_img, lr_x, lr_img = self.netG.forward(input_msked_img, input_msk, real_seg=input_seg)
        return compltd_img, reconst_img, lr_x 

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids) 
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids) 


    def update_fixed_params(self): 
        # after fixing the global generator for a # of iterations, also start finetuning it 
        params = list(self.netG.parameters()) 
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('----------------- Now also finetuning global generator -----------------') 

    def update_learning_rate(self): 
        lrd = self.opt.lr / self.opt.niter_decay 
        lr = self.old_lr - lrd 
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr * 4.0
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr 
        print('update learning rate: %f -> %f' % (self.old_lr, lr)) 
        self.old_lr = lr 

class InferenceModel(OurModel): 
    def forward(self, inp1, inp2, inp3): 
        masked_img = inp1 
        mask = inp2 
        seg = inp3 
        return self.inference(masked_img, mask, seg) 

