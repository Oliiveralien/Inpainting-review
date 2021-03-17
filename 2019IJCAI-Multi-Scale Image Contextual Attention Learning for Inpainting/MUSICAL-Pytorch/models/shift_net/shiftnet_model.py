import torch
from torch.nn import functional as F
import util.util as util
from models import networks
from models.shift_net.base_model import BaseModel
import torch.nn as nn
from torchvision import models
import models.shift_net.LossNetwork as LossNetwork


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

class ShiftNetModel(BaseModel):
    def name(self):
        return 'ShiftNetModel'


    def create_random_mask(self):
        if self.mask_type == 'random':
            if self.opt.mask_sub_type == 'fractal':
                mask = util.create_walking_mask ()  # create an initial random mask.

            elif self.opt.mask_sub_type == 'rect':
                mask = util.create_rand_mask ()

            elif self.opt.mask_sub_type == 'island':
                mask = util.wrapper_gmask (self.opt)
        return mask

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.opt.show_flow:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'flow_srcs']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.loss_Network = VGG16FeatureExtractor()
        self.loss_Network.cuda()
        # batchsize should be 1 for mask_global
        self.mask_global = torch.ByteTensor(1, 1, \
                                 opt.fineSize, opt.fineSize)

        # Here we need to set an artificial mask_global(not to make it broken, so center hole is ok.)
        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1
        self.TV_loss = LossNetwork.TVLoss()
        self.mask_type = opt.mask_type
        self.gMask_opts = {}
        self.fixed_mask = opt.fixed_mask if opt.mask_type == 'center' else 0
        if opt.mask_type == 'center':
            assert opt.fixed_mask == 1, "Center mask must be fixed mask!"

        if self.mask_type == 'random':
            self.create_random_mask()

        self.wgan_gp = False
        # added for wgan-gp
        if opt.gan_type == 'wgan_gp':
            self.gp_lambda = opt.gp_lambda
            self.ncritic = opt.ncritic
            self.wgan_gp = True


        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.to(self.device)

        # load/define networks
        # self.ng_innerCos_list is the constraint list in netG inner layers.
        # self.ng_mask_list is the mask list constructing shift operation.
        if opt.add_mask2input:
            input_nc = opt.input_nc + 1
        else:
            input_nc = opt.input_nc

        self.netG, self.ng_innerCos_list, self.ng_shift_list = networks.define_G(input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain) # add opt, we need opt.shift_sz and other stuffs
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion
            # don't use cGAN
            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            if self.wgan_gp:
                opt.beta1 = 0
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

        self.print_networks(opt.verbose)

    def set_input(self, input):
        real_A = input['A'].to(self.device)
        real_B = input['B'].to(self.device)

        # Add mask to real_A
        # When the mask is random, or the mask is not fixed, we all need to create_gMask
        if self.fixed_mask:
            if self.opt.mask_type == 'center':
                self.mask_global.zero_()
                self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                    int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1
            elif self.opt.mask_type == 'random':
                self.mask_global = self.create_random_mask().type_as(self.mask_global)
            else:
                raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)
        else:
            self.mask_global = self.create_random_mask().type_as(self.mask_global)

        self.set_latent_mask(self.mask_global, 3)

        #print(torch.max(real_A), torch.min(real_A))

        real_A.narrow(1,0,1).masked_fill_(self.mask_global, 0.)#2*123.0/255.0 - 1.0
        real_A.narrow(1,1,1).masked_fill_(self.mask_global, 0.)#2*104.0/255.0 - 1.0
        real_A.narrow(1,2,1).masked_fill_(self.mask_global, 0.)#2*117.0/255.0 - 1.0

        if self.opt.add_mask2input:
            # make it 4 dimensions.
            # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1.
            real_A = torch.cat((real_A, (1 - self.mask_global).expand(real_A.size(0), 1, real_A.size(2), real_A.size(3)).type_as(real_A)), dim=1)

        self.real_A = real_A
        self.real_B = real_B
        self.image_paths = input['A_paths']

    # TODO: it has not been implemented totally.
    def set_input_with_mask(self, input, mask):
        real_A = input['A'].to(self.device)
        real_B = input['B'].to(self.device)

        self.mask_global = mask

        self.set_latent_mask(mask, 3)

        real_A.narrow(1,0,1).masked_fill_(mask, 0.)#2*123.0/255.0 - 1.0
        real_A.narrow(1,1,1).masked_fill_(mask, 0.)#2*104.0/255.0 - 1.0
        real_A.narrow(1,2,1).masked_fill_(mask, 0.)#2*117.0/255.0 - 1.0

        self.real_A = real_A
        self.real_B = real_B
        self.image_paths = input['A_paths']       

    def set_latent_mask(self, mask_global, layer_to_last):
        for ng_shift in self.ng_shift_list: # ITERATE OVER THE LIST OF ng_shift_list
            ng_shift.set_mask(mask_global, layer_to_last)

    def set_gt_latent(self):
        if not self.opt.skip:
            if self.opt.add_mask2input:
                # make it 4 dimensions.
                # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1.
                real_B = torch.cat([self.real_B, (1 - self.mask_global).expand(self.real_B.size(0), 1, self.real_B.size(2), self.real_B.size(3)).type_as(self.real_B)], dim=1)
            else:
                real_B = self.real_B
            self.netG(real_B) # input ground truth

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    # Just assume one shift layer.
    def set_flow_src(self):
        self.flow_srcs = self.ng_shift_list[0].get_flow()
        self.flow_srcs = F.interpolate(self.flow_srcs, scale_factor=8, mode='nearest')
        # Just to avoid forgetting setting show_map_false
        self.set_show_map_false()

    # Just assume one shift layer.
    def set_show_map_true(self):
        self.ng_shift_list[0].set_flow_true()

    def set_show_map_false(self):
        self.ng_shift_list[0].set_flow_false()

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_AB = self.fake_B
        # Real
        real_AB = self.real_B # GroundTruth

        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)


        self.pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        self.loss_D_real = self.criterionGAN (self.pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * 0.5
        self.loss_D.backward()

    def calculate_style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = None
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            b, c, h, w = A_feat.size()
            B_feat = B_feats[i]
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))

            if i == 0:
                loss_value = torch.mean(torch.abs(A_style - B_style)) / (c * h * w)
            else:
                loss_value += torch.mean(torch.abs(A_style - B_style)) / (c * h * w)
        return loss_value
    
    def calculate_preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = None
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]            
            if i == 0:
                loss_value = torch.mean(torch.abs(A_feat - B_feat))
            else:
                loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
    
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B
        real_AB = self.real_B
        mask = self.mask_global.float().cuda()
        comp_AB = fake_AB*mask + real_AB * (1 - mask)
        
        real_B_feat = self.loss_Network(real_AB)
        fake_B_feat = self.loss_Network(fake_AB)
        comp_B_feat = self.loss_Network(comp_AB)
        
        content_loss = self.calculate_preceptual_loss(real_B_feat, fake_B_feat) + self.calculate_preceptual_loss(real_B_feat, comp_B_feat)
        style_loss = self.calculate_style_loss(real_B_feat, fake_B_feat) +  self.calculate_style_loss(real_B_feat, comp_B_feat)
                
        tv_loss = self.TV_loss(fake_AB)
        
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.3
        self.loss_G_L1 = 0
        self.loss_G_L1 += self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        hole_loss = torch.mean(torch.abs((fake_AB - real_AB) * mask))
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight

        self.loss_G += (tv_loss * self.opt.tv_weight 
                        + style_loss * self.opt.style_weight
                        + content_loss * self.opt.preceptual_weight
                        + hole_loss * self.opt.hole_weight) 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.ncritic = 1
        for i in range(self.ncritic):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


