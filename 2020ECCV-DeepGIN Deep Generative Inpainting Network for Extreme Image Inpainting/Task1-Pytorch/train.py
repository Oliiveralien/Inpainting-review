from util.visualizer import Visualizer
import util.util as util
from data.data_loader import CreateDataLoader
from options.train_options import TrainOptions
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict

import fractions
def lcm(a, b):
    return abs(a*b) / fractions.gcd(a, b) if a and b else 0

from models.models import create_model 

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt') 
param_path = os.path.join(opt.checkpoints_dir, opt.name, 'param.txt') 

# continue training or start from scratch
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(
            iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)


########################################
# load dataset
########################################
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('# of training images = %d' % dataset_size)

########################################
# define model and optimizer
########################################
# define own model
model, num_params_G, num_params_D = create_model(opt) 
optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D 
# output the model size (i.e. num_params_G and num_params_D to txt file) 
np.savetxt(param_path, (num_params_G, num_params_D), delimiter=',', fmt='%d') 

########################################
# define visualizer
########################################
visualizer = Visualizer(opt) 

########################################
# define train and/val loop
########################################
total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    model.train() 
    # by default, start_epoch, epoch_iter = 1, 0
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        
        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############### Forward pass ###############
        # get pred and calculate loss
        B, C, H, W = data['masked_image'].shape 
        msk_img = data['masked_image'].view(-1, 3, H, W)
        msk = data['mask'].view(-1, 1, H, W)
        real_img = data['real_image'].view(-1, 3, H, W)

        losses, compltd_img, reconst_img, lr_x = model(Variable(msk_img), Variable(msk), 
                                                       Variable(real_img), infer=save_fake) 
        
        # sum per device losses 
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses] 
        loss_dict = dict(zip(model.module.loss_names, losses)) 

        # calculate final loss scalar 
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 
        loss_G = loss_dict['G_GAN'] + loss_dict['G_COARSE_L1'] + loss_dict['G_OUT_L1'] + loss_dict['G_TV'] + loss_dict['G_STYLE'] + loss_dict.get('G_VGG', 0) 

        ############### Backward pass ###############
        # update Generator parameters 
        optimizer_G.zero_grad() 
        loss_G.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer_G.step()

        # update Discriminator parameters 
        optimizer_D.zero_grad() 
        loss_D.backward() 
        optimizer_D.step() 

        ############### Display results and losses ###############
        # print out losses
        if total_steps % opt.print_freq == print_delta: 
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()} 
            t = (time.time() - iter_start_time) / opt.print_freq 
            visualizer.print_current_errors(epoch, epoch_iter, errors, t) 
            visualizer.plot_current_errors(errors, total_steps) 

        # display completed images
        if save_fake:
            visuals = OrderedDict([('real_image', util.tensor2im(real_img[0], normalize=False)), 
                                   ('masked_image_1', util.tensor2im(msk_img[0], normalize=False)),
                                   ('coarse_reconst_1', util.tensor2im(lr_x.data[0], normalize=False)), 
                                   ('output_image_1', util.tensor2im(reconst_img.data[0], normalize=False)), 
                                   ('completed_image_1', util.tensor2im(compltd_img.data[0], normalize=False)),
                                   ('masked_image_2', util.tensor2im(msk_img[1], normalize=False)), 
                                   ('coarse_reconst_2', util.tensor2im(lr_x.data[1], normalize=False)), 
                                   ('output_image_2', util.tensor2im(reconst_img.data[1], normalize=False)), 
                                   ('completed_image_2', util.tensor2im(compltd_img.data[1], normalize=False)),
                                   ('masked_image_3', util.tensor2im(msk_img[2], normalize=False)), 
                                   ('coarse_reconst_3', util.tensor2im(lr_x.data[2], normalize=False)), 
                                   ('output_image_3', util.tensor2im(reconst_img.data[2], normalize=False)), 
                                   ('completed_image_3', util.tensor2im(compltd_img.data[2], normalize=False))]) 
            visualizer.display_current_results(visuals, epoch, total_steps) 

        # save the latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest') 
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time=time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest') 
        model.module.save(epoch) 
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    # instead of only training the local enhancer, train the entire network after certain iterations 
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global): 
        model.module.update_fixed_params() 
    
    # linearly decay learning rate after certain iters
    if epoch > opt.niter:
        print('update learning rate')
        model.module.update_learning_rate() 
