####################################################################################
# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license
# (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
####################################################################################
import argparse 
import os 
import torch 
from util import util 


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Inpainting') 
        self.initialized = False 

    def initialize(self):
        # experiment specifics 
        self.parser.add_argument('--name', type=str, default='experiment', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2  0,2. use -1 for cpu')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='Ours', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance or batch normalization') 


        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=768, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='image size to the model') 
        self.parser.add_argument('--input_nc', type=int, default=7, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels') 

        # for setting inputs 
        self.parser.add_argument('--dataroot', type=str, default='./datasets/ade20k/')
        self.parser.add_argument('--resize_or_crop', type=str, default='standard', help='scaling and/or cropping of images at load time')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, no shuffle') 
        self.parser.add_argument('--no_flip', action='store_true', help='if true, no flip')
        self.parser.add_argument('--nThreads', type=int, default=6, help='# of threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='max # of images allowed per dataset') 

        # for displays 
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')

        # for generator 
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=5, help='# of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=6, help='# of resnet blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='# of resnet blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='# of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='# of epochs that we only train the outmost local enhancer') 

        self.initialized = True 

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = [] 
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids 
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        
        args = vars(self.opt) 

        # print options 
        print('-------------------- Options --------------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('---------------------- End ----------------------')

        # save the options to disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)

        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt') 
            with open(file_name, 'wt') as opt_file:
                opt_file.write('-------------------- Options --------------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('---------------------- End ----------------------\n')

        return self.opt 

