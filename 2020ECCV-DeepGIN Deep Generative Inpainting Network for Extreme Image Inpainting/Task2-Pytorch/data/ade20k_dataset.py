####################################################################################
# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license
# (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
####################################################################################
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_with_conditions, make_dataset_with_condition_list
from data.masks import Masks
from PIL import Image 
import random
import numpy as np
from natsort import natsorted
import torch 

class ADE20kDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        ################################################
        # input A       : masked images (rgb)
        # input B       : real images (rgb)
        # input mask    : masks (gray)
        # input B_seg   : real seg images (rgb)
        ################################################

        if opt.phase == 'train':
            # the 10,330 images in ade20k train set
            self.dir_A = os.path.join(opt.dataroot, opt.phase)
            self.B_seg_paths, _ = make_dataset_with_conditions(self.dir_A, '_seg.png')
            self.A_paths, _ = make_dataset_with_conditions(self.dir_A, '.jpg')
            
            self.B_seg_paths = natsorted(self.B_seg_paths)
            self.A_paths = natsorted(self.A_paths)
            self.B_paths = self.A_paths

            self.dir_mask = []
            self.mask_paths = []

            self.mask = Masks()

        elif opt.phase == 'test':
            # aim 2020 eccv validation set and test set.
            # no ground truth
            self.dir_A = os.path.join(opt.dataroot, opt.phase)
            self.A_paths, _ = make_dataset_with_conditions(self.dir_A, '_with_holes')
            self.A_paths = natsorted(self.A_paths)
            
            self.dir_B = []
            self.B_paths = []
            self.seg_paths, _ = make_dataset_with_conditions(self.dir_A, '_seg')
            self.seg_paths = natsorted(self.seg_paths) 

            self.B_seg_paths = [] # depends on track 1 or 2

            self.dir_mask = os.path.join(opt.dataroot, opt.phase)
            self.mask_paths, _ = make_dataset_with_conditions(self.dir_mask, '_mask')
            self.mask_paths = natsorted(self.mask_paths)

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ################################################
        # input A       : masked images (rgb)
        # input B       : real images (rgb)
        # input mask    : masks (gray)
        # input B_seg   : real seg images (rgb)
        ################################################

        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')

        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params, normalize=False)
        A_tensor = transform_A(A)

        B_tensor = mask_tensor = 0
        B_seg_tensor = 0

        if self.opt.phase == 'train':
            # the 10,330 images in ade20k train set
            B_seg_path = self.B_seg_paths[index]
            B_seg = Image.open(B_seg_path).convert('RGB')

            new_A, new_B_seg = self.resize_or_crop(A, B_seg)
            ## data augmentation rotate
            new_A, new_B_seg = self.rotate(new_A, new_B_seg) 
            new_A, new_B_seg = self.ensemble(new_A, new_B_seg) 

            width, height = new_A.size
            f_A = np.array(new_A, np.float32)
            f_A1 = np.array(new_A, np.float32)
            f_A2 = np.array(new_A, np.float32)
            f_A3 = np.array(new_A, np.float32)

            f_mask = self.mask.get_random_mask(height, width)

            f_A[:, :, 0] = f_A[:, :, 0] * (1.0 - f_mask) + 255.0 * f_mask
            f_A[:, :, 1] = f_A[:, :, 1] * (1.0 - f_mask) + 255.0 * f_mask
            f_A[:, :, 2] = f_A[:, :, 2] * (1.0 - f_mask) + 255.0 * f_mask
            f_masked_A = f_A

            f_mask1 = self.mask.get_box_mask(height, width)
            f_mask2 = self.mask.get_ca_mask(height, width)
            f_mask3 = self.mask.get_ff_mask(height, width)

            f_A1[:, :, 0] = f_A1[:, :, 0] * (1.0 - f_mask1) + 255.0 * f_mask1
            f_A1[:, :, 1] = f_A1[:, :, 1] * (1.0 - f_mask1) + 255.0 * f_mask1
            f_A1[:, :, 2] = f_A1[:, :, 2] * (1.0 - f_mask1) + 255.0 * f_mask1
            f_masked_A1 = f_A1

            f_A2[:, :, 0] = f_A2[:, :, 0] * (1.0 - f_mask2) + 255.0 * f_mask2
            f_A2[:, :, 1] = f_A2[:, :, 1] * (1.0 - f_mask2) + 255.0 * f_mask2
            f_A2[:, :, 2] = f_A2[:, :, 2] * (1.0 - f_mask2) + 255.0 * f_mask2
            f_masked_A2 = f_A2

            f_A3[:, :, 0] = f_A3[:, :, 0] * (1.0 - f_mask3) + 255.0 * f_mask3
            f_A3[:, :, 1] = f_A3[:, :, 1] * (1.0 - f_mask3) + 255.0 * f_mask3
            f_A3[:, :, 2] = f_A3[:, :, 2] * (1.0 - f_mask3) + 255.0 * f_mask3
            f_masked_A3 = f_A3

            # masked images
            masked_A = Image.fromarray((f_masked_A).astype(np.uint8)).convert('RGB')
            mask_img = Image.fromarray((f_mask * 255.0).astype(np.uint8)).convert('L')

            masked_A1 = Image.fromarray((f_masked_A1).astype(np.uint8)).convert('RGB')
            mask_img1 = Image.fromarray((f_mask1 * 255.0).astype(np.uint8)).convert('L')

            masked_A2 = Image.fromarray((f_masked_A2).astype(np.uint8)).convert('RGB')
            mask_img2 = Image.fromarray((f_mask2 * 255.0).astype(np.uint8)).convert('L')

            masked_A3 = Image.fromarray((f_masked_A3).astype(np.uint8)).convert('RGB')
            mask_img3 = Image.fromarray((f_mask3 * 255.0).astype(np.uint8)).convert('L')

            A_tensor = transform_A(masked_A)

            ##
            A_tensor1 = transform_A(masked_A1)
            A_tensor2 = transform_A(masked_A2)
            A_tensor3 = transform_A(masked_A3)

            # real images
            B_path = self.B_paths[index]
            B = new_A
            transform_B = get_transform(self.opt, params, normalize=False)
            B_tensor = transform_B(B)

            B_tensor1 = transform_B(B)
            B_tensor2 = transform_B(B)
            B_tensor3 = transform_B(B)

            transform_B_seg = get_transform(self.opt, params, normalize=False)
            B_seg_tensor = transform_B_seg(new_B_seg)

            B_seg_tensor1 = transform_B_seg(new_B_seg)
            B_seg_tensor2 = transform_B_seg(new_B_seg)
            B_seg_tensor3 = transform_B_seg(new_B_seg)

            # masks
            mask_path = []
            transform_mask = get_transform(self.opt, params, normalize=False)
            mask_tensor = transform_mask(mask_img)

            mask_tensor1 = transform_mask(mask_img1)
            mask_tensor2 = transform_mask(mask_img2)
            mask_tensor3 = transform_mask(mask_img3)

            A_tensor = torch.cat((A_tensor1, A_tensor2, A_tensor3), dim=0)
            B_tensor = torch.cat((B_tensor1, B_tensor2, B_tensor3), dim=0)
            mask_tensor = torch.cat((mask_tensor1, mask_tensor2, mask_tensor3), dim=0) 
            B_seg_tensor = torch.cat((B_seg_tensor1, B_seg_tensor2, B_seg_tensor3), dim=0) 
        
        elif self.opt.phase == 'test':
            # aim 2020 eccv validation set and test set.
            # no ground truth
            B_path = []
            B_seg_path = []

            seg_path = self.seg_paths[index] 
            seg = Image.open(seg_path).convert('RGB') 

            # masks
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path).convert('L')
            transform_mask = get_transform(self.opt, params, normalize=False)
            mask_tensor = transform_mask(mask) 
 
        if self.opt.phase == 'test':
            input_dict = {
                'masked_image': A_tensor,
                'mask': mask_tensor,
                'path_mskimg': A_path, 
                'path_msk': mask_path, 
                'path_seg': seg_path}
            return input_dict

        input_dict = {'masked_image': A_tensor,
                      'mask': mask_tensor,
                      'real_image': B_tensor,
                      'real_seg': B_seg_tensor}
        return input_dict 

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'ADE20kDataset'

    def resize_or_crop(self, img, seg_img, b_map, method=Image.BICUBIC):
        w, h = img.size
        new_w = w
        new_h = h

        if w > self.opt.loadSize and h > self.opt.loadSize:
            return img.resize((self.opt.fineSize * 2, self.opt.fineSize * 2), method), seg_img.resize((self.opt.fineSize * 2, self.opt.fineSize * 2), method)
        else:
            return img.resize((self.opt.fineSize * 2, self.opt.fineSize * 2), method), seg_img.resize((self.opt.fineSize * 2, self.opt.fineSize * 2), method)

    def rotate(self, img, seg):
        bFlag = random.randint(0, 3)
        if bFlag == 0:
            return img.rotate(0), seg.rotate(0)
        elif bFlag == 1:
            return img.rotate(90), seg.rotate(90) 
        elif bFlag == 2:
            return img.rotate(180), seg.rotate(180) 
        elif bFlag == 3:
            return img.rotate(270), seg.rotate(270) 

    def ensemble(self, img, seg):
        bFlag = random.randint(0, 3)
        width, height = img.size 
        new_w = width // 2
        new_h = height // 2
        new_img = img.resize((new_w, new_h), Image.BICUBIC) 
        np_img = np.array(img, np.float32)
        np_new_img = np.array(new_img, np.float32) 

        new_seg = seg.resize((new_w, new_h), Image.BICUBIC) 
        np_seg = np.array(seg, np.float32)
        np_new_seg = np.array(new_seg, np.float32) 
        
        if bFlag == 0:
            for i in range(new_w):
                for j in range(new_h):
                    np_new_img[i, j, 0] = np_img[2*i, 2*j, 0] 
                    np_new_img[i, j, 1] = np_img[2*i, 2*j, 1] 
                    np_new_img[i, j, 2] = np_img[2*i, 2*j, 2] 

                    np_new_seg[i, j, 0] = np_seg[2*i, 2*j, 0] 
                    np_new_seg[i, j, 1] = np_seg[2*i, 2*j, 1] 
                    np_new_seg[i, j, 2] = np_seg[2*i, 2*j, 2] 
        elif bFlag == 1:
            for i in range(new_w):
                for j in range(new_h):
                    np_new_img[i, j, 0] = np_img[1 + 2*i, 1 + 2*j, 0] 
                    np_new_img[i, j, 1] = np_img[1 + 2*i, 1 + 2*j, 1] 
                    np_new_img[i, j, 2] = np_img[1 + 2*i, 1 + 2*j, 2] 

                    np_new_seg[i, j, 0] = np_seg[1 + 2*i, 1 + 2*j, 0] 
                    np_new_seg[i, j, 1] = np_seg[1 + 2*i, 1 + 2*j, 1] 
                    np_new_seg[i, j, 2] = np_seg[1 + 2*i, 1 + 2*j, 2] 
        elif bFlag == 2:
            for i in range(new_w):
                for j in range(new_h):
                    np_new_img[i, j, 0] = np_img[1 + 2*i, 2*j, 0] 
                    np_new_img[i, j, 1] = np_img[1 + 2*i, 2*j, 1] 
                    np_new_img[i, j, 2] = np_img[1 + 2*i, 2*j, 2] 

                    np_new_seg[i, j, 0] = np_seg[1 + 2*i, 2*j, 0] 
                    np_new_seg[i, j, 1] = np_seg[1 + 2*i, 2*j, 1] 
                    np_new_seg[i, j, 2] = np_seg[1 + 2*i, 2*j, 2] 
        else:
            for i in range(new_w):
                for j in range(new_h):
                    np_new_img[i, j, 0] = np_img[2*i, 1 + 2*j, 0] 
                    np_new_img[i, j, 1] = np_img[2*i, 1 + 2*j, 1] 
                    np_new_img[i, j, 2] = np_img[2*i, 1 + 2*j, 2] 
                    
                    np_new_seg[i, j, 0] = np_seg[2*i, 1 + 2*j, 0] 
                    np_new_seg[i, j, 1] = np_seg[2*i, 1 + 2*j, 1] 
                    np_new_seg[i, j, 2] = np_seg[2*i, 1 + 2*j, 2]

        new_A = Image.fromarray((np_new_img).astype(np.uint8)).convert('RGB') 
        new_seg = Image.fromarray((np_new_seg).astype(np.uint8)).convert('RGB') 
        return new_A, new_seg 
