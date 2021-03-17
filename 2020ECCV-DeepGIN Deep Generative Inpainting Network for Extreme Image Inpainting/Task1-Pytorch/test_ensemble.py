from util.visualizer import Visualizer
import util.util as util
from data.data_loader import CreateDataLoader
from options.test_options import TestOptions
import torchvision.transforms as transforms
import time
import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from collections import OrderedDict 

from models.models import create_model 
from util import html 

import copy 

opt = TestOptions().parse(save=False) 
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True 
opt.no_flip = True 

data_loader = CreateDataLoader(opt) 
dataset = data_loader.load_data() 
dataset_size = len(data_loader) 

def __make_power_2(img, base=256, method=Image.BICUBIC):
    ow, oh = img.size

    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)

    if h == 0:
        h = base 
    if w == 0: 
        w = base 

    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

model, num_params_G, num_params_D = create_model(opt) 
model.eval()

rlt_dir = os.path.join(opt.results_dir, 'test') 
util.mkdirs([rlt_dir])

transform_list = []
transform_list += [transforms.ToTensor()]
mskimg_transform = transforms.Compose(transform_list)
transform_list = []
transform_list += [transforms.ToTensor()]
msk_transform = transforms.Compose(transform_list)

start_time = time.time() 
for i, data in enumerate(dataset): 
    with torch.no_grad(): 
        msk_img_path = data['path_mskimg'][0] 
        filename = os.path.basename(msk_img_path) 
        msk_path = data['path_msk'][0] 

        oimg = Image.open(msk_img_path).convert('RGB') 
        omsk = Image.open(msk_path).convert('L') 
        ow, oh = oimg.size 
        
        ###
        resized_img = __make_power_2(oimg) 
        resized_msk = __make_power_2(omsk, method=Image.BILINEAR) 
        rw, rh = resized_img.size 

        hori_ver = rw // 256 
        vert_ver = rh // 256 

        tmp_img = oimg.resize((256, 256), Image.BICUBIC) 
        tmp_msk = omsk.resize((256, 256), Image.BICUBIC) 

        np_tmp_img = np.array(tmp_img, np.uint8) 
        np_tmp_msk = np.array(tmp_msk, np.uint8) 

        np_resized_img = np.array(resized_img, np.uint8)
        np_resized_msk = np.array(resized_msk, np.uint8)
        np_resized_msk = np_resized_msk > 0
        np_resized_img[:,:,0] = np_resized_img[:,:,0] * (1 - np_resized_msk) + 255 * np_resized_msk
        np_resized_img[:,:,1] = np_resized_img[:,:,1] * (1 - np_resized_msk) + 255 * np_resized_msk
        np_resized_img[:,:,2] = np_resized_img[:,:,2] * (1 - np_resized_msk) + 255 * np_resized_msk
        np_resized_msk = np_resized_msk * 255

        img_arr = []
        msk_arr = [] 

        ###
        for hv in range(hori_ver):
            for vv in range(vert_ver):
                for i in range(256):
                    for j in range(256):
                        np_tmp_img[i, j, 0] = np_resized_img[vv + vert_ver*j, hv + hori_ver*i, 0] 
                        np_tmp_img[i, j, 1] = np_resized_img[vv + vert_ver*j, hv + hori_ver*i, 1] 
                        np_tmp_img[i, j, 2] = np_resized_img[vv + vert_ver*j, hv + hori_ver*i, 2] 
                        np_tmp_msk[i, j] = np_resized_msk[vv + vert_ver*j, hv + hori_ver*i] 
                img_arr.append(np.copy(np_tmp_img)) 
                msk_arr.append(np.copy(np_tmp_msk)) 
        
        ### 
        compltd_arr = []
        for i in range(len(img_arr)):
            img = Image.fromarray(img_arr[i])
            msk = Image.fromarray(msk_arr[i])

            img_90 = img.rotate(90) 
            msk_90 = msk.rotate(90) 
            img_180 = img.rotate(180) 
            msk_180 = msk.rotate(180) 
            img_270 = img.rotate(270) 
            msk_270 = msk.rotate(270) 
            img_flp = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            msk_flp = msk.transpose(method=Image.FLIP_LEFT_RIGHT)

            compltd_img, reconst_img, lr_x = model(mskimg_transform(img).unsqueeze(0), msk_transform(msk).unsqueeze(0)) 
            compltd_img_90, reconst_img_90, lr_x_90 = model(mskimg_transform(img_90).unsqueeze(0), msk_transform(msk_90).unsqueeze(0)) 
            compltd_img_180, reconst_img_180, lr_x_180 = model(mskimg_transform(img_180).unsqueeze(0), msk_transform(msk_180).unsqueeze(0)) 
            compltd_img_270, reconst_img_270, lr_x_270 = model(mskimg_transform(img_270).unsqueeze(0), msk_transform(msk_270).unsqueeze(0)) 
            compltd_img_flp, reconst_img_flp, lr_x_flp = model(mskimg_transform(img_flp).unsqueeze(0), msk_transform(msk_flp).unsqueeze(0)) 
            np_compltd_img = util.tensor2im(reconst_img.data[0], normalize=False) 
            np_compltd_img_90 = util.tensor2im(reconst_img_90.data[0], normalize=False) 
            np_compltd_img_180 = util.tensor2im(reconst_img_180.data[0], normalize=False) 
            np_compltd_img_270 = util.tensor2im(reconst_img_270.data[0], normalize=False) 
            np_compltd_img_flp = util.tensor2im(reconst_img_flp.data[0], normalize=False) 
            
            new_img_90 = Image.fromarray(np_compltd_img_90) 
            new_img_90 = new_img_90.rotate(270) 
            np_new_img_90 = np.array(new_img_90, np.float) 
            
            new_img_180 = Image.fromarray(np_compltd_img_180) 
            new_img_180 = new_img_180.rotate(180) 
            np_new_img_180 = np.array(new_img_180, np.float) 
            
            new_img_270 = Image.fromarray(np_compltd_img_270) 
            new_img_270 = new_img_270.rotate(90) 
            np_new_img_270 = np.array(new_img_270, np.float) 
            
            new_img_flp = Image.fromarray(np_compltd_img_flp) 
            new_img_flp = new_img_flp.transpose(method=Image.FLIP_LEFT_RIGHT) 
            np_new_img_flp = np.array(new_img_flp, np.float) 

            np_compltd_img = (np_compltd_img + np_new_img_90 + np_new_img_180 + np_new_img_270 + np_new_img_flp) / 5.0 
            np_compltd_img = np.array(np.round(np_compltd_img), np.uint8) 
            final_img = Image.fromarray(np_compltd_img, mode="RGB") 
            np_compltd_img = np.array(final_img, np.uint8) 

            compltd_arr.append(np.copy(np_compltd_img)) 

        ###
        ver_idx = 0
        for hv in range(hori_ver):
            for vv in range(vert_ver):
                #np_tmp_img = compltd_arr[ver_idx] 
                for i in range(256):
                    for j in range(256):
                        np_resized_img[vv + vert_ver*j, hv + hori_ver*i, 0] = compltd_arr[ver_idx][i, j, 0] 
                        np_resized_img[vv + vert_ver*j, hv + hori_ver*i, 1] = compltd_arr[ver_idx][i, j, 1] 
                        np_resized_img[vv + vert_ver*j, hv + hori_ver*i, 2] = compltd_arr[ver_idx][i, j, 2] 
                ver_idx += 1 

        ### 
        new_compltd_img = Image.fromarray(np_resized_img)
        new_compltd_img = new_compltd_img.resize((ow, oh), Image.BICUBIC) 
        new_compltd_img = new_compltd_img.resize((int(ow*0.5), int(oh*0.5)), Image.BICUBIC) 
        new_compltd_img = new_compltd_img.resize((ow, oh), Image.BICUBIC) 
        np_new_compltd_img = np.array(new_compltd_img) 
        np_oimg = np.array(oimg) 
        np_omsk = np.array(omsk) 

        np_new_compltd_img[:, :, 0] = np_new_compltd_img[:, :, 0] * (np_omsk / 255.0) + ((255.0 - np_omsk) / 255.0) * np_oimg[:, :, 0] 
        np_new_compltd_img[:, :, 1] = np_new_compltd_img[:, :, 1] * (np_omsk / 255.0) + ((255.0 - np_omsk) / 255.0) * np_oimg[:, :, 1] 
        np_new_compltd_img[:, :, 2] = np_new_compltd_img[:, :, 2] * (np_omsk / 255.0) + ((255.0 - np_omsk) / 255.0) * np_oimg[:, :, 2] 

        newfilename = filename.replace("_with_holes", "")
        compltd_path = os.path.join(rlt_dir, newfilename) 
        util.save_image(np_new_compltd_img, compltd_path) 

        print(compltd_path)

end_time = time.time() - start_time 
print('Avg Time Taken: %.3f sec' % (end_time / dataset_size))

print('done')
