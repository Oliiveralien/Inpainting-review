"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from MPRNet import MPRNet
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Denoising using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/DND/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/DND/test/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_denoising.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

result_dir = os.path.join(args.result_dir, 'mat')
utils.mkdir(result_dir)

if args.save_images:
    result_dir_img = os.path.join(args.result_dir, 'png')
    utils.mkdir(result_dir_img)

model_restoration = MPRNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

israw = False
eval_version="1.0"

# Load info
infos = h5py.File(os.path.join(args.input_dir, 'info.mat'), 'r')
info = infos['info']
bb = info['boundingboxes']

# Process data
with torch.no_grad():
    for i in tqdm(range(50)):
        Idenoised = np.zeros((20,), dtype=np.object)
        filename = '%04d.mat'%(i+1)
        filepath = os.path.join(args.input_dir, 'images_srgb', filename)
        img = h5py.File(filepath, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)

        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T

        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            noisy_patch = torch.from_numpy(Inoisy[idx[0]:idx[1],idx[2]:idx[3],:]).unsqueeze(0).permute(0,3,1,2).cuda()
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch[0],0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            Idenoised[k] = restored_patch

            if args.save_images:
                save_file = os.path.join(result_dir_img, '%04d_%02d.png'%(i+1,k+1))
                denoised_img = img_as_ubyte(restored_patch)
                utils.save_img(save_file, denoised_img)

        # save denoised data
        sio.savemat(os.path.join(result_dir, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )
