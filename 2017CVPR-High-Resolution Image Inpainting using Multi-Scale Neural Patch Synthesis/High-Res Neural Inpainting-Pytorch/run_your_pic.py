import argparse
import os
import random

import cv2
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.autograd import Variable

from ContentModel import ContentNet
from process_texture import texture

parser = argparse.ArgumentParser()
parser.add_argument('--bottleneck', type=int, default=4000, help='of dim for bottleneck of encoder')
parser.add_argument('--channal', type=int, default=3)
parser.add_argument('--deFilter', type=int, default=64)
parser.add_argument('--filter1', type=int, default=64, help='of encoder filters in first conv layer')

parser.add_argument('--DFilter', type=int, default=64)
parser.add_argument('--imageSize_raw', type=int, default=512, help='the height / width of the input image to network')

parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--dataroot', default='G:\Pairs_streetVeiw', help='path to dataset')
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')

parser.add_argument('--contentNet', default='model/contentNet_cifar10.pth',
                    help="path to netG (to continue training)")
parser.add_argument('--discriNet', default='model/dicriNet_cifar10.pth', help="path to netD (to continue training)")
parser.add_argument('--content_path',type=str, default='For_test/001101_2.jpg')

parser.add_argument('--overlapPred', type=int, default=4, help='overlapping edges')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')

# ??
parser.add_argument('--wtl2', type=float, default=0.998, help='0 means do not use else use with this weight')

opt = parser.parse_args()
print(opt)

pic=opt.content_path
dir='pic_result'

if not os.path.exists(dir):
    os.mkdir(dir)

#cut edge
transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform1 = transforms.Compose([transforms.Scale(opt.imageSize_raw),
                                 transforms.CenterCrop(opt.imageSize_raw),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
#keep edge
# transform = transforms.Compose([transforms.Scale((opt.imageSize,opt.imageSize)),
#
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# transform1 = transforms.Compose([transforms.Scale((opt.imageSize_raw,opt.imageSize_raw)),
#
#                                  transforms.ToTensor(),
#                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

denorm_transform = transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2))
denorm_transform1 = transforms.Normalize(mean=(-2.12, -2.04, -1.80), std=(4.37, 4.46, 4.44))
transform3 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# content_image_ori = cv2.imread(opt.content_path1)
content_image_ori = cv2.imread(pic)
content_image_ori = cv2.cvtColor(content_image_ori, cv2.COLOR_BGR2RGB)
content_image_ori_PIL = Image.fromarray(content_image_ori)
content_image = transform(content_image_ori_PIL).unsqueeze(0)
content_images = content_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
content_512 = transform1(content_image_ori_PIL).unsqueeze(0).to(device)
vutils.save_image(denorm_transform1(content_512[0]), dir + '/real.jpg')

content_512[:, :, int(opt.imageSize_raw / 4):int(opt.imageSize_raw / 4 + opt.imageSize_raw / 2),
int(opt.imageSize_raw / 4):int(opt.imageSize_raw / 4 + opt.imageSize_raw / 2)] = 0.0
content_512[:, :, int(opt.imageSize_raw / 4):int(opt.imageSize_raw / 4 + opt.imageSize_raw / 2),
int(opt.imageSize_raw / 4):int(opt.imageSize_raw / 4 + opt.imageSize_raw / 2)] = torch.mean(content_512)

vutils.save_image(denorm_transform1(content_512[0]), dir + '/cropped.jpg')

photos = [
    'Pairs_streetVeiw\part1/000001_1.jpg',
    'Pairs_streetVeiw\part1/000001_2.jpg',
    'Pairs_streetVeiw\part1/000001_3.jpg',
    'Pairs_streetVeiw\part1/000001_4.jpg',
    'Pairs_streetVeiw\part1/000002_0.jpg',
    'Pairs_streetVeiw\part1/000002_1.jpg',
    'Pairs_streetVeiw\part1/000002_2.jpg',
    'Pairs_streetVeiw\part1/000002_3.jpg',
    'Pairs_streetVeiw\part1/000002_4.jpg',
    'Pairs_streetVeiw\part1/000003_0.jpg',
    'Pairs_streetVeiw\part1/000003_1.jpg',
    'Pairs_streetVeiw\part1/000003_2.jpg',
    'Pairs_streetVeiw\part1/000003_3.jpg',
    'Pairs_streetVeiw\part1/000003_4.jpg',
    'Pairs_streetVeiw\part1/000004_0.jpg']

for path in photos:
    content_image_ori1 = cv2.imread(path)
    content_image_ori1 = cv2.cvtColor(content_image_ori1, cv2.COLOR_BGR2RGB)
    content_image_ori_PIL1 = Image.fromarray(content_image_ori1)
    content_image1 = transform(content_image_ori_PIL1).unsqueeze(0)
    content_images = torch.cat((content_images, content_image1), 0)

# content_image_ori2 = cv2.imread(opt.content_path2)
# content_image_ori2 = cv2.cvtColor(content_image_ori2, cv2.COLOR_BGR2RGB)
# content_image_ori_PIL2=Image.fromarray(content_image_ori2)
# content_image2 = transform(content_image_ori_PIL2).unsqueeze(0)
#
# content_image_ori3 = cv2.imread(opt.content_path2)
# content_image_ori3 = cv2.cvtColor(content_image_ori3, cv2.COLOR_BGR2RGB)
# content_image_ori_PIL3=Image.fromarray(content_image_ori3)
# content_image3 = transform(content_image_ori_PIL3).unsqueeze(0)

# content_images=torch.cat((content_images,content_images),0)
# content_images=torch.cat((content_images,content_images),0)
# content_images=torch.cat((content_images,content_images),0)
# content_images=torch.cat((content_images,content_images),0)
# content_images=torch.cat((content_images,content_images),0)
# content_images=torch.cat((content_images,content_images),0)

input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = Variable(input_cropped)

input_cropped.resize_(content_images.size()).copy_(content_images)

input_cropped.data[:, 0,
int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
int(opt.imageSize / 4 + opt.overlapPred):int(
    opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 117.0 / 255.0 - 1.0
input_cropped.data[:, 1,
int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
int(opt.imageSize / 4 + opt.overlapPred):int(
    opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 104.0 / 255.0 - 1.0
input_cropped.data[:, 2,
int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
int(opt.imageSize / 4 + opt.overlapPred):int(
    opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 123.0 / 255.0 - 1.0

contentNet = ContentNet(opt)
if opt.contentNet != '' and os.path.exists(opt.contentNet):
    contentNet.load_state_dict(
        torch.load(opt.contentNet, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.contentNet)['epoch']
print(contentNet)

synthesis = contentNet(input_cropped)

recon_image = input_cropped.clone()
recon_image.data[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2),
int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2)] = synthesis.data

vutils.save_image(denorm_transform(recon_image.data[0]), dir + '/input.jpg')
vutils.save_image(denorm_transform(synthesis.data[0]), dir + '/output.jpg')

content_result = denorm_transform(synthesis.data[0])
content_result = transform3(content_result)

content_result = content_result.unsqueeze(0)
content_result = F.interpolate(content_result, [int(opt.imageSize_raw / 2), int(opt.imageSize_raw / 2)],
                               mode="bilinear")
content_result.cuda()

result = texture(content_512, content_result, dir)
# result=F.interpolate(result,[256,256],mode="/bilinear")
content_512 = denorm_transform1(content_512.data[0])
for i in range(3):
    content_512.data[i, int(opt.imageSize_raw / 4):int(opt.imageSize_raw / 4 + opt.imageSize_raw / 2), \
    int(opt.imageSize_raw / 4):int(opt.imageSize_raw / 4 + opt.imageSize_raw / 2)] = result.data[0][i]

vutils.save_image(content_512, dir + '/result.jpg')
