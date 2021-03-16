import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from ContentModel import ContentNet
from Discriminator import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--bottleneck', type=int, default=4000, help='of dim for bottleneck of encoder')
parser.add_argument('--channal', type=int, default=3)
parser.add_argument('--deFilter', type=int, default=64)
parser.add_argument('--filter1', type=int, default=64, help='of encoder filters in first conv layer')

parser.add_argument('--DFilter', type=int, default=64)

parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--dataroot', default='G:\Pairs_streetVeiw', help='path to dataset')
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')

parser.add_argument('--contentNet', default='model/contentNet_cifar10.pth', help="path to netG (to continue training)")
parser.add_argument('--discriNet', default='model/dicriNet_cifar10.pth', help="path to netD (to continue training)")

parser.add_argument('--overlapPred', type=int, default=4, help='overlapping edges')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')

# ??
parser.add_argument('--wtl2', type=float, default=0.998, help='0 means do not use else use with this weight')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

# random seed set
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

# import dataset
# transform = transforms.Compose([transforms.Scale(opt.imageSize),
#                                 transforms.CenterCrop(opt.imageSize),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# dataset = dset.CIFAR10(root=opt.dataroot, download=True,
#                        transform=transforms.Compose([
#                            transforms.Scale(opt.imageSize),
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                        ])
#                        )
transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, )


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch = 0

contentNet = ContentNet(opt)
contentNet.apply(weights_init)
if opt.contentNet != '' and os.path.exists(opt.contentNet):
    contentNet.load_state_dict(torch.load(opt.contentNet, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.contentNet)['epoch']
print(contentNet)

discriNet = Discriminator(opt)
discriNet.apply(weights_init)
if opt.discriNet != '' and os.path.exists(opt.discriNet):
    discriNet.load_state_dict(torch.load(opt.discriNet, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.discriNet)['epoch']
print(discriNet)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)

real_label = 1
fake_label = 0

real_center = torch.FloatTensor([opt.batchSize, 3, opt.imageSize / 2, opt.imageSize / 2])

# move to cuda
contentNet.cuda()
discriNet.cuda()
criterion.cuda()
criterionMSE.cuda()
input_real, input_cropped, label = input_real.cuda(), input_cropped.cuda(), label.cuda()
real_center = real_center.cuda()

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

real_center = Variable(real_center)

# setup optimizer
optimizerD = optim.Adam(discriNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(contentNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

data = iter(dataloader).next()
real_raw, _ = data
real_center_raw = real_raw[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2),
                  int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2)]
batch_size = real_raw.size(0)

input_real.resize_(real_raw.size()).copy_(real_raw)
input_cropped.resize_(real_raw.size()).copy_(real_raw)
real_center.resize_(real_center_raw.size()).copy_(real_center_raw)

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

# 优化判别器
discriNet.zero_grad()
label.resize_(batch_size).fill_(real_label)

output = discriNet(real_center)
err_real_D = criterion(output, label)
err_real_D.backward()
D_x = output.data.mean()

# train with fake
fake = contentNet(input_cropped)
label.data.fill_(fake_label)
output = discriNet(fake.detach())
err_fake_D = criterion(output, label)
err_fake_D.backward()
D_G_1 = output.data.mean()
errD = err_real_D + err_fake_D
optimizerD.step()

# 优化生成器  maximize log(D(G(z)))
contentNet.zero_grad()
# 目标是将其完全变为真
label.data.fill_(real_label)
output = discriNet(fake)
err_G_D = criterion(output, label)

errG_l2 = criterionMSE(fake, real_center)
# ？？？？
wtl2Matrix = real_center.clone()
wtl2Matrix.data.fill_(float(opt.wtl2) * 10)
wtl2Matrix.data[:, :, int(opt.overlapPred):int(opt.imageSize / 2 - opt.overlapPred),
int(opt.overlapPred):int(opt.imageSize / 2 - opt.overlapPred)] = float(opt.wtl2)

errG_l2 = (fake - real_center).pow(2)
errG_l2 = errG_l2 * wtl2Matrix
errG_l2 = errG_l2.mean()

errG = (1 - float(opt.wtl2)) * err_G_D + float(opt.wtl2) * errG_l2

errG.backward()

D_G_z2 = output.data.mean()
optimizerG.step()

print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
      % (epoch, opt.niter, i, len(dataloader),
         errD.item(), err_G_D.item(), errG_l2.item(), D_x, D_G_1,))

if i % 15 == 0:
    vutils.save_image(real_raw,
                      'result/train/real/real_samples_epoch_%03d_%03d.png' % (epoch, i))
    vutils.save_image(input_cropped.data,
                      'result/train/cropped/cropped_samples_epoch_%03d_%03d.png' % (epoch, i))
    recon_image = input_cropped.clone()
    recon_image.data[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2),
    int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2)] = fake.data
    vutils.save_image(recon_image.data,
                      'result/train/recon/recon_center_samples_epoch_%03d_%03d.png' % (epoch, i))

if i % 10 == 0:
    # do checkpointing
    torch.save({'epoch': epoch,
                'state_dict': contentNet.state_dict()},
               'model/contentNet_cifar10.pth')
    torch.save({'epoch': epoch,
                'state_dict': discriNet.state_dict()},
               'model/discriNet_cifar10.pth')
    print("第{}已保存".format(i))
