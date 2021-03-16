import argparse
import os

import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

from CNNMRF_origin import CNNMRF


def get_synthesis_image(synthesis, denorm, device):
    cpu_device = torch.device('cpu')
    image = synthesis.clone().squeeze().to(cpu_device)

    image = denorm(image)
    return image.to(device).clamp_(0, 1)


def unsample_synthesis(height, width, synthesis, device):
    synthesis = F.interpolate(synthesis, size=[height, width], mode='bilinear')
    synthesis = synthesis.clone().detach().requires_grad_(True).to(device)
    return synthesis


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    "-------------------transform and denorm transform-----------------"
    # VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406]
    # and std=[0.229, 0.224, 0.225].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    denorm_transform = transforms.Normalize(mean=(-2.12, -2.04, -1.80), std=(4.37, 4.46, 4.44))

    "--------------read image------------------"
    if not os.path.exists(config.content_path):
        raise ValueError('file %s does not exist.' % config.content_path)
    if not os.path.exists(config.style_path):
        raise ValueError('file %s does not exist.' % config.style_path)
    content_image = cv2.imread(config.content_path)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    content_image = transform(content_image).unsqueeze(0).to(device)

    style_image = cv2.imread(config.style_path)
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    style_image = transform(style_image).unsqueeze(0).to(device)

    content_image = F.interpolate(content_image, [512, 512], mode="bilinear")
    style_image = F.interpolate(style_image, [512, 512], mode="bilinear")

    "resize image in several level for training"
    pyramid_content_image = []
    pyramid_style_image = []
    for i in range(config.num_res):
        content = F.interpolate(content_image, scale_factor=1 / pow(2, config.num_res - 1 - i), mode='bilinear')

        style = F.interpolate(style_image, scale_factor=1 / pow(2, config.num_res - 1 - i), mode='bilinear')

        pyramid_content_image.append(content)
        pyramid_style_image.append(style)
    "-----------------start training-------"
    global iter
    iter = 0
    synthesis = None
    # create cnnmrf model
    cnnmrf = CNNMRF(style_image=pyramid_style_image[0], content_image=pyramid_content_image[0], device=device,
                    content_weight=config.content_weight, style_weight=config.style_weight, tv_weight=config.tv_weight,
                    gpu_chunck_size=config.gpu_chunck_size, mrf_synthesis_stride=config.mrf_synthesis_stride,
                    mrf_style_stride=config.mrf_style_stride).to(device)

    # Sets the module in training mode.
    cnnmrf.train()
    for i in range(0, config.num_res):
        # synthesis = torch.rand_like(content_image, requires_grad=True)
        if i == 0:
            # in lowest level init the synthesis from content resized image
            synthesis = pyramid_content_image[0].clone().requires_grad_(True).to(device)
        else:
            # in high level init the synthesis from unsampling the upper level synthesis
            synthesis = unsample_synthesis(pyramid_content_image[i].shape[2], pyramid_content_image[i].shape[3],
                                           synthesis, device)
            cnnmrf.update_style_and_content_image(style_image=pyramid_style_image[i],
                                                  content_image=pyramid_content_image[i])
        # max_iter (int): maximal number of iterations per optimization step
        optimizer = optim.LBFGS([synthesis], lr=1, max_iter=config.max_iter)
        "--------------------"

        def closure():
            global iter
            optimizer.zero_grad()
            loss = cnnmrf(synthesis)
            loss.backward(retain_graph=True)
            # print loss
            if (iter + 1) % 10 == 0:
                print('res-%d-iteration-%d: %f' % (i + 1, iter + 1, loss.item()))
            # save image
            if (iter + 1) % config.sample_step == 0 or iter + 1 == config.max_iter:
                image = get_synthesis_image(synthesis, denorm_transform, device)
                image = F.interpolate(image.unsqueeze(0), size=content_image.shape[2:4], mode='bilinear')
                torchvision.utils.save_image(image.squeeze(), 'res-%d-result-%d.jpg' % (i + 1, iter + 1))
                print('save image: res-%d-result-%d.jpg' % (i + 1, iter + 1))
            iter += 1
            if iter == config.max_iter:
                iter = 0
            return loss

        "----------------------"
        optimizer.step(closure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str, default='./dataset/content.jpg')
    parser.add_argument('--style_path', type=str, default='./dataset/style.jpg')
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=50)
    parser.add_argument('--content_weight', type=float, default=1)
    parser.add_argument('--style_weight', type=float, default=0.4)
    parser.add_argument('--tv_weight', type=float, default=0.1)
    parser.add_argument('--num_res', type=int, default=3)
    parser.add_argument('--gpu_chunck_size', type=int, default=256)
    parser.add_argument('--mrf_style_stride', type=int, default=2)
    parser.add_argument('--mrf_synthesis_stride', type=int, default=2)
    config = parser.parse_args()
    print(config)
    main(config)
