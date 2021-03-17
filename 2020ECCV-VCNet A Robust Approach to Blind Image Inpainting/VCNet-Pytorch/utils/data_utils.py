import torch
import random
import string

from torch.autograd import Variable
from torch.utils import data


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def unnormalize_batch(batch, mean_, std_, div_factor=1.0):
    """
    Unnormalize batch
    :param batch: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :param div_factor: normalizing factor before data whitening
    :return: unnormalized data, tensor with shape
     (batch_size, nbr_channels, height, width)
    """
    # normalize using dataset mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = mean_[0]
    mean[:, 1, :, :] = mean_[1]
    mean[:, 2, :, :] = mean_[2]
    std[:, 0, :, :] = std_[0]
    std[:, 1, :, :] = std_[1]
    std[:, 2, :, :] = std_[2]
    batch = torch.div(batch, div_factor)

    batch *= Variable(std)
    batch = torch.add(batch, Variable(mean))
    return batch


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def linear_scaling(x):
    return (x * 255.) / 127.5 - 1.


def linear_unscaling(x):
    return (x + 1.) * 127.5 / 255.


import os
from PIL import Image
from torchvision import transforms
from utils.config import get_cfg_defaults


class RaindropDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.data_path = os.path.join(root, "data")
        self.gt_path = os.path.join(root, "gt")

        self.data = self._make_dataset(self.data_path)
        self.gt = self._make_dataset(self.gt_path)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self._read_img(self.data[index])
        y = self._read_img(self.gt[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return img, y

    def __len__(self):
        return len(self.data)

    def _make_dataset(self, target_dir):
        instances = list()
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                instances.append(path)
        return instances

    def _read_img(self, im_path):
        return Image.open(im_path).convert("RGB")


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    transform = transforms.Compose([transforms.Resize(cfg.DATASET.SIZE),
                                    transforms.CenterCrop(cfg.DATASET.SIZE),
                                     transforms.ToTensor()
                                     ])
    dataset = RaindropDataset("../datasets/raindrop/train20/train", transform=transform, target_transform=transform)
    img, y = dataset.__getitem__(3)
    print(img.size(), y.size())
