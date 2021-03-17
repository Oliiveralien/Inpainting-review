import math
import numbers
import torch
import numpy as np
import scipy.stats as st

from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw


class MaskGenerator:
    def __init__(self, opt):
        self.min_num_vertex = opt.MIN_NUM_VERTEX
        self.max_num_vertex = opt.MAX_NUM_VERTEX
        self.mean_angle = opt.MEAN_ANGLE
        self.angle_range = opt.ANGLE_RANGE
        self.min_width = opt.MIN_WIDTH
        self.max_width = opt.MAX_WIDTH
        self.min_removal_ratio = opt.MIN_REMOVAL_RATIO
        self.max_removal_ratio = opt.MAX_REMOVAL_RATIO

    def generate(self, H, W):
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)
            angle_min = self.mean_angle - np.random.uniform(0, self.angle_range)
            angle_max = self.mean_angle + np.random.uniform(0, self.angle_range)
            angles, vertex = list(), list()
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(np.random.normal(loc=average_radius, scale=average_radius // 2), 0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(self.min_width, self.max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (1, 1, H, W))

        if compute_known_pixels_weights(mask) < self.min_removal_ratio or \
                compute_known_pixels_weights(mask) > self.max_removal_ratio:
            return self.generate(H, W)
        return mask


def gauss_kernel(size=21, sigma=3, inchannels=3, outchannels=3):
    interval = (2 * sigma + 1.0) / size

    x = np.linspace(-sigma - interval / 2, sigma + interval / 2, size + 1)

    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()

    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, size, size))
    out_filter = np.tile(out_filter, [outchannels, inchannels, 1, 1])

    return out_filter


class GaussianBlurLayer(nn.Module):
    def __init__(self, size, sigma, in_channels=1, stride=1, pad=1):
        super(GaussianBlurLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.ch = in_channels
        self.stride = stride
        self.pad = nn.ReflectionPad2d(pad)

    def forward(self, x):
        kernel = gauss_kernel(self.size, self.sigma, self.ch, self.ch)
        kernel_tensor = torch.from_numpy(kernel)
        kernel_tensor = kernel_tensor.cuda()
        x = self.pad(x)
        blurred = F.conv2d(x, kernel_tensor, stride=self.stride)
        return blurred


class ConfidenceDrivenMaskLayer(nn.Module):
    def __init__(self, size=65, sigma=1.0 / 40, iters=7):
        super(ConfidenceDrivenMaskLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.iters = iters
        self.propagation_layer = GaussianBlurLayer(size, sigma, pad=size // 2)

    def forward(self, mask):
        # here mask 1 indicates missing pixels and 0 indicates the valid pixels
        init = 1 - mask
        mask_confidence = None
        for i in range(self.iters):
            mask_confidence = self.propagation_layer(init)
            mask_confidence = mask_confidence * mask
            init = mask_confidence + (1 - mask)
        return mask_confidence


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = kernel_size // 2
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


def compute_known_pixels_weights(m):
    known_ratio = np.sum(m) / (m.shape[2] * m.shape[3])
    return known_ratio


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


COLORS = {
    "BLUE": [0., 0., 1.],
    "RED": [1., 0., 0.],
    "GREEN": [0., 1., 0.],
    "WHITE": [1., 1., 1.],
    "BLACK": [0., 0., 0.],
    "YELLOW": [1., 1., 0.],
    "PINK": [1., 0., 1.],
    "CYAN": [0., 1., 1.],
}

if __name__ == '__main__':
    from config import _C

    generator = MaskGenerator(_C.MASK)

    # print(mask.size)
    # print(mask.sum())
    # print(compute_known_pixels_weights(mask))
    # im = Image.fromarray(mask.squeeze() * 255)
    # im.show()

    mask = torch.zeros((1, 1, 256, 256))
    for i in range(16):
        mask += torch.from_numpy(generator.generate(256, 256))
    mask = torch.clamp(mask, max=1., min=0.)

    smoother = GaussianSmoothing(1, 3, 1)
    mask = smoother(mask)
    im = Image.fromarray(mask.squeeze().numpy() * 255)
    im.show()
