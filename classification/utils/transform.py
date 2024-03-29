import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
import random

np.random.seed(0)


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class ResizeToResolution(object):
    def __init__(self, resolution):
        self.resolution = resolution
        
    def __call__(self, img):
        w,h = img.size
        if h <= w and h < self.resolution[0]:
            img = transforms.Resize((int(self.resolution[0]),int(w/h*self.resolution[0])))(img)
        elif w <= h and w < self.resolution[1]:
            img = transforms.Resize((int(h/w*self.resolution[1]),int(self.resolution[1])))(img)
        return img


class Identity(object):
    def __init__(self):
        pass
    def __call__(self, img):
        return img


class RandomResize(object):
    def __init__(self,scale,resolution):
        self.scale = scale
        self.resolution = resolution
    def __call__(self, img):
        frac = random.uniform(1.0, self.scale)
        img = transforms.Resize((int(frac*self.resolution), int(frac*self.resolution)))(img)
        return img
