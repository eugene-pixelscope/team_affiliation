import torchvision
import cv2
import numpy as np
from torchvision.transforms import v2
import torch


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


class Transforms:
    def __init__(self, size, mean=None, std=None):
        if type(size) == int:
            size = (size, size)
        transform = [
            v2.Resize(size=size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]
        if mean and std:
            transform.append(v2.Normalize(mean=mean, std=std))
        self.transform = v2.Compose(transform)

    def __call__(self, x):
        return self.transform(x)


class TrainTransforms:
    def __init__(self, size, mean=None, std=None):
        if type(size) == int:
            size = (size, size)
        transform = [
            v2.Resize(size=size),
            # v2.ColorJitter(brightness=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]
        if mean and std:
            transform.append(v2.Normalize(mean=mean, std=std))
        self.transform = v2.Compose(transform)

    def __call__(self, x):
        return self.transform(x)


class TransformsForInfer:
    def __init__(self, size, mean=None, std=None):
        if type(size) == int:
            size = (size, size)
        transform = [
            v2.Resize(size=size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]
        if mean and std:
            transform.append(v2.Normalize(mean=mean, std=std))
        self.transform = v2.Compose(transform)

    def __call__(self, x):
        return self.transform(x)


class TransformsForCV:
    def __init__(self, size, mean=None, std=None):
        self.transform = [
            torchvision.transforms.CenterCrop(size=size),
            torchvision.transforms.ToTensor()
        ]
        if mean and std:
            self.transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.transform = torchvision.transforms.Compose(self.transform)

    def __call__(self, *args):
        return self.transform(*args)
