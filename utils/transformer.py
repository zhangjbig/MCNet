import numpy as np
import random
from PIL import Image,ImageOps,ImageFilter,ImageEnhance
import torch.nn as nn
import torch.nn.functional as F
import torch

class Normalize():
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
    def __call__(self, image, label):
        image = np.array(image).astype(np.float32)
        label = np.array(label).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return  image, label

class Tensor2Image():
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        image = F.interpolate(image, size=self.size, mode='bilinear', align_corners=True)
        image = image * 255
        image = image.permute([0, 2, 3, 1]).cpu().numpy().astype(np.uint8)
        return image

class Tensor2Label():
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        image = F.interpolate(image, size=self.size, mode="nearest")
        image = image.permute([0, 2, 3, 1]).cpu().numpy().astype(np.uint8)*255
        return image

class Tensor2Predict():
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = F.interpolate(image, size=self.size, mode='bilinear', align_corners=True)
        image = torch.argmax(image, dim=1)
        image = F.one_hot(image.long(), 3) * 255
        image = image.cpu().numpy().astype(np.uint8)
        return image

class RandomHorizontalFlip():
    def __init__(self, factor = 0.5):
        self.factor = factor

    def __call__(self, image, label):
        if random.random() < self.factor:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return image, label

class RandomRotate():
    def __init__(self, degree = [0, 90, 180, 270]):
        self.degree = degree

    def __call__(self, image, label):
        rotate_degree = random.choice(self.degree)
        image = image.rotate(rotate_degree, Image.BILINEAR)
        label = label.rotate(rotate_degree, Image.NEAREST)
        return image, label

class RandomGaussianBlur():
    def __init__(self, factor=0.5):
        self.factor = factor

    def __call__(self, image, label):
        if random.random() < self.factor:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return image, label

class RandomScale():
    def __init__(self, factor=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0]):
        self.factor = factor
    def __call__(self, image, label, seed):
        random.seed(seed)
        self.random_factor = random.choice(self.factor)
       # print(seed, self.random_factor)
        image = F.interpolate(image, scale_factor=self.random_factor, mode='bilinear', align_corners=True)
        label = F.interpolate(label, scale_factor=self.random_factor, mode='nearest')
        return image, label

class RandomCrop():
    def __init__(self, base_size, crop_size=(80,80)):
        self.base_size = base_size
        self.crop_size = crop_size
        self.expand_size = (base_size[0]+crop_size[0], base_size[1]+crop_size[1])
    
    def __call__(self, image, label):
        x = random.randint(0, self.crop_size[0])
        y = random.randint(0, self.crop_size[1])
        image = image.resize(self.expand_size, Image.BILINEAR)
        label = label.resize(self.expand_size, Image.NEAREST)
        image = image.crop((x, y, x+self.base_size[0], y+ self.base_size[1]))
        label = label.crop((x, y, x+self.base_size[0], y+self.base_size[1]))
        return image, label 

class FixedResize():
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = image.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)
        return image, label 

class RandomColor():
    def __init__(self, factor=0.5):
        self.factor = factor
        self.color_factor = np.random.randint(8, 12) / 10.
        self.brightness_factor = np.random.randint(8, 12) / 10.
        self.contrast_factor = np.random.randint(8, 12) / 10.
        self.sharp_factor = np.random.randint(8, 12) / 10.
    
    def __call__(self, image, label):
        color_image = ImageEnhance.Color(image).enhance(self.color_factor)  # 调整图像的饱和度
        brightness_image = ImageEnhance.Brightness(color_image).enhance(self.brightness_factor)  # 调整图像的亮度
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(self.contrast_factor)  # 调整图像对比度
        sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(self.sharp_factor)
        return  sharp_image, label # 调整图像锐度
