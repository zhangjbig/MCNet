import sys
import os
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import numpy as np
from utils.transformer import Normalize, RandomHorizontalFlip, RandomRotate, RandomGaussianBlur, RandomCrop, FixedResize, RandomColor
import torch

class dataset(Dataset):
    def __init__(self, dataset_dir, transformers = None):
        self.dataset_dir = dataset_dir
        self.transformers = transformers
        self.image_paths = []
        self.label_paths = []
        self.generate_path()
       
    def save_img_path(self):
        with open('eval_img_paths.txt','w') as f:
            for i in self.label_paths:
                f.write(i + '\n')
   
    def save_paths(self):
        with open('img_paths.txt','w') as f:
            for i in self.image_paths:
                f.write(i + '\n')

        with open('lab_Paths.txt','w') as f:
            for i in self.label_paths:
                f.write(i + '\n')

    def generate_path(self):
    
        paths = os.walk(self.dataset_dir)
        label_name = []
        for root,dir,name in paths:
            if 'label' in root:
                self.label_paths.extend([os.path.join(root,label) for label in name if '.png' in label])
                label_name.extend([label for label in name if '.png' in label])
        
        self.image_paths = [label.replace('labels','images') for label in self.label_paths]

        self.image_paths.sort()
        print(len(self.image_paths))
        self.label_paths.sort()
        print(len(self.label_paths))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert("RGB")
        if self.transformers is not None:
            for transformer in self.transformers:
                image, label = transformer(image, label)
        
        #shape = label.shape
        '''将label图片转化为 0, 1标签'''
        #label = label.reshape([-1, 3]).argmax(1)
        #label = np.eye(3)[label].reshape(shape)
        return np.array(image), np.array(label)

'''获取迭代器'''
def dataload(config = None, net_name = None, mode = "train"):
    if mode == "train":
        transformers = [
            RandomRotate(),
            RandomHorizontalFlip(),
            RandomGaussianBlur(),
            RandomColor(),
            RandomCrop(config["image_size"]),
            FixedResize(config["image_size"]),
            Normalize(),
        ]
        #transformers = [
            #RandomRotate(),
            #RandomHorizontalFlip(),
            #RandomGaussianBlur(),
            #RandomColor(),
            #RandomCrop(config["image_size"]),
        #    FixedResize(config["image_size"]),
        #    RandomHorizontalFlip(),
        #    Normalize(),
        #]
                                   
        dataseter = dataset(config["train_dataset_dir"], transformers=transformers)
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataseter)
        print("dataseter-len{}".format(len(dataseter)))
        dataloader = DataLoader(dataseter, batch_size=config[net_name]["batch_size"],
                num_workers = 8,sampler=data_sampler,pin_memory=True)
        print("dataloader-len:{}".format(len(dataloader)))

    if mode == "eval":
        transformers = [
            FixedResize(config["image_size"]),
            Normalize(),
        ]
        dataseter = dataset(config["eval_dataset_dir"], transformers=transformers)
        dataloader = DataLoader(dataseter, batch_size=config[net_name]["batch_size"], shuffle=False)

    return dataloader
