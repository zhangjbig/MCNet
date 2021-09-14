import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import collections
from utils.dataloader import dataload
from utils.transformer import RandomScale, Tensor2Image, Tensor2Predict, Tensor2Label
from utils.criterion import CrossEntropyLoss2d, FocalLoss
from utils.measure import measure
import matplotlib.pyplot as plt
import collections
from PIL import Image
import collections
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DDP
from utils.logger import init_log_config
from pandas import DataFrame as DF

class evaler:
    def __init__(self, net, net_name, config, device, args, path, save_image):
        '''模型的参数'''
        self.config = config
        self.net = net
        self.net_name = net_name
        self.path = path
        
        self.weight_decay = config[net_name]["weight_decay"]
        self.learning_rate = config[net_name]["learning_rate"]
        self.iteration = 0
        self.batch_size = config[net_name]["batch_size"]
    
        self.is_save_image = save_image

        self.iou, self.acc = [], []
        
        '''评估函数'''
        self.measure = measure(net_name, config)

        '''优化器'''
        if config[net_name]["optimizer"] == "momentum":
            self.optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=self.learning_rate, weight_decay=self.weight_decay)
        if config[net_name]["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.dataloader = dataload(config,net_name,mode = "eval")
        
        '''tensor转图片的类'''
        self.tensor2image = Tensor2Image(config["image_size"])
        self.tensor2predict = Tensor2Predict(config["image_size"])
        self.tensor2label = Tensor2Label(config["image_size"])
        self.net.cuda()

       # self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level="O0")
       
        #print("check point:{}".format(os.path.join(config["save_model_dir"], net_name, path)))
        #print("path :{}".format(path))
        checkpoint = torch.load(os.path.join(config["save_model_dir"], net_name, path), map_location='cpu')
        #print(checkpoint['model'].keys())
        self.net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint["model"].items()})

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O0')
        amp.load_state_dict(checkpoint['amp'])
        self.net = self.net.to(device)
        self.args = args

    def process(self,epoch=0):

        for index, (images, labels) in enumerate(self.dataloader):

            images = images.permute([0, 3, 1, 2])
            labels = labels.permute([0, 3, 1, 2])
            images.float()
            #print(type(images))
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            #print(type(images))

            if self.net_name == "dfnnet":
                smooth_sides, _ = self.net(images)
                predicts = smooth_sides[0]
            else:
                predicts = self.net(images)
                if isinstance(predicts,collections.OrderedDict) or isinstance(predicts,dict):
                    predicts = predicts['out']
                if isinstance(predicts, collections.Sequence):
                    predicts = predicts[0]
            #print('pre_shape:{}'.format(predicts.shape)) 
            temp_acc = self.measure.acc(predicts, labels)
            temp_iou = np.array(self.measure.w_iou(predicts, labels))
            
            self.iou.extend(list(temp_iou))
            self.iteration += 1

            if self.is_save_image:
                self.save_image(images, predicts, labels)
    
    def save_image(self, images, predicts, labels):
        '''保存实际图片 预测图片 标记图片'''
        save_dir = os.path.join(self.config["save_image_dir"], self.net_name,"eval", self.path.split('.')[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        a = self.tensor2image(images)
        b = self.tensor2predict(predicts)
        c = self.tensor2label(labels)

        save_img = np.concatenate((np.concatenate(a),np.concatenate(b),np.concatenate(c)), 1)
        save_img = Image.fromarray(save_img)
        save_img.save(os.path.join(save_dir, str(self.iteration)+'.png'))

    def save_result(self):
        '''保存acc, iou 到 csv中'''
        save_dir = os.path.join(self.config["save_result_dir"], self.net_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df = DF(index = [ i+1 for i in range(self.iteration)], columns=["bg_iou","liquid_iou","solid_iou", "acc"])
        indexs = df.columns.tolist()
        self.iou = np.array(self.iou)
        for i in range(3):
            df[indexs[i]] = self.iou[:, i]
        df['acc'] = self.acc
        #print("save dir-{}".format(save_dir))
        #print("path -{}".format(self.path))
        df.to_csv(os.path.join(save_dir, self.path.split('.')[0] + '.csv'))

    def save_result_w(self):
        '''保存acc, iou 到 csv中'''
        save_dir = os.path.join(self.config["save_result_dir"], self.net_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df = DF(index = [ i+1 for i in range(len(self.iou))], columns=['w_miou'])
        indexs = df.columns.tolist()
        self.iou = np.array(self.iou)
        #print(self.iou)
        df[indexs[0]] = self.iou
        #df['acc'] = self.acc
        #print("save dir-{}".format(save_dir))
        #print("path -{}".format(self.path))
        df.to_csv(os.path.join(save_dir, self.path.split('.')[0] + '.csv'))


    def __call__(self):
        self.net.eval()
        self.process()
        self.save_result_w()
        

def eval(net = None, net_name =None, config=None, device=None, args=None, path=None, save_image=True):
    Evaler = evaler(net, net_name, config, device, args, path, save_image)
    Evaler()
