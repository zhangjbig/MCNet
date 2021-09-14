import cv2
import numpy as np
import os
from PIL import Image
from tqdm import trange
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import collections
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DDP
from utils.logger import init_log_config
from pandas import DataFrame as DF
from utils.transformer import Tensor2Predict
import sys

class VideoPredicter:
    def __init__(self, net, net_name, config, device, model_path, mp4_path):
        '''预测的频率'''
        self.frequency = 1
        
        '''模型的参数'''
        self.config = config
        self.net = net
        self.net_name = net_name
        self.path = model_path
        self.weight_decay = config[net_name]["weight_decay"]
        self.learning_rate = config[net_name]["learning_rate"]
        self.mp4_path = mp4_path
        
        if config[net_name]["optimizer"] == "momentum":
            self.optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=self.learning_rate, weight_decay=self.weight_decay)
        if config[net_name]["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.net.cuda()

        checkpoint = torch.load(model_path, map_location='cpu')
        self.net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint["model"].items()})
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O0')
        amp.load_state_dict(checkpoint['amp'])
        self.net = self.net.to(device)
        self.net.eval()
        self.tensor2predict = Tensor2Predict((512, 512))

    '''逐帧读取mp4文件并且预测'''
    def video_to_frames(self):
        videoCapture = cv2.VideoCapture()
        videoCapture.open(self.mp4_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        size = (512*2 , 512)
        avi_path = os.path.join(os.path.split(self.mp4_path)[0], self.net_name + "_001.avi")
        videoWriter = cv2.VideoWriter(avi_path,cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)
        for i in trange(int(frames)):
            ret , frame = videoCapture.read()
            if i % self.frequency != 0:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype("float32")
            frame = frame[350:880,990:1640,:] / 255.
            predict = self.image_predict(frame)
            frame = cv2.resize(frame, (512, 512))
            result = np.concatenate([frame*255, predict*255], 1).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            videoWriter.write(result)
             
    def image_predict(self, image):
        # print(image.shape)
        image = transforms.ToTensor()(image).unsqueeze(0)
        image = F.upsample(image, size=(512, 512))
        image = image.cuda(non_blocking=True)
        if self.net_name == "dfnnet":
                smooth_sides, border_side = self.net(image)
                predicts = smooth_sides[0]
        else:
            predicts = self.net(image)
            if isinstance(predicts,collections.OrderedDict) or isinstance(predicts,dict):
                predicts = predicts['out']
            if isinstance(predicts, collections.Sequence):
                predicts = predicts[0]
        predict = self.tensor2predict(predicts)
        predict = predict[0]
        return predict

    def __call__(self):
        self.video_to_frames()

def video2predict(net, net_name, config, device, model_path, mp4_path):
    predicter = VideoPredicter(net, net_name, config, device, model_path, mp4_path)
    predicter()

