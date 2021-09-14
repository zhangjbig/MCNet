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


def adjust_learning_rate(origin_lr, optimizer, epoch, num_epoch, power = 0.9):
    lr = origin_lr * (1 - epoch / num_epoch) ** power
    for p in optimizer.param_groups:
        p['lr'] = lr
    return lr

class trainer:
    def __init__(self, net, net_name, config, device, args):
        '''模型的参数'''
        self.config = config
        self.net = net
        self.net_name = net_name
        self.batch_size = config[net_name]["batch_size"]
        self.epochs = config[net_name]["epochs"]
        self.weight_decay = config[net_name]["weight_decay"]
        self.lr_decay = config[net_name]["lr_decay"]
        self.iterations = config[net_name]["iterations"]
        #self.frequency = config[net_name]["frequency"]
        self.learning_rate = config[net_name]["learning_rate"]
        self.iteration = 0

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        
        save_dir = os.path.join(self.config["save_result_dir"], self.net_name)
        # self.print_file = open(os.path.join(save_dir ,self.net_name + '.txt'),'w')
        '''随机尺度缩放'''
        self.randomscale = RandomScale()
        
        '''评估函数'''
        self.measure = measure(net_name, config)

        '''优化器'''
        if config[net_name]["optimizer"] == "momentum":
            self.optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=self.learning_rate, weight_decay=self.weight_decay)
        if config[net_name]["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.dataloader = dataload(config, net_name ,mode = "train")
        
        '''损失函数'''
        self.cross_entropy_loss = CrossEntropyLoss2d().to(device)
        self.focal_loss = FocalLoss().to(device)
        
        '''各项损失的权重'''
        if "loss_weight" in config[net_name]:
            self.loss_weight = config[net_name]["loss_weight"]
        else:
            self.loss_weight = 1
        
        '''tensor转图片的类'''
        self.tensor2image = Tensor2Image(config["image_size"])
        self.tensor2predict = Tensor2Predict(config["image_size"])
        self.tensor2label = Tensor2Label(config["image_size"])

        if args.local_rank == 0:
            self.logger = init_log_config(save_dir, net_name)
        

        if args.pretrain == 0:
            self.net = self.net.to(device)
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level="O0")
            self.net = DDP(self.net, delay_allreduce=True)
            self.args = args

        else:
            self.net.cuda()
            checkpoint = torch.load(os.path.join(config["save_model_dir"], net_name, net_name+"_"+str(args.pretrain)+".pt"), map_location='cpu')
            #print(checkpoint['model'].keys())
            self.net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint["model"].items()})
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O0')
            amp.load_state_dict(checkpoint['amp'])
            self.net = self.net.to(device)
            self.net = DDP(self.net, delay_allreduce=True)
            self.args = args
    
    def process(self,epoch):
        self.acc, self.iou, self.loss = [], [], []
        for batch_id, (images, labels) in enumerate(self.dataloader):
            images = images.permute([0, 3, 1, 2])
            labels = labels.permute([0, 3, 1, 2])
            images, labels = self.randomscale(images, labels, self.iteration)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            #print("image-shape:{}".format(images.shape))
            if self.net_name == "dfnnet":
                smooth_sides, border_side = self.net(images)
                smooth_loss = [self.cross_entropy_loss(smooth_side, labels) for smooth_side in smooth_sides]
                border_loss = self.focal_loss(border_side, labels)
                loss = sum(smooth_loss) + border_loss
                predicts = smooth_sides[0]

            else:
                predicts = self.net(images)
                if isinstance(predicts,collections.OrderedDict) or isinstance(predicts,dict):
                    predicts = predicts['out']
                if isinstance(predicts, collections.Sequence):
                    if not isinstance(self.loss_weight, collections.Sequence):
                        self.loss_weight = [1 for _ in range(len(predicts))]
                    loss = [self.cross_entropy_loss(predict, labels, loss_weight) for predict, loss_weight 
                                                                            in zip(predicts, self.loss_weight)]
                    predicts = predicts[0]
                    loss = sum(loss)

                else:
                    #print(predicts.keys())
                    #print(predicts.shape)
                    #print(labels.shape)
                    loss = self.cross_entropy_loss(predicts, labels)

            temp_acc = self.measure.acc(predicts, labels)
            temp_iou = np.array(self.measure.iou(predicts, labels))
            temp_loss = loss.detach().cpu().numpy()
            # if batch_id % 10 == 0:
            #   print("{},epoch-{}, batch_id-{}, loss-{}, iou-{}, acc-{}".format(self.net_name,epoch, batch_id, temp_loss, temp_iou,temp_acc)
            self.optimizer.zero_grad()

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
               scaled_loss.backward()

            # loss.backward()
            self.optimizer.step()

            self.iteration += 1

            if self.args.local_rank == 0:
                self.logger.info("{},iteration-{}, loss-{}, iou-{}, acc-{}\n".format(self.net_name,self.iteration, temp_loss, temp_iou,temp_acc))
                if self.iteration % self.config["image_save_frequency"] == 0:
                    self.save_image(images, predicts, labels)
                    pass
            
    def save_image(self, images, predicts, labels):
        '''保存实际图片 预测图片 标记图片'''
        save_dir = os.path.join(self.config["save_image_dir"], self.net_name, "train")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        a = self.tensor2image(images)
        b = self.tensor2predict(predicts)
        c = self.tensor2label(labels)

        save_img = np.concatenate((np.concatenate(a),np.concatenate(b),np.concatenate(c)), 1)
        save_img = Image.fromarray(save_img)
        save_img.save(os.path.join(save_dir, str(self.iteration)+'.png'))
    
    def save_model(self, epoch):
        '''保存模型文件'''
        #save_dir = os.path.join(self.config["save_model_dir"], self.net_name)
        save_dir = os.path.join(self.config["save_model_dir"], self.net_name)        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint = {
            'model':self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': amp.state_dict()
        }
        torch.save(checkpoint, os.path.join(save_dir, self.net_name+'_'+str(epoch) + '.pt'))

    # def save_result(self):
    #     '''保存acc, iou 到 csv中'''
    #     save_dir = os.path.join(self.config["save_result_dir"], self.net_name)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     df = DF(index = [ i+1 for i in range(self.iteration)], columns=["bg_iou","liquid_iou","solid_iou", "acc", "loss"])
    #     indexs = df.columns.tolist()
    #     self.iou = np.array(self.iou)
    #     print(self.iou.shape)
    #     for i in range(3):
    #         df[indexs[i]] = self.iou[:, i]
    #     df['acc'] = self.acc
    #     df["loss"] = self.loss
    #     df.to_csv(os.path.join(save_dir, self.net_name + '.csv'))

    def __call__(self):
        self.net.train()
        #if self.args.local_rank == 0:
         #   self.save_model(0)
        for epoch in range(self.epochs):
            self.process(epoch)
            if (epoch+1) % self.config["model_save_frequency"] == 0 and self.args.local_rank == 0:
                 self.save_model(epoch+1)
            lr = adjust_learning_rate(self.learning_rate ,self.optimizer, epoch, self.epochs)
            if self.args.local_rank == 0:
                self.logger.info("lr-{}\n".format(lr))

def train(net = None, net_name =None, config=None, device=None, args=None):
    Trainer = trainer(net, net_name, config, device, args)
    Trainer()
