import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class measure:
    def __init__(self, net_name=None, config=None, mode='train'):
        self.config = config
        self.num_classes = config["num_classes"]
        if mode == "train":
            self.batch_size = config[net_name]["batch_size"]
        else:
            self.batch_size = 1
        self.shape = 0

    '''计算结果的矩阵'''
    def get_result(self, logits, labels):
        self.shape = logits.shape[2:]
        logits = logits.permute([0, 2, 3, 1])
        logits = logits.reshape([-1, self.num_classes])
        logits = logits.argmax(dim = 1)
        logits = F.one_hot(logits.long(), self.num_classes).int()

        labels = labels.permute([0, 2, 3, 1])
        labels = labels.reshape([-1, self.num_classes]).int()
        # print(torch.max(labels.argmax(1)))
        results = logits * labels
        return results, logits, labels
    
    '''计算acc'''
    def acc(self, logits, labels):
        results, logits, _ = self.get_result(logits, labels)
        AC = torch.sum(results).cpu().numpy().astype(np.float)
        SUM = np.prod(np.array(self.shape)) * self.batch_size
        return AC / SUM

    '''计算各项的iou'''
    def iou(self, logits, labels):
        results, logits, labels = self.get_result(logits, labels)
        TP = torch.sum(results, dim=0).float()
        SUM = torch.sum(labels, dim=0) + torch.sum(logits, dim=0)
        TP = TP.cpu().numpy()
        SUM = SUM.cpu().numpy()
        single_iou = np.zeros(TP.shape)
        idxNonZeros = np.where((SUM-TP) != 0)
        idxZeros = np.where((SUM-TP)==0)
        single_iou[idxNonZeros] = TP[idxNonZeros] / (SUM-TP)[idxNonZeros]
        single_iou[idxZeros] = 0
        #single_iou = (TP/(SUM-TP)).cpu().numpy()
        return single_iou


    def w_iou(self,logits, labels):
        #print(logits.shape)
        #print(labels.shape)
        #print(self.batch_size)
        #print('------------')
        batch_size = logits.shape[0]
        mious = np.zeros(batch_size)
        image_size = logits.shape[2] * logits.shape[3]
        for i in range(batch_size):
            #print(logits[i:i+1].shape)
            #print(labels[i:i+1].shape)
            temp_miou = self.iou(logits[i:i+1],labels[i:i+1])
            c1 = labels[i,0].sum()
            c2 = labels[i,1].sum()
            c3 = labels[i,2].sum()
            #temp_miou[0] *= c1 / image_size
            temp_miou[1] *= c2 / (c2 + c3)
            temp_miou[2] *= c3 / (c2 + c3)
            mious[i] += temp_miou[1] + temp_miou[2]
        #mious /= batch_size
        return mious
