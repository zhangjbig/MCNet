import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()
        weight_CE = torch.FloatTensor([1,1,1])
        self.loss = nn.CrossEntropyLoss(weight=weight_CE)

    def forward(self, logits, labels, weight=1):
        if logits.shape != labels.shape:
            labels = F.interpolate(labels, size=logits.shape[2:], mode="nearest")
        labels = torch.argmax(labels, dim=1)
        return self.loss(logits, labels)

class FocalLoss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, logits, labels, weight=1, alpha=0.25, gamma=2.0, num_classes=3):

        labels = torch.argmax(labels, dim=1)
        logits = logits.permute([0, 2, 3, 1])
        logits = logits.reshape([-1,num_classes])
        labels = labels.reshape([-1])
        labels = F.one_hot(labels.long(), num_classes)
        pk =  torch.sum(logits * labels, 1, keepdim= True)
        f1 = -alpha * torch.mean(torch.pow(1 - pk, gamma))* torch.log(torch.clamp(pk, 1e-12, 1.0))
        f1 = torch.mean(f1)
        f1 = weight*torch.sum(f1)
        return f1
