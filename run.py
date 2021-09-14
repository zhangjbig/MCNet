import yaml
import os
from utils.trainer import train
from models import ec3net
from models import dfanet
from models import dfnnet
from models import u2net
from models import unet
from models import icnet
from models import dfanet
from models import hrnet
from models import pspnet
from models import bisenetv2
from models import bisenetv1
from models import mcnet

from apex.parallel import convert_syncbn_model
from models import mynet17

import torchvision.models as model
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()
    return args
def main():
    
    args = parse()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group('nccl',init_method='env://')
    file_path = "config.yml"
    file = open(file_path, 'r', encoding='utf-8')
    net_configs = yaml.load(file.read(), Loader=yaml.Loader)

    for net_name in net_configs["net_names"]:

        if net_name == "unet":
            net = unet.unet(net_configs["num_classes"])
        
        if net_name == "mcnet":
            net = mcnet.MCNet(net_configs["num_channels"])

        if net_name == "deeplabv3+":
            net = model.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=3)

        if net_name == "bisenetv1":
            net = bisenetv1.BiSeNetV1(net_configs["num_channels"])

        if net_name == "bisenetv2":
            net = bisenetv2.bisenetv2(net_configs["num_channels"])

        if net_name == "icnet":
            net = icnet.icnet(net_configs["num_channels"])

        if net_name == "pspnet":
            net = pspnet.pspnet(net_configs["num_classes"])

        if net_name == "hrnet":
            net = hrnet.hrnet(net_configs["num_classes"])

        if net_name == "u2net":
            net = u2net.u2net(net_configs["num_classes"])

        if net_name == "ec3net":
            net =  ec3net.ExtremeC3Net(net_configs["num_classes"])

        if net_name == "dfnnet":
            net = dfnnet.dfnnet(net_configs["num_classes"])

        if net_name == "dfanet":
            net = dfanet.DFANet(net_configs["num_classes"])        

        device = torch.device('cuda:{}'.format(args.local_rank))
        net = convert_syncbn_model(net)
        print(net_name)
        train(net, net_name = net_name, config=net_configs, device=device, args = args)
        

if __name__ == "__main__":
    main()
