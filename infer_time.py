from utils.evaler import eval
import yaml
from models import unet
from models import ec3net
from models import dfanet
from models import dfnnet
from models import u2net
from models import unet
from models import icnet
from models import dfanet
from models import hrnet
from models import pspnet
from models import bisenetv1
from models import bisenetv2
from models import mcnet

import torchvision.models as model

import time
import argparse
import torch.nn as nn
import torch
from pandas import DataFrame as DF
import pandas as pd
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DDP
import os
import tqdm
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args
    

def main():
    args = parse()
    torch.cuda.set_device(args.local_rank)


    # net_name_list = ['unet',"dfanet", "bisenetv2", "hrnet", "icnet", "u2net", "ec3net", "bisenetv1", "pspnet","dfnnet"]#, "deeplabv3+"]

    # net_name_list.extend(['mynet' + str(i) for i in range(1,18)])
    # net_name_list.remove('mynet4')
    # net_name_list.append("deeplabv3+")

    net_name_list = ["ec3net"]

    file_path = "config.yml"
    file = open(file_path, 'r', encoding='utf-8')
    net_configs = yaml.load(file.read(), Loader=yaml.Loader)

    result = pd.DataFrame(columns =["net_name", "fps"])

    i = 0 
    for net_name in net_name_list:
        if net_name == "unet":
            net = unet.unet(3)
        elif net_name == "mcnet":
            from models import mcnet
            net = mcnet.MCNet(3)

        elif net_name == "deeplabv3+":
            net = model.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=3)

        elif net_name == "bisenetv1":
            net = bisenetv1.BiSeNetV1(3)

        elif net_name == "bisenetv2":
            net =  bisenetv2.bisenetv2(3)

        elif net_name == "icnet":
            net = icnet.icnet(3)

        elif net_name == "pspnet":
            net = pspnet.pspnet(3)

        elif net_name == "hrnet":
            net = hrnet.hrnet(3)

        elif net_name == "u2net":
            net = u2net.u2net(3)

        elif net_name == "ec3net":
            net =  ec3net.ExtremeC3Net(3)

        elif net_name == "dfnnet":
            net = dfnnet.dfnnet(3)

        elif net_name == "dfanet":
            net =  dfanet.DFANet(3)

        else:
            print(net_name)
            continue   
        i += 1
        batch = 4
        inputt = torch.randn(batch, 3, 512, 512)
        #inputt = inputt.cuda()
        fps = 0
        net.eval()
       # net.cuda()
        #net.to(0)
        start = time.time()
        for _ in tqdm(range(100)):
            #start = time.time()
            outputs = net(inputt)
            #end = time.time()
            #temp_fps = 1/(end-start)
            #fps += temp_fps
        end = time.time()
        fps = (100 * batch) / (end - start)
        print("net:{}, fps-{}".format(net_name,fps))
        result.loc[i] = [net_name, fps]
    
    result.to_csv("cpu_result.csv")
        


if __name__ == "__main__":
    main()
