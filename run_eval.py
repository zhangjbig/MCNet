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
from models import mcnet
from models import bisenetv2
import torchvision.models as model
from models import mynet7
import argparse
import torch.nn as nn
import torch
from tqdm import trange
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DDP
import os



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--start_model", type=int, default=0)
    args = parser.parse_args()
    return args
    

def main():
    args = parse()
    torch.cuda.set_device(args.local_rank)

    file_path = "config.yml"
    file = open(file_path, 'r', encoding='utf-8')
    net_configs = yaml.load(file.read(), Loader=yaml.Loader)

    net_name_list = net_configs["net_names"]

    for net_name in net_name_list:
        if net_name == "unet":
            net = unet.unet(3)

        elif net_name == "mcnet":
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
            net =  ec3net.ExtremeC3Net(net_configs["num_classes"])

        elif net_name == "dfnnet":
            net = dfnnet.dfnnet(3)

        elif net_name == "dfanet":
            net =  dfanet.DFANet(3)

        else:
            continue

        device = torch.device('cuda:{}'.format(args.local_rank))
        #print(device)
        save_model_path  = os.path.join(net_configs["save_model_dir"], net_name)
        save_model_paths = os.listdir(save_model_path)
        for i in trange(len(save_model_paths)):
            save_model_path = save_model_paths[i]
            eval(net, net_name, net_configs, device, args, save_model_path, True)
        

if __name__ == "__main__":
    main()
