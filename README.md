# MCNet

![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)

## Document description

```
---| models  # Folder for storing various neural networks
---| utils # Folder for storing various scripts
---| selectBest.py # Select the best result from the result dir
---| run.py  # the start script of training
---| run_eval.py # the start script of evaling
---| infer_time.py # the script for calculate the infer speed of network
---| config.yml # configuration file
```

## Model Zoo

|Model|
|---|
|[UNet](./models/unet.py)|
|[PSPNet](./models/pspnet.py)|
|[HRNet](./models/hrnet.py)|
|[ICNet](./models/icnet.py)|
|[DFNNet](./models/dfnnet.py)|
|[DFANet](./models/dfanet.py)|
|[EC3Net](./models/ec3net.py)|
|[DeepLabV3+](./models/deeplab.py)|
|[U2Net](./models/u2net.py)|
|[BiSeNetV1](./models/bisenetv1.py)|
|[BiSeNetV2](./models/bisenetv2.py)|
|[MCNet](./models/mcnet.py)|


## Installation

System Requirements:
* Python >= 3.6+
* Pytorch >= 1.7.0
* cuda >= 11.0

In order to obtain multi-card synchronization batchnorm larization, we use [APEX](https://github.com/NVIDIA/apex)


``` sh
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Start

How to train

```sh
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py
```

How to eval
```sh
python run_eval.py --local_rank=0 
```

