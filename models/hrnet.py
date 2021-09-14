import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 
                    kernel_size=3, stride=stride,
                    padding=1, bias=False)

BN_MOMENTUM = 0.01
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.dowsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        if self.dowsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)

        self.conv3 = nn.Conv2d(channels, channels*self.expansion, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(channels*self.expansion, momentum=BN_MOMENTUM)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.relu(out)
        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_in_channels, num_channels):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_in_channels, num_channels)

        self.num_in_channels = num_in_channels
        self.num_branches = num_branches
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fuse_layers = self._make_fuse_layers()

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_in_channels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_in_channels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_in_channels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_in_channels = self.num_in_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_in_channels[j],
                                  num_in_channels[i],
                                  1, 1, 0,
                                  bias=False),
                        nn.BatchNorm2d(num_in_channels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(nn.MaxPool2d(kernel_size=1,stride=1))
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_in_channels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_in_channels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_in_channels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_in_channels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_in_channels

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_in_channels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_in_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_in_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse



class hrnet(nn.Module):
    def __init__(self, num_classes = 3):
        super(hrnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,3,stride=2,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, 3,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 32, 4)
        
        stage1_out_channel = Bottleneck.expansion*32

        self.transition1 = self._make_transition_layer([stage1_out_channel], [32, 64])
        self.stage2, pre_stage_channels = self._make_stage([32, 64])

        self.transition2 = self._make_transition_layer(pre_stage_channels, [32, 64, 128])
        self.stage3, pre_stage_channels = self._make_stage([32,64,128])

        self.transition3 = self._make_transition_layer(pre_stage_channels, [32, 64, 128, 256])
        self.stage4, pre_stage_channels = self._make_stage([32,64,128,256])
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                last_inp_channels, last_inp_channels,
                kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(last_inp_channels,num_classes,kernel_size=1,stride=1,padding=0)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride !=1 or in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion, momentum=BN_MOMENTUM)
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels*block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    

    def _make_stage(self, num_in_channels):
        modules = []
        num_branches = len(num_in_channels)
        block = BasicBlock
        num_channels = num_in_channels
        num_blocks = [4 for _ in range(num_branches)]
        num_modules = 1
        for _ in range(num_modules):
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_in_channels,
                    num_channels
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], 
                                      num_channels_cur_layer[i], 
                                      kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.MaxPool2d(1, 1))
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    if j == i-num_branches_pre:
                        out_channels = num_channels_cur_layer[i]
                    else:
                        out_channels = in_channels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,bias=False),
                        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def forward(self, x):
        input = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        x = self.stage4(x_list)

        x0_h, x0_w = x[0].shape[2], x[0].shape[3]
        x1 = F.interpolate(x[1], size=(x0_h, x0_w))
        x2 = F.interpolate(x[2], size=(x0_h, x0_w))
        x3 = F.interpolate(x[3], size=(x0_h, x0_w))
        
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        x = F.interpolate(x, size=input.shape[2:])
        x = self.out_conv(x)
        return x 
if __name__ == "__main__":
    img = torch.ones((1,3,512,512))
    model = hrnet()
    out = model(img)
    print(out.shape)
