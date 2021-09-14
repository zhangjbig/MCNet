import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet50

class dfnnet(nn.Module):
    def __init__(self, num_class, encoder='resnet', weights=None, pic_size = (256,512)):
        super(dfnnet, self).__init__()
        last_block_pic_size = (pic_size[0]//32, pic_size[1]//32)
        if encoder == 'resnet':
            self.encoder = Stage_ResNet()
            stage_plane = [2048, 1024, 512, 256]
            global_plane = 2048

        if weights:
            self.encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            print('loading the {} pretrained weight ...'.format(encoder))
        self.smooth = SmoothNet(num_class, stage_plane, gloabl_plane=global_plane, pic_size=last_block_pic_size)
        self.border = BorderNet(num_class, stage_plane[::-1])

    def forward(self, x):
        out = self.encoder(x)
        b1, b2, b3, b4, fuse = self.smooth(out)
        r1 = self.border(out)
        return [fuse, b1, b2, b3, b4], r1

class Stage_ResNet(nn.Module):
    def __init__(self, encoder = 'resnet50'):
        super(Stage_ResNet, self).__init__()
        orig_resnet = resnet50()
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);
        global_pool = F.adaptive_avg_pool2d(x, (1, 1)); conv_out.append(global_pool);
        return conv_out


class RRB(nn.Module):
    def __init__(self, inplanes, planes, interplanes=512):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, interplanes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace= True),
            nn.Conv2d(interplanes, planes, 3, padding=1))
        self.relu = nn.LeakyReLU(inplace= True)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv2(out)
        return self.relu(out + out1)


class CAB(nn.Module):
    def __init__(self, inplanes, interplanes=512):
        super(CAB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes * 2, interplanes, 1),
            nn.LeakyReLU(inplace= True),
            nn.Conv2d(interplanes, inplanes, 1))

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        out = F.avg_pool2d(x, x.size()[2:4], stride=x.size()[2:4])
        out = self.conv1(out)
        out = torch.sigmoid(out)
        return out * x1 + x2


def side_branch(class_num, factor, inplanes=512):
    branch = nn.Sequential(
        nn.Conv2d(inplanes, class_num, 1),
        # hout = (hin - kernel + 2*padding)/stride +1 :(32-1+0)/1+1=32  hout = (hin-1)*stride-2*padding+kernel+outputpadding
        nn.Upsample(scale_factor= factor, mode='bilinear'))# nn.ConvTranspose2d(class_num, class_num, factor, stride=factor))  # (512-1)*2-2*0+2
    return branch


class SmoothBlock(nn.Module):
    def __init__(self, plane1, plane2, kernel):
        super(SmoothBlock, self).__init__()
        self.rrb = RRB(plane1, 512)
        self.cab = CAB(512)
        self.rrbb = RRB(512, 512)
        # self.transpose = nn.ConvTranspose2d(plane2, 512, kernel, stride=1)

    def forward(self, x1, x2):
        input1 = self.rrb(x1)
        input2 = F.interpolate(x2, x1.size()[2:4], mode='bilinear') # input2 = self.transpose(x2)
        cab_out = self.cab(input1, input2)
        rrb_out = self.rrbb(cab_out)
        return input2, rrb_out


class SmoothBlockwithBranch(SmoothBlock):
    def __init__(self, plane1, factor, class_num, inplanes=512):
        super(SmoothBlockwithBranch, self).__init__(plane1, 512, 1)
        # self.transpose = nn.ConvTranspose2d(512, 512, 1, stride=2, output_padding=1)
        self.branch = side_branch(class_num, factor, inplanes)

    def forward(self, x1, x2):
        input2, rrb_out = super().forward(x1, x2)
        b = self.branch(input2)
        return b, rrb_out




class SmoothNet(nn.Module):
    def __init__(self, class_num, planes, gloabl_plane, pic_size):
        super(SmoothNet, self).__init__()
        factor = [16, 8, 4, 2]
        self.class_num = class_num
        self.first_cov = nn.Conv2d(gloabl_plane, 512, 1)
        self.block4 = SmoothBlock(planes[0], plane2 = gloabl_plane, kernel=pic_size)
        self.block3 = SmoothBlockwithBranch(planes[1], factor[0], class_num=class_num)
        self.block2 = SmoothBlockwithBranch(planes[2], factor[1], class_num=class_num)
        self.block1 = SmoothBlockwithBranch(planes[3], factor[2], class_num=class_num)
        # self.transpose = nn.ConvTranspose2d(512, 512, 1, stride=2, output_padding=1)
        self.branch = side_branch(class_num, factor[3], 512)
        self.conv = nn.Conv2d(class_num * 4, class_num, 1)
        # self.patched = False

    def forward(self, x):
        # if not self.patched:
        #     self.patch(tuple(x[-2].size()[2:4]), x[-2].device, x[-1].size()[1])

        _, out = self.block4(x[-2], self.first_cov(x[-1]))
        b4, out = self.block3(x[-3], out)
        b3, out = self.block2(x[-4], out)
        b2, out = self.block1(x[-5], out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear') # out = self.transpose(out)
        b1 = self.branch(out)
        b = torch.cat((b1, b2, b3, b4), 1)
        fuse = self.conv(b)
        return b1, b2, b3, b4, fuse # return b1, b2, b3, b4, b1


class BorderBlock(nn.Module):
    def __init__(self, planes, stride=2):
        super(BorderBlock, self).__init__()
        self.rrb1 = RRB(planes, 512)
        # self.transpose = nn.ConvTranspose2d(512, 512, 1, stride=stride, output_padding=stride - 1)  # (32-1)*4-2*0+1+3
        self.rrb2 = RRB(512, 512)

    def forward(self, x1, x2):
        out = self.rrb1(x1)
        out = F.interpolate(out, x2.size()[2:4], mode='bilinear')# out = self.transpose(out)
        out = out + x2
        out = self.rrb2(out)
        return out


class BorderNet(nn.Module):
    def __init__(self, class_num, planes):
        super(BorderNet, self).__init__()
        self.rrb = RRB(planes[0], 512)
        self.block1 = BorderBlock(planes[1], 2)
        self.block2 = BorderBlock(planes[2], 4)
        self.block3 = BorderBlock(planes[3], 8)
        # self.transpose = nn.ConvTranspose2d(512, 512, 1, stride=2, output_padding=1)
        self.branch = side_branch(class_num, 2, 512)

    def forward(self, x):
        out = self.rrb(x[0])
        out = self.block1(x[1], out)
        out = self.block2(x[2], out)
        out = self.block3(x[3], out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')# out = self.transpose(out)
        return self.branch(out)

if __name__ == "__main__":
    img = torch.ones((1, 3 ,512, 512))
    model = DFN(3)
    outs = model(img)
    for out in outs:
        print(out.shape)
