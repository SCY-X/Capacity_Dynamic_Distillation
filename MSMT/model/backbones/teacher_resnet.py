import torch.nn as nn
import torch.nn.functional as F
from model.backbones.convnet_block.convnet_utils import conv_bn, conv_bn_relu, ConvBN
import math
import torch

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, deploy_flag=False):
        super(BasicBlock, self).__init__()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_bn(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride, deploy_flag=deploy_flag)
        else:
            self.shortcut = nn.Identity()
        self.conv1 = conv_bn_relu(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, deploy_flag=deploy_flag)
        self.conv2 = conv_bn(in_channels=planes, out_channels=self.expansion * planes, kernel_size=3, stride=1, padding=1, deploy_flag=deploy_flag)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, deploy_flag=False):
        super(Bottleneck, self).__init__()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = conv_bn(in_planes, self.expansion*planes, kernel_size=1, stride=stride, deploy_flag=deploy_flag)
        else:
            self.shortcut = nn.Identity()

        self.conv1 = conv_bn_relu(in_planes, planes, kernel_size=1,  deploy_flag=deploy_flag)
        self.conv2 = conv_bn(planes, planes, kernel_size=3, stride=stride, padding=1, deploy_flag=deploy_flag)
        self.conv3 = conv_bn(planes, self.expansion*planes, kernel_size=1, deploy_flag=deploy_flag)
        self.feature_map = 0

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)
        out += self.shortcut(x)
        self.feature_map = F.adaptive_avg_pool2d(out, output_size=1).squeeze()      
        out = F.relu(out, inplace=True)
        return out

    def get_feature_map(self):
        return self.feature_map


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=BasicBlock, num_blocks=[2, 2, 2, 2], deploy_flag=False):
        super(ResNet, self).__init__()
       
        self.in_planes = 64
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1', ConvBN(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3, deploy=deploy_flag, nonlinear=nn.ReLU(inplace=True)))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = self._make_stage(block, 64, num_blocks[0], stride=1, deploy_flag=deploy_flag)
        self.stage2 = self._make_stage(block, 128, num_blocks[1], stride=2, deploy_flag=deploy_flag)
        self.stage3 = self._make_stage(block, 256, num_blocks[2], stride=2, deploy_flag=deploy_flag)
        self.stage4 = self._make_stage(block, 512, num_blocks[3], stride=last_stride, deploy_flag=deploy_flag)

    def _make_stage(self, block, planes, num_blocks, stride, deploy_flag):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            if block is Bottleneck:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride, deploy_flag=deploy_flag))
            else:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride, deploy_flag=deploy_flag))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)     
        out = self.stage2(out)
        out = self.stage3(out)
        f3 = out
        out = self.stage4(out)
        f4 = out
        return [f3, f4], out

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            else:
                model_name = i.replace('base.', '')
                try:
                    self.state_dict()[model_name].copy_(param_dict[i])
                except:
                    continue

def T_ResNet152(last_stride, deploy_flag):
    return ResNet(last_stride=last_stride, block=Bottleneck, num_blocks=[3, 8, 36, 3], deploy_flag=deploy_flag)

def T_ResNet101(last_stride, deploy_flag):
    return ResNet(last_stride=last_stride, block=Bottleneck, num_blocks=[3, 4, 23, 3], deploy_flag=deploy_flag)

def T_ResNet50(last_stride, deploy_flag):
    return ResNet(last_stride=last_stride, block=Bottleneck, num_blocks=[3, 4, 6, 3], deploy_flag=deploy_flag)

def T_ResNet34(last_stride,  deploy_flag):
    return ResNet(last_stride=last_stride, block=BasicBlock, num_blocks=[3, 4, 6, 3], deploy_flag=deploy_flag)

def T_ResNet18(last_stride, deploy_flag):
    return ResNet(last_stride=last_stride, block=BasicBlock, num_blocks=[2, 2, 2, 2], deploy_flag=deploy_flag)