import torch.nn as nn
import torch.nn.functional as F
from model.backbones.convnet_block.convnet_utils import conv_bn, conv_bn_relu, ConvBN
# import numpy as np
import math
import torch
from .convnet_block.conv_transforms import transIII_1x1_kxk, transVII_reduction, transII_addbranch


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, deploy_flag=False, conv1_prune=None, conv3_prune=None):
        super(Bottleneck, self).__init__()
        flag = conv1_prune
        if conv1_prune is None or conv3_prune is None:
            conv1_prune = planes
            conv3_prune = planes
        self.conv1_prune = conv1_prune
        self.conv3_prune = conv3_prune

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_bn(in_planes, self.expansion * planes, kernel_size=1, stride=stride, deploy_flag=deploy_flag)
        else:
            self.shortcut = nn.Identity()

        if deploy_flag:
            self.conv1 = conv_bn_relu(in_planes, conv1_prune, kernel_size=1, padding=0, deploy_flag=deploy_flag)
            self.reparam = nn.Conv2d(in_channels=conv1_prune, out_channels=conv3_prune, kernel_size=3, stride=stride, padding=1, bias=True)
            self.conv3 = conv_bn(conv3_prune, self.expansion * planes, kernel_size=1, deploy_flag=deploy_flag)
        else:
            self.conv1 = conv_bn_relu(in_planes, conv1_prune, kernel_size=1, padding=0, deploy_flag=deploy_flag)
            self.compactor1 = nn.Conv2d(conv1_prune, planes, kernel_size=1, padding=0, bias=False)
            self.conv2 = conv_bn(planes, planes, kernel_size=3, stride=stride, padding=1, deploy_flag=deploy_flag)
            self.compactor2 = nn.Conv2d(planes, conv3_prune, kernel_size=1, padding=0, bias=False)
            self.conv3 = conv_bn(conv3_prune, self.expansion*planes, kernel_size=1, deploy_flag=deploy_flag)
            self.feature_map = 0
            #
            if flag is None:
                identity_mat = torch.eye(planes)
                self.compactor1.weight.data.copy_(identity_mat.view((planes, planes, 1, 1)))
                self.compactor2.weight.data.copy_(identity_mat.view((planes, planes, 1, 1)))


    def forward(self, x):
        if hasattr(self, 'reparam'):
            out = self.conv1(x)
            out = self.reparam(out)
            out = F.relu(out)
            out = self.conv3(out)
            out += self.shortcut(x)
            out = F.relu(out, inplace=True)
            return out
        else:
            out = self.conv1(x)
            out = self.compactor1(out)
            out = self.conv2(out)
            out = self.compactor2(out)
            out = F.relu(out)
            out = self.conv3(out)
            out += self.shortcut(x)
            self.feature_map = F.adaptive_avg_pool2d(out, output_size=1).squeeze()
            out = F.relu(out, inplace=True)
            return out

    def get_compactor1_kernel_detach(self):
        return self.compactor1.weight

    def get_compactor2_kernel_detach(self):
        return self.compactor2.weight

    def get_lasso_vetor(self):
        lasso_vector_conv1 = torch.sum(torch.sqrt(torch.sum(self.get_compactor1_kernel_detach()**2, dim=(0, 2, 3))))
        lasso_vector_conv2 = torch.sum(torch.sqrt(torch.sum(self.get_compactor2_kernel_detach()**2, dim=(1, 2, 3))))
        return lasso_vector_conv1, lasso_vector_conv2

    def get_feature_map(self):
        return self.feature_map

    def get_equivalent_kernel_bias(self):
        if hasattr(self.shortcut, 'switch_to_deploy'):
            self.shortcut.switch_to_deploy()
        if hasattr(self.conv1, 'switch_to_deploy'):
            self.conv1.switch_to_deploy()
        if hasattr(self.conv2, 'switch_to_deploy'):
            self.conv2.switch_to_deploy()
        if hasattr(self.conv3, 'switch_to_deploy'):
            self.conv3.switch_to_deploy()

       
    
        c1_weight, c1_bias = self.compactor1.weight.data, 0
        k2_weight, k2_bias = self.conv2.conv.weight.data, self.conv2.conv.bias.data
        c3_weight, c3_bias = self.compactor2.weight.data, 0

        k_weight, k_bias = transIII_1x1_kxk(c1_weight, c1_bias, k2_weight, k2_bias)
        k_weight, k_bias = transVII_reduction(k_weight, k_bias, c3_weight, c3_bias)

        return k_weight, k_bias

    def switch_to_deploy(self):
        if hasattr(self, 'reparam'):
            return
        else:
            kernel, bias = self.get_equivalent_kernel_bias()
            self.reparam = nn.Conv2d(in_channels=self.conv2.conv.in_channels, out_channels=self.conv2.conv.out_channels,
                                         kernel_size=self.conv2.conv.kernel_size, stride=self.conv2.conv.stride,
                                         padding=self.conv2.conv.padding, bias=True)

            self.reparam.weight.data = kernel
            self.reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv2')
        self.__delattr__('compactor1')
        self.__delattr__('compactor2')




class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, num_blocks=[3, 4, 6, 3], deploy_flag=False, prune_depth=None):
        super(ResNet, self).__init__()
        if prune_depth is not None:
            self.prune_depth = prune_depth
            self.count = 0
        else:
            self.prune_depth = None
            self.count = 0
        self.in_planes = 64
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1', ConvBN(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3, deploy=deploy_flag, nonlinear=nn.ReLU(inplace=True)))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = self._make_stage(block, 64, num_blocks[0], stride=1,  deploy_flag=deploy_flag)
        self.stage2 = self._make_stage(block, 128, num_blocks[1], stride=2, deploy_flag=deploy_flag)
        self.stage3 = self._make_stage(block, 256, num_blocks[2], stride=2,  deploy_flag=deploy_flag)
        self.stage4 = self._make_stage(block, 512, num_blocks[3], stride=last_stride, deploy_flag=deploy_flag)

    def _make_stage(self, block, planes, num_blocks, stride, deploy_flag):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            if self.prune_depth is not None:
                conv1_prune = self.prune_depth[self.count]
                conv3_prune = self.prune_depth[self.count + 1]
            else:
                conv1_prune = None
                conv3_prune = None

            if block is Bottleneck:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride, deploy_flag=deploy_flag, conv1_prune=conv1_prune, conv3_prune=conv3_prune))
            else:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride, deploy_flag=deploy_flag, conv1_prune=conv1_prune, conv3_prune=conv3_prune))
            self.in_planes = int(planes * block.expansion)
            self.count = self.count + 2
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
            if 'fc' in i:
                continue
            if 'layer' not in i:
                model_name = 'stage0.conv1.' + i.replace('1', '')
            elif 'layer' in i:
                new_i = i.split('.')
                layer_num = new_i[0][-1]
                if 'downsample' in i:
                    if new_i[3] == '0':
                        model_name = 'stage' + layer_num + '.' + new_i[1] + '.shortcut.conv.' + new_i[-1]
                    else:
                        model_name = 'stage' + layer_num + '.' + new_i[1] + '.shortcut.bn.' + new_i[-1]
                else:
                    if 'conv' in i:
                        model_name = 'stage' + layer_num + '.' + new_i[1] + '.' + new_i[2] + '.conv.' + new_i[3]
                    else:
                        model_name = 'stage' + layer_num + '.' + new_i[1] + '.' + new_i[2].replace('bn', 'conv') + '.bn.' + new_i[3]   
            self.state_dict()[model_name].copy_(param_dict[i])

def S_ResNet152(last_stride, deploy_flag, prune_depth):
    return ResNet(last_stride=last_stride, block=Bottleneck, num_blocks=[3, 8, 36, 3], deploy_flag=deploy_flag, prune_depth=prune_depth)

def S_ResNet101(last_stride, deploy_flag, prune_depth):
    return ResNet(last_stride=last_stride, block=Bottleneck, num_blocks=[3, 4, 23, 3], deploy_flag=deploy_flag, prune_depth=prune_depth)

def S_ResNet50(last_stride, deploy_flag, prune_depth):
    return ResNet(last_stride=last_stride, block=Bottleneck, num_blocks=[3, 4, 6, 3], deploy_flag=deploy_flag, prune_depth=prune_depth)