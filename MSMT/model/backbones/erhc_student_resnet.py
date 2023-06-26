import torch.nn as nn
import torch.nn.functional as F
from model.backbones.convnet_block.convnet_utils import conv_bn, conv_bn_relu, ConvBN
# import numpy as np
import math
import torch
from .convnet_block.conv_transforms import transVII_reduction
import torch.nn.init as init

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, deploy_flag=False, compactor_prune=None):
        super(Bottleneck, self).__init__()
        flag = compactor_prune
        if compactor_prune is None:
            compactor_prune = planes
        self.compactor_prune = compactor_prune

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_bn(in_planes, self.expansion * planes, kernel_size=1, stride=stride, deploy_flag=deploy_flag)
        else:
            self.shortcut = nn.Identity()

        if deploy_flag:
            self.conv1 = conv_bn_relu(in_planes, planes, kernel_size=1, padding=0, deploy_flag=deploy_flag)
            self.reparam = nn.Conv2d(in_channels=planes, out_channels=self.compactor_prune, kernel_size=3, stride=stride, padding=1, bias=True)
            self.conv3 = conv_bn(self.compactor_prune, self.expansion * planes, kernel_size=1, deploy_flag=deploy_flag)
        else:
            self.conv1 = conv_bn_relu(in_planes, planes, kernel_size=1, padding=0, deploy_flag=deploy_flag)
            self.conv2 = conv_bn(planes, planes, kernel_size=3, stride=stride, padding=1, deploy_flag=deploy_flag)
            self.compactor = nn.Conv2d(planes, self.compactor_prune, kernel_size=1, padding=0, bias=False)
            self.conv3 = conv_bn(self.compactor_prune, self.expansion*planes, kernel_size=1, deploy_flag=deploy_flag)


            self.feature_map = 0
            self.mask_map = 0
            #
            if flag is None:
                self.register_buffer('compactor_mask', torch.ones(self.compactor_prune))
                init.ones_(self.compactor_mask)

                identity_mat = torch.eye(planes)
                self.compactor.weight.data.copy_(identity_mat.view((planes, planes, 1, 1)))


    def forward(self, x):
        if hasattr(self, 'reparam'):
            out = self.conv1(x)
            out = self.reparam(out)
            out = F.relu(out, inplace=True)
            out = self.conv3(out)
            out += self.shortcut(x)
            out = F.relu(out, inplace=True)
            return out
        else:
            out = self.conv1(x)
            out = self.conv2(out)
            self.feature_map = F.adaptive_avg_pool2d(out, output_size=1).squeeze()
            out = self.compactor(out)
            self.mask_map = F.adaptive_avg_pool2d(out, output_size=1).squeeze()
            out = F.relu(out)
            out = self.conv3(out)
            out += self.shortcut(x)
            out = F.relu(out, inplace=True)
            return out

    def get_compactor_kernel_detach(self):
        return self.compactor.weight

    def get_lasso_vetor(self):
        lasso_vector_compactor = torch.sum(torch.sqrt(torch.sum(self.get_compactor_kernel_detach()**2, dim=(1, 2, 3))))
        return lasso_vector_compactor

    def get_feature_map(self):
        return self.feature_map, self.mask_map

    def get_equivalent_kernel_bias(self):
        if hasattr(self.shortcut, 'switch_to_deploy'):
            self.shortcut.switch_to_deploy()
        if hasattr(self.conv1, 'switch_to_deploy'):
            self.conv1.switch_to_deploy()
        if hasattr(self.conv2, 'switch_to_deploy'):
            self.conv2.switch_to_deploy()
        if hasattr(self.conv3, 'switch_to_deploy'):
            self.conv3.switch_to_deploy()

        k2_weight, k2_bias = self.conv2.conv.weight.data, self.conv2.conv.bias.data

        c3_weight, c3_bias = self.compactor.weight.data, 0
        k_weight, k_bias = transVII_reduction(k2_weight, k2_bias, c3_weight, c3_bias)

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
        self.__delattr__('compactor')
        #self.__delattr__('compactor_mask')


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, num_blocks=[3, 4, 6, 3], deploy_flag=False, prune_depth=None):
        super(ResNet, self).__init__()
        if prune_depth is not None:
            self.prune_depth = prune_depth
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
                compactor_prune = self.prune_depth[self.count]
            else:
                compactor_prune = None

            if block is Bottleneck:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride, deploy_flag=deploy_flag, compactor_prune=compactor_prune))
            else:
                blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride, deploy_flag=deploy_flag, compactor_prune=compactor_prune))
            self.in_planes = int(planes * block.expansion)
            self.count = self.count + 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        f1 = out
        out = self.stage2(out)
        f2 = out
        out = self.stage3(out)
        f3 = out
        out = self.stage4(out)
        f4 = out
        return [f1, f2, f3, f4], out

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
                        model_name = 'stage' + layer_num + '.' + new_i[1] + '.' + new_i[2].replace('bn', 'conv') + '.bn.' + \
                                     new_i[3]
            self.state_dict()[model_name].copy_(param_dict[i])


def ERHC_S_ResNet152(last_stride, deploy_flag, prune_depth):
    return ResNet(last_stride=last_stride, block=Bottleneck, num_blocks=[3, 8, 36, 3], deploy_flag=deploy_flag, prune_depth=prune_depth)

def ERHC_S_ResNet101(last_stride, deploy_flag, prune_depth):
    return ResNet(last_stride=last_stride, block=Bottleneck, num_blocks=[3, 4, 23, 3], deploy_flag=deploy_flag, prune_depth=prune_depth)

def ERHC_S_ResNet50(last_stride, deploy_flag, prune_depth):
    return ResNet(last_stride=last_stride, block=Bottleneck, num_blocks=[3, 4, 6, 3], deploy_flag=deploy_flag, prune_depth=prune_depth)