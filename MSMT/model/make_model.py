import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .backbones.build_backbone import build_backbone
import torch.nn.init as init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, num_bottleneck=256):
        super(ClassBlock, self).__init__()
        add_block1 = [nn.BatchNorm1d(input_dim), nn.ReLU(inplace=True),
                      nn.Linear(input_dim, num_bottleneck, bias=False),
                      nn.BatchNorm1d(num_bottleneck)]
        add_block = nn.Sequential(*add_block1)
        add_block.apply(weights_init_kaiming)

        classifier = nn.Linear(num_bottleneck, class_num, bias=False)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        y = self.classifier(x)
        return x, y


class Single_Teacher(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Single_Teacher, self).__init__()
        self.num_classes = num_classes
        print('teacher model backbone:', end='')
        self.in_planes, self.base = build_backbone(cfg.MODEL.TEACHER_NAME, cfg.MODEL.LAST_STRIDE, cfg, deploy_flag=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassBlock(self.in_planes, self.num_classes, num_bottleneck=512)

    def forward(self, x):
        feat, x = self.base(x)
        global_feat = self.gap(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        _, y = self.classifier(global_feat)
        return feat, y


class Teacher_model(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Teacher_model, self).__init__()
        self.Teacher_model = Single_Teacher(num_classes, cfg)
        self.Teacher_model.load_state_dict({k.replace('module.', ""):
                                                v for k, v in torch.load(cfg.MODEL.TEACHER_PATH).items()})
        print('Loading pretrained ImageNet model......from {}'.format(cfg.MODEL.TEACHER_PATH))

    def forward(self, x):
        feat, y = self.Teacher_model(x)
        return feat, y

class Backbone(nn.Module):
    def __init__(self, num_classes, cfg, deploy_flag, prune_depth=None):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE

        self.num_classes = num_classes
        print('student model backbone:', end='')
        self.in_planes, self.base = build_backbone(model_name, last_stride, cfg, deploy_flag, prune_depth=prune_depth)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassBlock(self.in_planes, self.num_classes, num_bottleneck=512)

        if deploy_flag == False and prune_depth is None:
            if pretrain_choice == 'imagenet':
                self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
                print('Loading pretrained ImageNet model......from {}'.format(cfg.MODEL.PRETRAIN_PATH))

            self.Teacher = Teacher_model(self.num_classes, cfg)
            for p in self.Teacher.parameters():
                p.requires_grad = False

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        stu_feat, stu_x = self.base(x)
        global_feat = self.gap(stu_x)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat, y = self.classifier(global_feat)

        if self.training:
            teacher_feat, teacher_y = self.Teacher(x)
            compactor_lasso_list = []
            s_map = []
            t_map = []
            s_mask_map = []
            t_mask_map = []
            for m in self.base.modules():
                if hasattr(m, 'get_lasso_vetor'):
                    compactor_lasso = m.get_lasso_vetor()
                    compactor_lasso_list.append(compactor_lasso)

                if hasattr(m, 'get_feature_map'):
                    smap, s_mask = m.get_feature_map()
                    s_map.append(smap)
                    s_mask_map.append(s_mask)

            for m in self.Teacher.Teacher_model.base.modules():
                if hasattr(m, 'get_feature_map'):
                    tmap, t_mask = m.get_feature_map()
                    t_map.append(tmap)
                    t_mask_map.append(t_mask)

            return [y, teacher_y], global_feat, compactor_lasso_list, [[s_mask_map, t_mask_map], [s_map, t_map]]
        else:
            return feat

    def load_param_test(self, trained_path):
        param_dict = {k.replace('module.', ""): v for k, v in torch.load(trained_path).items()}
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def fc_load_parm(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'classifier' in i:
                self.state_dict()[i].copy_(param_dict[i])


def make_model(cfg, num_class, deploy_flag=False, prune=None):
    model = Backbone(num_class, cfg, deploy_flag, prune_depth=prune)
    return model



if __name__ == '__main__':
    # debug model structure
    import torchvision.models as models
    from config import cfg
    import torch
    from ptflops import get_model_complexity_info
    from backbones.build_backbone import build_backbone
    import time
    import numpy as np
    import os

    with torch.cuda.device(0):
        x = torch.randn(2, 3, 256, 256)
        net = Backbone(576, cfg, deploy_flag=False)
        #net.load_param_test('/home/xieyi/SSD/xieyi/Vehicle_Reid/KD/KD_Prune/LKD_Rep_ResNet_2/0.003_0.7_10/s_resnet50_original_90.pth')
        print("--------------")
        net.eval()
        a = time.time()
        result1 = net(x)
        print(time.time() - a)
      
        for m in net.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        print(net)
        result2 = net(x)
        print(((result2 - result1) ** 2).sum())
    