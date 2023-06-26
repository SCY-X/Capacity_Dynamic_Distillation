from .resnet import ResNet, BasicBlock, Bottleneck
from .teacher_resnet import T_ResNet18, T_ResNet34, T_ResNet50, T_ResNet101, T_ResNet152
from .student_resnet import S_ResNet50, S_ResNet101, S_ResNet152
from .erhc_teacher_resnet import ERHC_T_ResNet18, ERHC_T_ResNet34, ERHC_T_ResNet50, ERHC_T_ResNet101, ERHC_T_ResNet152
from .erhc_student_resnet import ERHC_S_ResNet50, ERHC_S_ResNet101, ERHC_S_ResNet152
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .se_resnet_ibn_a import se_resnet101_ibn_a
from .resnet_ibn_b import resnet101_ibn_b


def build_backbone(model_name, last_stride, cfg, deploy_flag, prune_depth=None):
    if model_name == 'resnet18':
        in_planes = 512
        base = ResNet(last_stride=last_stride, block=BasicBlock, frozen_stages=cfg.MODEL.FROZEN,
                           layers=[2, 2, 2, 2])
        print('using resnet18 as a backbone')
    elif model_name == 'resnet34':
        in_planes = 512
        base = ResNet(last_stride=last_stride, block=BasicBlock, frozen_stages=cfg.MODEL.FROZEN,
                           layers=[3, 4, 6, 3])
        print('using resnet34 as a backbone')
    elif model_name == 'resnet50':
        in_planes = 2048
        base = ResNet(last_stride=last_stride,
                           block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                           layers=[3, 4, 6, 3])
        print('using resnet50 as a backbone')

    elif model_name == 't_resnet18':
        in_planes = 512
        base = T_ResNet18(last_stride, deploy_flag)
        print('using resnet18 as a teacher backbone')

    elif model_name == 't_resnet34':
        in_planes = 512
        base = T_ResNet34(last_stride, deploy_flag)
        print('using resnet34 as a teacher backbone')

    elif model_name == 't_resnet50':
        in_planes = 2048
        base = T_ResNet50(last_stride, deploy_flag)
        print('using resnet50 as a teacher backbone')
    
    elif model_name == 't_resnet101':
        in_planes = 2048
        base = T_ResNet101(last_stride, deploy_flag)
        print('using resnet101 as a teacher backbone')
        
    elif model_name == 't_resnet152':
        in_planes = 2048
        base = T_ResNet152(last_stride, deploy_flag)
        print('using resnet152 as a teacher backbone')

    elif model_name == 's_resnet50':
        in_planes = 2048
        base = S_ResNet50(last_stride, deploy_flag, prune_depth)
        print('using resnet50 as a student backbone')


    elif model_name == 's_resnet101':
        in_planes = 2048
        base = S_ResNet101(last_stride, deploy_flag, prune_depth)
        print('using resnet101 as a student backbone')

    elif model_name == 's_resnet152':
        in_planes = 2048
        base = S_ResNet152(last_stride, deploy_flag, prune_depth)
        print('using resnet152 as a student backbone')


    elif model_name == 'erhc_t_resnet50':
        in_planes = 2048
        base = ERHC_T_ResNet50(last_stride, deploy_flag)

    elif model_name == 'erhc_t_resnet101':
        in_planes = 2048
        base = ERHC_T_ResNet101(last_stride, deploy_flag)


    elif model_name == 'erhc_s_resnet50':
        in_planes = 2048
        base = ERHC_S_ResNet50(last_stride, deploy_flag, prune_depth)

    elif model_name == 'erhc_s_resnet101':
        in_planes = 2048
        base = ERHC_S_ResNet101(last_stride, deploy_flag, prune_depth)

    elif model_name == 'resnet50_ibn_a':
        in_planes = 2048
        base = resnet50_ibn_a(last_stride)
        print('using resnet50_ibn_a as a backbone')
    elif model_name == 'resnet101_ibn_a':
        in_planes = 2048
        base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
        print('using resnet101_ibn_a as a backbone')
    elif model_name == 'se_resnet101_ibn_a':
        in_planes = 2048
        base = se_resnet101_ibn_a(last_stride)
        print('using se_resnet101_ibn_a as a backbone')
    elif model_name == 'resnet101_ibn_b':
        in_planes = 2048
        base = resnet101_ibn_b(last_stride)
        print('using resnet101_ibn_b as a backbone')
    else:
        print('unsupported backbone! but got {}'.format(model_name))

    return in_planes, base