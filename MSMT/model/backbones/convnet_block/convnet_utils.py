import torch.nn as nn
from model.backbones.convnet_block.conv_transforms import transI_fusebn


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                             stride, padding, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        if hasattr(self, 'bn'):
            kernel, bias = transI_fusebn(self.conv.weight, self.bn)
            conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                          stride=self.conv.stride, padding=self.conv.padding, bias=True)
            conv.weight.data = kernel
            conv.bias.data = bias
            for para in self.parameters():
                para.detach_()
            self.__delattr__('conv')
            self.__delattr__('bn')
            self.conv = conv
        else:
            return


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, deploy_flag=False):
        block_type = ConvBN
        return block_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, deploy=deploy_flag)


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, deploy_flag=False):
    block_type = ConvBN
    return block_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, deploy=deploy_flag, nonlinear=nn.ReLU(inplace=True))




