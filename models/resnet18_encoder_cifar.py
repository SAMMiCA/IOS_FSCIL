import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'CIFAR_ResNet', 'CIFAR_ResNet18', 'CIFAR_ResNet34',
           'CIFAR_ResNet10']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, opt1=None, opt2=None):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn1_2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        self.opt1 = opt1
        self.opt2 = opt2

    def forward(self, x):
        if self.opt1 == None and self.opt2 == None:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += shortcut
            return out
        elif self.opt1 == True:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += shortcut
            out = F.relu(out)
            return out
        elif self.opt2 == True:
            # same with supcon_resnet_18
            out = self.conv1(x)
            out = F.relu(self.bn1_2(out))
            #out = self.conv2(F.relu(self.bn2(out)))
            #out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out
        else:
            raise NotImplementedError



class CIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bias=True,opt1=None,opt2=None,opt3=None):
        super(CIFAR_ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.opt1 = opt1
        self.opt2 = opt2
        self.opt3 = opt3
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.opt1, self.opt2))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = self.layer4(out3)

        if self.opt3:
            out = F.relu(out)
        #out = F.avg_pool2d(out, 4)
        #out4 = out.view(out.size(0), -1)
        #out = self.linear(out4)

        return out


def CIFAR_ResNet10(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [1, 1, 1, 1], **kwargs)


def CIFAR_ResNet18(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2, 2, 2, 2], **kwargs)
def CIFAR_ResNet18_1(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2, 2, 2, 2], opt1=True, **kwargs)
def CIFAR_ResNet18_2(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2, 2, 2, 2], opt2=True, **kwargs)
def CIFAR_ResNet18_3(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2, 2, 2, 2], opt3=True, **kwargs)

def CIFAR_ResNet34(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [3, 4, 6, 3], **kwargs)



