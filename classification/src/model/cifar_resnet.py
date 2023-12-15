'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, size, bn_type, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if bn_type == "bn":
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        elif bn_type == "ln":
            assert stride == 1 or stride == 2, "LayerNorm can only be used with stride 1 or 2"
            self.bn1 = nn.LayerNorm([planes, size // stride, size // stride])
            self.bn2 = nn.LayerNorm([planes, size // stride, size // stride])
        else:
            raise ValueError(f"{bn_type} is not supported for basic block.")

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if bn_type == "bn":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            elif bn_type == "ln":
                assert stride == 1 or stride == 2, "LayerNorm can only be used with stride 1 or 2"
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.LayerNorm([self.expansion*planes, size // stride, size // stride])
                )
            else:
                raise ValueError(f"{bn_type} is not supported for basic block.")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# not for current usage
# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bn_type="bn"):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.size = 32

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False) # (64, 32, 32)
        if bn_type == "bn":
            self.bn1 = nn.BatchNorm2d(64)
        elif bn_type == "ln":
            self.bn1 = nn.LayerNorm([64, self.size, self.size])
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, size=self.size, bn_type=bn_type)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, size=self.size, bn_type=bn_type)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, size=self.size//2, bn_type=bn_type)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, size=self.size//4, bn_type=bn_type)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, size, bn_type):
        strides = [stride] + [1]*(num_blocks-1)
        sizes = [size] + [size // stride]*(num_blocks-1)
        layers = []
        for stride, size in zip(strides, sizes):
            layers.append(block(self.in_planes, planes, size, bn_type, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes: int=10, bn_type: str="bn"):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, bn_type)

# not for current usage
def ResNet34(num_classes: int=10, bn_type: str="bn"):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, bn_type)


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18(10, "ln")
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

test()