'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, size, bn_type):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False) # (in_planes, H, W) -> (4*growth_rate, H, W)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False) # (4*growth_rate, H, W) -> (growth_rate, H, W)
        if bn_type == "bn":
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(4*growth_rate)
        elif bn_type == "ln":
            self.bn1 = nn.LayerNorm([in_planes, size, size])
            self.bn2 = nn.LayerNorm([4*growth_rate, size, size])
        else:
            raise ValueError(f"unknown bn_type: {bn_type}")

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, size, bn_type):
        super(Transition, self).__init__()
        if bn_type == "bn":
            self.bn = nn.BatchNorm2d(in_planes)
        elif bn_type == "ln":
            self.bn = nn.LayerNorm([in_planes, size, size])
        else:
            raise ValueError(f"unknown bn_type: {bn_type}")
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False) # (in_planes, H, W) -> (out_planes, H, W)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2) # (out_planes, H, W) -> (out_planes, H/2, W/2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, bn_type="bn", reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.size = 32
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate # 24
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False) # (N, 24, 32, 32)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], self.size, bn_type)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, self.size, bn_type)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], self.size//2, bn_type)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, self.size//2, bn_type)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], self.size//4, bn_type)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, self.size//4, bn_type)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], self.size//8, bn_type)
        num_planes += nblocks[3]*growth_rate

        if bn_type == "bn":
            self.bn = nn.BatchNorm2d(num_planes)
        elif bn_type == "ln":
            self.bn = nn.LayerNorm([num_planes, self.size//8, self.size//8])
        else:
            raise NotImplementedError
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, size, bn_type):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, size, bn_type))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x) # (N, 24, 32, 32)
        out = self.trans1(self.dense1(out)) # (N, 48, 16, 16)
        out = self.trans2(self.dense2(out)) # (N, 96, 8, 8) 
        out = self.trans3(self.dense3(out)) # (N, 192, 4, 4)
        out = self.dense4(out) # (N, 384, 4, 4) 
        out = F.avg_pool2d(F.relu(self.bn(out)), 4) # (N, 384, 1, 1)
        out = out.view(out.size(0), -1) # (N, 384)
        out = self.linear(out) # (N, num_classes)
        return out

def DenseNet121(bn_type: str='bn'):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, bn_type=bn_type)

def DenseNet169(bn_type: str='bn'):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, bn_type=bn_type)

# def DenseNet201():
#     return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

# def DenseNet161():
#     return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

# def densenet_cifar(bn_type: str='bn'):
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, bn_type=bn_type)

def test():
    net = DenseNet121()
    print(net)
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

test()