import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


# resnet reference: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py

class PixelNorm3d(nn.Module):
    # noinspection PyUnusedLocal
    def __init__(self, num_features, eps=1E-8, *args, **kwargs):
        super().__init__()

        # Too few dimensions
        assert num_features > 1, 'Too few dimensions to perform PixelNorm!'

        # Copy over
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        l2_norm = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)  # [N, 1, S, S, S]
        return torch.div(x, l2_norm)


class VarNorm3d(nn.Module):

    def __init__(self, num_features, norm_type, *args, **kwargs):
        super().__init__()

        assert norm_type in ['BN', 'PN', 'NO', 'IN']

        self.num_features = num_features

        if norm_type == 'BN':
            self.layer = nn.BatchNorm3d(num_features, **kwargs)
        elif norm_type == 'PN':
            self.layer = PixelNorm3d(num_features, **kwargs)
        elif norm_type == 'IN':
            self.layer = nn.InstanceNorm3d(num_features, **kwargs)
        elif norm_type == 'NO':
            self.layer = None

        self.norm_type = norm_type

    def forward(self, x):
        if self.norm_type == 'NO':
            return x
        else:
            return self.layer(x)


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm_type, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = VarNorm3d(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = VarNorm3d(planes, norm_type)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, norm_type, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = VarNorm3d(planes, norm_type)
        self.bn1 = VarNorm3d(planes, norm_type)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = VarNorm3d(planes, norm_type)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = VarNorm3d(planes * self.expansion, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 norm_type='BN',
                 widen_factor=1.0,
                 n_classes=400,
                 use_layer=[1, 1, 1, 1],
                 strides=[1, 2, 2, 2],
                 use_position=False):
        super().__init__()

        self.norm_type = norm_type

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)

        self.bn1 = VarNorm3d(self.in_planes, self.norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.use_layer = use_layer

        if self.use_layer == 1:
            self.layer1 = self._make_layer(block,
                                           block_inplanes[0],
                                           layers[0],
                                           shortcut_type,
                                           stride=1)
        if self.use_layer == 2:
            self.layer2 = self._make_layer(block,
                                           block_inplanes[1],
                                           layers[1],
                                           shortcut_type,
                                           stride=1)

        if self.use_layer == 3:
            self.layer3 = self._make_layer(block,
                                           block_inplanes[2],
                                           layers[2],
                                           shortcut_type,
                                           stride=1)

        if self.use_layer == 4:
            self.layer4 = self._make_layer(block,
                                           block_inplanes[3],
                                           layers[3],
                                           shortcut_type,
                                           stride=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # include patch position ?
        self.use_position = use_position
        add_pos_features = 3 if self.use_position else 0
        self.fc = nn.Linear(block_inplanes[self.use_layer - 1] * block.expansion + add_pos_features, n_classes)

        # grad cam
        self.gradients = None

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, VarNorm3d):
                if isinstance(m.layer, nn.BatchNorm3d):
                    nn.init.constant_(m.layer.weight, 1)
                    nn.init.constant_(m.layer.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    VarNorm3d(planes * block.expansion, self.norm_type))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  norm_type=self.norm_type))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x, pos=None, hook=False):
        # extract features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        if self.use_layer == 1:
            x = self.layer1(x)
        if self.use_layer == 2:
            x = self.layer2(x)
        if self.use_layer == 3:
            x = self.layer3(x)
        if self.use_layer == 4:
            x = self.layer4(x)
        # register the hook
        if hook:
            x.register_hook(self.activations_hook)
        # pooling and flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # if given, include slice position
        if self.use_position:
            x = torch.cat([x, pos.flatten(start_dim=1)], dim=1)
        # fully connected part
        x = self.fc(x)
        return x

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        if self.use_layer == 1:
            x = self.layer1(x)
        if self.use_layer == 2:
            x = self.layer2(x)
        if self.use_layer == 3:
            x = self.layer3(x)
        if self.use_layer == 4:
            x = self.layer4(x)
        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
