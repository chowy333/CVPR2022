# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):   # 다중 이미지 입력용 ResNet
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(   # 다중 이미지 입력일 경우, concat하여 입력
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 여기까지 ResNet의 1st layer와 완전히 동일 (concat 부분 제외)
        self.layer1 = self._make_layer(block, 64, layers[0])   # ResNet18 : [2, 2, 2, 2] / ResNet50 : [3, 4, 6, 3]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():   # Weights 초기화
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):   # 다중 이미지 입력용 ResNet 구성하기
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"   # ResNet18 / ResNet50 선정
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)   # ResNet18 : (BasicBlock, [2, 2, 2, 2], n)
                                                                                           # ResNet50 : (Bottleneck, [3, 4, 6, 3], n)

    if pretrained:   # Pre-trained weights 공식 repo에서 불러오기
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(   # 다중 이미지 입력이므로, 첫 conv layer weight 재설정 : weights concat 후, 입력 이미지 수로 나누기
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)   # Weights 탑재하기
    return model


class ResnetEncoder(nn.Module):   # ResNet (다중 / 단일 이미지 입력)
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:   # 다중 이미지 입력일 경우 (Reference frames 전용)
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:   # 단일 이미지 입력일 경우 (Target frame 전용)
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):   # ResNet inference, 다중 이미지 입력인 경우, conv1에서 concat함
        self.features = []
        x = input_image   # Shape : [4, 3, 256, 832] (Target frame) or [4, 6, 256, 832] (Reference frames) / [Batch, RGB, H, W]
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))   # Shape : [4, 64, 128, 416] (self.features[0])
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))   # Shape : [4, 64, 64, 208] (self.features[1])
        self.features.append(self.encoder.layer2(self.features[-1]))   # Shape : [4, 128, 32, 104] (self.features[2])
        self.features.append(self.encoder.layer3(self.features[-1]))   # Shape : [4, 256, 16, 52] (self.features[3])
        self.features.append(self.encoder.layer4(self.features[-1]))   # Shape : [4, 512, 8, 26] (self.features[4])

        return self.features   # 모든 features (중간 features 포함)를 decoder로 전달
