from __future__ import absolute_import, division, print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_encoder import *


import numpy as np
from collections import OrderedDict

class ConvBlock(nn.Module):   # DepthDecoder 전용 conv block
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)   # ELU 사용

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):   # Conv block 전용 conv3x3
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):   # Upsampling (Nearest-neighbor interpolation)
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc   # num_ch_enc=[64, 64, 128, 256, 512]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):   # 각 scale에 맞는 conv block 생성
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = []

        # decoder
        x = input_features[-1]   # 가장 low-resolution features / Shape : [4, 512, 8, 26]
        for i in range(4, -1, -1):   # Low to high
            x = self.convs[("upconv", i, 0)](x)   # 3x3 conv
            x = [upsample(x)]   # Nearest-neighbor upsampling
            if self.use_skips and i > 0:   # 한 단계 위 level features와 summation
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)   # 3x3 conv
            if i in self.scales:   # alpha * sigmoid + beta
                self.outputs.append(self.alpha * self.sigmoid(self.convs[("dispconv", i)](x)) + self.beta)
                # self.outputs[0] shape : [4, 1, 32, 104]
                # self.outputs[1] shape : [4, 1, 64, 208]
                # self.outputs[2] shape : [4, 1, 128, 416]
                # self.outputs[3] shape : [4, 1, 256, 832]

        self.outputs = self.outputs[::-1]
        return self.outputs   # 모든 scale의 depth maps (중간 depth maps 포함) 반환
                              # Training시에 모든 scale에 대하여 loss를 구하기 때문임


class DispResNet(nn.Module):

    def __init__(self, num_layers = 18, pretrained = True):
        super(DispResNet, self).__init__()
        self.encoder = ResnetEncoder(num_layers = num_layers, pretrained = pretrained, num_input_images=1)
        # ResNet (다중 / 단일 이미지 입력), num_input_images=1일 경우, 일반 ResNet
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)   # num_ch_enc=[64, 64, 128, 256, 512]

    def init_weights(self):
        pass

    def forward(self, x):
        features = self.encoder(x)   # 모든 features (중간 features 포함)를 decoder로 전달
        outputs = self.decoder(features)   # 모든 scale의 depth maps (중간 depth maps 포함) 반환

        if self.training:
            return outputs   # Training시 모든 scale의 depth maps (중간 depth maps 포함) 반환
        else:
            return outputs[0]   # Test시 가장 (최종) high-resolution depth map만 반환


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = DispResNet().cuda()
    model.train()

    B = 12

    tgt_img = torch.randn(B, 3, 256, 832).cuda()
    ref_imgs = [torch.randn(B, 3, 256, 832).cuda() for i in range(2)]

    tgt_depth = model(tgt_img)

    print(tgt_depth[0].size())
