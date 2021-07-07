# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
from .resnet_encoder import *

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc   # num_ch_enc=[64, 64, 128, 256, 512]
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()   # 기본 CNN 구조
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):   # Resolution 변경 없음
        last_features = [f[-1] for f in input_features]   # 가장 low-resolution features / Shape : [4, 512, 8, 26]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)   # Shape : [4, 256, 8, 26]

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        # out shape : [4, 6, 8, 26]

        out = out.mean(3).mean(2)   # out shape : [4, 6]

        pose = 0.01 * out.view(-1, 6)

        return pose   # 6-DoF poses 반환


class PoseResNet(nn.Module):

    def __init__(self, num_layers = 18, pretrained = True):
        super(PoseResNet, self).__init__()
        self.encoder = ResnetEncoder(num_layers = num_layers, pretrained = pretrained, num_input_images=2)   # ResNet (다중 이미지 입력)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc)   # num_ch_enc=[64, 64, 128, 256, 512]

    def init_weights(self):
        pass

    def forward(self, img1, img2):
        x = torch.cat([img1,img2],1)   # img1 : Target frames, img2 : Reference frames (두 이웃 frames) concat
        features = self.encoder(x)   # 모든 features (중간 features 포함)를 decoder로 전달
        pose = self.decoder([features])
        return pose

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = PoseResNet().cuda()
    model.train()

    tgt_img = torch.randn(4, 3, 256, 832).cuda()
    ref_imgs = [torch.randn(4, 3, 256, 832).cuda() for i in range(2)]

    pose = model(tgt_img, ref_imgs[0])

    print(pose.size())
