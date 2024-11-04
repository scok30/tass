import copy
import math
from itertools import chain

import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget

from MyGradCAM import MyGradCAM
from ResNet import ResNet


class joint_network(nn.Module):
    def __init__(self, numclass, feature_extractor: ResNet):
        super(joint_network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(512, numclass * 4, bias=True)
        self.classifier = nn.Linear(512, numclass, bias=True)
        # Decoder
        self.downsample = 8
        self.fc_pixel_sal = nn.Conv2d(512 * self.feature.expansion, self.downsample ** 2, 1)
        self.fc_pixel_edge = nn.Conv2d(512 * self.feature.expansion, self.downsample ** 2, 1)
        # Grad-CAM
        self.grad_cam = MyGradCAM(model=nn.Sequential(self.feature, self.classifier),
                                  target_layers=[self.feature.layer2[-1], self.feature.layer3[-1],
                                                 self.feature.layer4[-1]])  # 这个只能创建一个，不然调用一次创建一次会缓存过多导致训练速度下降！

    def forward(self, input):
        x = self.feature(input)
        x = self.classifier(x)
        return x

    def forward_sal(self, x, targets=None, noise=False):
        if targets is not None:
            targets = [ClassifierOutputTarget(category) for category in targets]

        # noise_inject_mode
        self.feature.noise = noise
        with self.grad_cam as cam:
            cam_maps = cam(input_tensor=x, targets=targets)[::4]
            # 获取 layer4 的 feature-map
            x = cam.activations_and_grads.activations[-1]
            outputs = cam.outputs
        # noise_inject_mode OFF
        self.feature.noise = False

        # decoder
        def convert(module, x):
            x = F.sigmoid(module(x))
            spatial_size = x.size()[-2:]
            x = x.view(x.size(0), self.downsample, self.downsample, *x.size()[-2:]).permute(0, 1, 3, 2, 4)
            x = x.contiguous().view(x.size(0), self.downsample * spatial_size[0], self.downsample * spatial_size[1])
            return x

        x_sal = convert(self.fc_pixel_sal, x[::4])
        x_edge = convert(self.fc_pixel_edge, x[::4])
        # x -> f
        dim = x.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)
        x = pool(x)
        x = x.view(x.size(0), -1)

        return x_sal, x_edge, cam_maps, x, outputs

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass * 4, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_feature = self.classifier.in_features
        out_feature = self.classifier.out_features

        self.classifier = nn.Linear(in_feature, numclass, bias=True)
        self.classifier.weight.data[:out_feature] = weight[:out_feature]
        self.classifier.bias.data[:out_feature] = bias[:out_feature]

        del self.grad_cam
        self.grad_cam = MyGradCAM(model=nn.Sequential(self.feature, self.classifier),
                                  target_layers=[self.feature.layer2[-1], self.feature.layer3[-1],
                                                 self.feature.layer4[-1]])

    def feature_extractor(self, inputs):
        return self.feature(inputs)