import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import chain

from Cifar100.ResNet import resnet18_cbam, ResNet


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=2*in_channels, out_channels=in_channels, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)    # 在通道维度拼接 (B * C * H * W)
        assert 2*self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class joint_network_dual(nn.Module):
    def __init__(self, numclass, numsuperclass, feature_extractor: ResNet):
        super(joint_network_dual, self).__init__()
        # Slow Learner
        self.feature = feature_extractor
        self.classifier_coarse = nn.Linear(512, numsuperclass, bias=True)

        # Fast Learner (CIL)
        self.f_conv1 = self._make_conv2d_layer(3, 64, padding=1, max_pool=False)
        self.fusion_blocks1 = FeatureFusionModule(in_channels=64)
        self.f_conv2 = self._make_conv2d_layer(64, 128, padding=1, max_pool=True)
        self.fusion_blocks2 = FeatureFusionModule(in_channels=128)
        self.f_conv3 = self._make_conv2d_layer(128, 256, padding=1, max_pool=True)
        self.fusion_blocks3 = FeatureFusionModule(in_channels=256)
        self.f_conv4 = self._make_conv2d_layer(256, 512, padding=1, max_pool=True)
        self.fusion_blocks4 = FeatureFusionModule(in_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, numclass * 4, bias=True)
        self.classifier = nn.Linear(512, numclass, bias=True)

        # Decoder
        self.downsample = 8
        self.fc_pixel_sal = nn.Conv2d(512 * self.feature.expansion, self.downsample ** 2, 1)
        self.fc_pixel_edge = nn.Conv2d(512 * self.feature.expansion, self.downsample ** 2, 1)

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps, max_pool=False, padding = 1):
        layers = [nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=padding),
                nn.BatchNorm2d(out_maps), nn.ReLU()]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def slow_learner(self):
        param = chain(self.feature.parameters(), self.classifier_coarse.parameters())
        for p in param:
            yield p

    def fast_learner(self):
        param = chain(self.f_conv1.parameters(), self.f_conv2.parameters(), self.f_conv3.parameters(),
                    self.f_conv4.parameters(), self.fc.parameters(), self.classifier.parameters(),
                    self.fusion_blocks1.parameters(), self.fusion_blocks2.parameters(),self.fusion_blocks3.parameters(),
                    self.fusion_blocks4.parameters())
        for p in param:
            yield p

    def learnable_parameters(self):
        param = chain(self.feature.parameters(), self.classifier_coarse.parameters(),
                      self.f_conv1.parameters(), self.f_conv2.parameters(), self.f_conv3.parameters(),
                      self.f_conv4.parameters(), self.fc.parameters(), self.classifier.parameters(),
                      self.fusion_blocks1.parameters(), self.fusion_blocks2.parameters(),
                      self.fusion_blocks3.parameters(),self.fusion_blocks4.parameters())
        for p in param:
            yield p


    def forward(self, x, **kwargs):
        noise, sal = kwargs['noise'], kwargs['sal']

        coarse_feature, (h0, h1, h2, h3, h4) = self.feature(x, noise=noise)

        m1_ = self.f_conv1(x)
        # m1 = m1_ * h1
        m1 = self.fusion_blocks1(m1_, h1)
        m2_ = self.f_conv2(m1)
        # m2 = m2_ * h2
        m2 = self.fusion_blocks2(m2_, h2)
        m3_ = self.f_conv3(m2)
        # m3 = m3_ * h3
        m3 = self.fusion_blocks3(m3_, h3)
        m4_ = self.f_conv4(m3)
        # m4 = m4_ * h4
        m4 = self.fusion_blocks4(m4_, h4)
        fine_feature = self.avgpool(m4)
        fine_feature = fine_feature.view(fine_feature.size(0), -1)
        fine_output = self.classifier(fine_feature)


        if not sal:
            return fine_output, coarse_feature, fine_feature
        else:
            intermediate_x = [m2[::4], m3[::4], m4[::4]]
            for i in range(len(intermediate_x)):
                # intermediate_x[i] = self.gradcam_net[i](intermediate_x[i])
                intermediate_x[i] = torch.mean(intermediate_x[i], dim=1, keepdim=True)
            def convert(module, x):
                x = module(x)
                spatial_size = x.size()[-2:]
                x = x.view(x.size(0), self.downsample, self.downsample, *x.size()[-2:]).permute(0, 1, 3, 2, 4)
                x = x.contiguous().view(x.size(0), self.downsample * spatial_size[0], self.downsample * spatial_size[1])
                return x
            x_sal = convert(self.fc_pixel_sal, m4[::4])
            x_edge = convert(self.fc_pixel_edge, m4[::4])
            return fine_output, coarse_feature, fine_feature, (x_sal, x_edge, intermediate_x)

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

    def feature_extractor(self, inputs):
        return self.feature(inputs)