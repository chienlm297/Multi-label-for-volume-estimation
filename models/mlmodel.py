import torch
import torch.nn as nn
from .resnet import BasicBlock, Bottleneck


class Volume(nn.Module):
    def __init__(self, num_classes=10):
        super(Volume, self).__init__()
        self.block = BasicBlock
        self.inplanes = 64
        self.layers = [2, 2, 2, 2]
        self.backbone = self.make_resnet_backbone(self.block, self.layers)
        self.liner = nn.Sequential(nn.Linear(1024, 4096), nn.Sigmoid())
        self.head = self.create_head(1024, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def make_resnet_backbone(self, block, layers):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2),
            nn.AvgPool2d(7, stride=1),
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def create_head(
        self, num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU
    ):
        features_lst = [num_features, num_features // 2, num_features // 4]
        layers = []
        for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob != 0:
                layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1], number_classes))
        return nn.Sequential(*layers)

    def forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out3 = torch.cat((out1, out2), dim=1)
        out = self.head(out3)
        # return self.sigmoid(out)
        return out
