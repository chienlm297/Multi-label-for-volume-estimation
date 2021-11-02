import torch
import torch.nn as nn


class Volume(nn.Module):
    def __init__(self):
        super(Volume, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),  # 128@42*42
            nn.MaxPool2d(2),  # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),  # 128@18*18
            nn.MaxPool2d(2),  # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),  # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)
        self.head = self.create_head(8192, 10)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out3 = torch.stack((out1, out2))
        out3 = torch.cat((out1, out2), dim=1)
        out = self.head(out3)
        return self.sigmoid(out)
        return out

    def create_head(
        self, num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU
    ):
        features_lst = [num_features, num_features // 2, num_features // 4]
        print(features_lst)
        layers = []
        for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
            print(in_f, out_f)
            layers.append(nn.Linear(in_f, out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob != 0:
                layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1], number_classes))
        return nn.Sequential(*layers)
