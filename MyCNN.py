# from DeepTemplate.models.model import Model

import torch.nn as nn
import torchvision  # image datasets

import numpy as np
class MyCNN(nn.Module):  # 继承了nn.Module类
    def __init__(self):
        super(MyCNN, self).__init__()  # 初始化nn.Module

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5
        )

        self.relu = nn.ReLU()

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=120, out_features=84)

        self.fc2 = nn.Linear(in_features=84, out_features=10)

        self.sm = nn.LogSoftmax(dim=0)
        # self.sm = nn.Softmax(dim=0)

    def forward(self, x):   # x是一个batch
        x = self.relu(self.mp(self.conv1(x)))
        x = self.relu(self.mp(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(-1, np.product(x.shape[1:]))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sm(x)

        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.pool2(x)
        # x = x.view(-1, np.product(x.shape[1:]))
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.sm(x)
        return x
