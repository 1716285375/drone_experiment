import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import random
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class CnnDQN(nn.Module):
    def __init__(self, input_channel, num_actions):
        super(CnnDQN, self).__init__()

        self.input_channel = input_channel
        self.num_actions = num_actions

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=32, kernel_size=3, stride=1, padding=1)  # 输出: (32, 84, 84)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 池化层，输出: (32, 42, 42)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 输出: (64, 42, 42)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                               padding=1)  # 输出: (128, 42, 42)

        # 全连接层
        self.fc1 = nn.Linear(128 * 21 * 21, 512)  # 假设池化后输入是 (128, 21, 21)
        self.fc2 = nn.Linear(512, self.num_actions)  # 最终输出5个类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 输出: (32, 42, 42)
        x = self.pool(F.relu(self.conv2(x)))  # 输出: (64, 21, 21)
        x = F.relu(self.conv3(x))  # 输出: (128, 21, 21)

        x = x.view(-1, 128 * 21 * 21)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
