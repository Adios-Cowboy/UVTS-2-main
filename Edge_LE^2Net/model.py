import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

class WRCAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(WRCAM, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Tanh(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Tanh(),
        )

    def forward(self, input):
        return input * self.module(input).unsqueeze(2).unsqueeze(3).expand_as(input)

class version_4_6_1_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(), # SiLU 激活函数
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(),  # SiLU 激活函数
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(),  # SiLU 激活函数
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(),  # SiLU 激活函数
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(), # SiLU 激活函数
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(),  # SiLU 激活函数
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(),  # SiLU 激活函数
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.InstanceNorm2d(64),  # BatchNorm
            nn.LeakyReLU(),  # SiLU 激活函数
        )
        self.conv3 = WRCAM(64,8)##nn.Identity() ##CAM(64,8)
        self.conv4 = WRCAM(64,8)##nn.Identity() ##CAM(64, 8)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),  # 3x3卷积
            nn.Sigmoid(),
        )
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x1)
        x4 = self.conv4(x2)
        x_total = torch.cat([x3, x4], dim=1)
        x_out = self.conv5(x_total)
        return x_out
