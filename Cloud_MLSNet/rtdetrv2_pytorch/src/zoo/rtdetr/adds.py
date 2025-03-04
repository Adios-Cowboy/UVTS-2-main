import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ADD_Block']


class ADD_Block(nn.Module):
    def __init__(self, channels, ratio, factor=32):
        super(ADD_Block, self).__init__()

        if ratio > 1:
            self.re_size = torch.nn.AvgPool2d(kernel_size=ratio, stride=ratio) ##下采样
        elif ratio < 1:
            self.re_size = nn.Upsample(None, 1//ratio, 'nearest')
        elif ratio == 1:
            self.re_size = nn.Identity()
        else:
            raise ValueError("ratio is uncorrect")

        self.conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1_resized = self.re_size(x[0])
        x_concat = torch.cat((x1_resized, x[1]), dim=1)
        x_concat = self.conv(x_concat)
        b, c, h, w = x_concat.size()
        group_x = x_concat.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
