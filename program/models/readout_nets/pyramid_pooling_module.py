import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, reduction_rate='default', sizes=(1,2,3,6), last_conv=True):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        all_out_channels = in_channels
        for s in sizes:
            if reduction_rate == 'default':
                reduction_channels = in_channels // (s**2)
            else:
                reduction_channels = in_channels // reduction_rate
            all_out_channels += reduction_channels
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, reduction_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_channels),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.last_conv = last_conv
        if self.last_conv:
            self.conv = nn.Conv2d(all_out_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_size = x.size()
        out = [x] + [F.interpolate(input=f(x), size=x_size[2:], mode='bilinear') for f in self.features]
        out = torch.cat(out, 1)
        if self.last_conv:
            out = self.conv(out)
        return out
