import torch
import torch.nn as nn
from models.interpolate import Interpolate

class NearestInterpolationModule(nn.Sequential):
    def __init__(self, in_channels, scale_factor):
        super(NearestInterpolationModule, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0))
        self.add_module('nearest1', Interpolate(scale_factor=scale_factor, mode='nearest'))


class BilinearInterpolationModule(nn.Sequential):
    def __init__(self, in_channels, scale_factor):
        super(BilinearInterpolationModule, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0))
        self.add_module('bilinear1', Interpolate(scale_factor=scale_factor, mode='bilinear'))


class DeconvolutionModule(nn.Sequential):
    def __init__(self, in_channels, num_upsampling_layers):
        super(DeconvolutionModule, self).__init__()
        out_channels = 128
        for i in range(1, num_upsampling_layers+1):
            self.add_module('deconv{}'.format(i), nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=2))
            self.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
            if i != num_upsampling_layers:
                in_channels = out_channels
                out_channels = in_channels // 2
            else:
                self.add_module('conv.{}'.format(i+1), nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0))


class SubPixelConvModule(nn.Sequential):
    def __init__(self, in_channels, num_upsampling_layers):
        super(SubPixelConvModule, self).__init__()
        in_channels = int(in_channels / 4)
        for i in range(1, num_upsampling_layers+1):
            self.add_module('pixel_shuffle.{}'.format(i), nn.PixelShuffle(upscale_factor=2))
            self.add_module('conv{}'.format(i), nn.Conv2d(in_channels, in_channels, 3, padding=1))
            self.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
            if i != num_upsampling_layers:
                in_channels = in_channels // 4
            else:
                self.add_module('conv.{}'.format(i+1), nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0))

class ConvModule(nn.Sequential):
    def __init__(self, in_channels, num_layers):
        super(ConvModule, self).__init__()
        for i in range(1, num_layers+1):
            out_channels = in_channels // 4
            self.add_module('conv{}'.format(i), nn.Conv2d(in_channels, out_channels, 3, padding=1))
            self.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
            if i != num_layers:
                in_channels = out_channels
            else:
                self.add_module('conv{}'.format(i+1), nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0))
