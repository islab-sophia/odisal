"""
DenseSal's densesal.py

DenseNet modified from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
pytorch > 0.4
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from models.interpolate import Interpolate
from .attention_layer import Attention
from odimapping.unify import cropping

import re


def basedensesal(pretrained=True):
    model = BaseDenseSal(pretrained=pretrained)
    return model


def densesalni1(pretrained=True):
    model = DenseSalNI(pretrained=pretrained, scale_factor=2)
    return model


def densesalni2(pretrained=True):
    model = DenseSalNI(pretrained=pretrained, scale_factor=4)
    return model


def densesalni3(pretrained=True):
    model = DenseSalNI(pretrained=pretrained, scale_factor=8)
    return model


def densesalbi1(pretrained=True):
    model = DenseSalBI(pretrained=pretrained, scale_factor=2)
    return model


def densesalbi2(pretrained=True):
    model = DenseSalBI(pretrained=pretrained, scale_factor=4)
    return model


def densesalbi3(pretrained=True):
    model = DenseSalBI(pretrained=pretrained, scale_factor=8)
    return model

def densesaldc1(pretrained=True):
    model = DenseSalDC(pretrained=pretrained, num_upsampling_layers=1)
    return model


def densesaldc2(pretrained=True):
    model = DenseSalDC(pretrained=pretrained, num_upsampling_layers=2)
    return model


def densesaldc3(pretrained=True):
    model = DenseSalDC(pretrained=pretrained, num_upsampling_layers=3)
    return model


def densesalspc1(pretrained=True):
    model = DenseSalSPC(pretrained=pretrained, num_upsampling_layers=1)
    return model


def densesalspc2(pretrained=True):
    model = DenseSalSPC(pretrained=pretrained, num_upsampling_layers=2)
    return model


def densesalspc3(pretrained=True):
    model = DenseSalSPC(pretrained=pretrained, num_upsampling_layers=3)
    return model

def densesalbi3_with_attention(pretrained=True, use_attention=False):
    model = DenseSalBI3_With_Attention(pretrained=pretrained, use_attention=use_attention)
    return model

models = {
    'basedensesal': basedensesal,
    'densesalni1': densesalni1,
    'densesalni2': densesalni2,
    'densesalni3': densesalni3,
    'densesalbi1': densesalbi1,
    'densesalbi2': densesalbi2,
    'densesalbi3': densesalbi3,
    'densesaldc1': densesaldc1,
    'densesaldc2': densesaldc2,
    'densesaldc3': densesaldc3,
    'densesalspc1': densesalspc1,
    'densesalspc2': densesalspc2,
    'densesalspc3': densesalspc3,
    'densesalbi3_with_attention': densesalbi3_with_attention
}

'''
class BaseDenseSal(nn.Module):
    def __init__(self, pretrained=True):
        super(BaseDenseSal, self).__init__()
        self.main_net = densenet161(pretrained)
        self.readout_net = nn.Conv2d(4416, 1, kernel_size=1, stride=1, padding=0)
        del self.main_net.classifier

    def forward(self, x):
        h1 = self.main_net(x)
        x_half = F.interpolate(x, (x.size(2)//2, x.size(3)//2), mode='bilinear')
        h2_half = self.main_net(x_half)
        h2 = F.interpolate(h2_half, size=(h1.size(2),h1.size(3)), mode='bilinear')
        h = torch.cat([h1, h2], dim=1)
        h = self.readout_net(h)
        out = F.relu(h, inplace=True)
        return out
'''
class BaseDenseSal(nn.Module):
    def __init__(self, pretrained=True):
        super(BaseDenseSal, self).__init__()
        self.main_net = densenet161(pretrained)
        self.readout_net = nn.Conv2d(4416, 1, kernel_size=1, stride=1, padding=0)
        del self.main_net.classifier

    def forward(self, x):
        h1 = self.main_net(x)
        x_half = F.interpolate(x, (x.size(2)//2, x.size(3)//2), mode='bilinear')
        h2_half = self.main_net(x_half)
        h2 = F.interpolate(h2_half, size=(h1.size(2),h1.size(3)), mode='bilinear')
        h = torch.cat([h1, h2], dim=1)
        features = h
        h = self.readout_net(h)
        out = F.relu(h, inplace=True)
        return out, features

# Nobutsuneくんのプログラムでは使っていない。densesalbi3を使ってtrain_ms.pyで後からAttentionをかけて統合
class DenseSalBI3_With_Attention(nn.Module):
    def __init__(self, pretrained=True, attention_net_input_channel=3, view_angle_list=[80,100,120], use_attention=True):
        super(DenseSalBI3_With_Attention, self).__init__()
        self.main_net = densesalbi3(pretrained)
        self.attention_net = Attention(attention_net_input_channel)
        self.view_angle_list = view_angle_list
        self.use_attention = use_attention

    def forward(self, inputs):
        densesal_outputs = []
        # inputs.size() = torch.Size([3, 3, 500, 500])
        # 各画角の画像をmain_netに学習させて、同一位置の画像をまとめる
        for i in range(inputs.size(0)):
            x = inputs[i].unsqueeze(0)
            output = self.main_net(x)
            # 同一位置の画像を一番小さい画角の画像と合わせる
            # if i != 0:
            #    output = cropping(self.view_angle_list[0], self.view_angle_list[i], output)
            densesal_outputs.append(output)
        densesal_outputs=torch.cat(densesal_outputs, dim=1)

        # 同一位置のまとまった画像をattention_netで結合時の重みを学習
        if self.use_attention:
            attention_net_output = self.attention_net(densesal_outputs)
            w1, w2, w3 = torch.split(attention_net_output, split_size_or_sections=1, dim=1)
                
            g1 = densesal_outputs[0].mul(w1)
            g2 = densesal_outputs[1].mul(w2)
            g3 = densesal_outputs[2].mul(w3)
            h = torch.cat([g1, g2, g3], dim=1)
            
            h = torch.sum(h, dim=1, keepdim=True)
            final_out = F.softmax(h)
        else:
            final_out = torch.sum(densesal_outputs, dim=1, keepdim=True) / 3
            final_out = F.softmax(final_out)
        return final_out


class DenseSalNI(BaseDenseSal):
    def __init__(self, pretrained=True, scale_factor=8):
        super(DenseSalNI, self).__init__()
        self.main_net = densenet161(pretrained)
        self.readout_net = nn.Sequential(
            nn.Conv2d(4416, 1, kernel_size=1, stride=1, padding=0),
            Interpolate(scale_factor=scale_factor, mode='bilinear')
        )
        del self.main_net.classifier



class DenseSalBI(BaseDenseSal):
    def __init__(self, pretrained=True, scale_factor=8):
        super(DenseSalBI, self).__init__()
        self.main_net = densenet161(pretrained)
        self.readout_net = nn.Sequential(
            nn.Conv2d(4416, 1, kernel_size=1, stride=1, padding=0),
            Interpolate(scale_factor=scale_factor, mode='bilinear')
        )
        del self.main_net.classifier


class DenseSalDC(BaseDenseSal):
    def __init__(self, pretrained=True, num_upsampling_layers=3):
        super(DenseSalDC, self).__init__()
        self.main_net = densenet161(pretrained)
        layers = OrderedDict()
        in_chs = 4416
        out_chs = 128
        for i in range(1, num_upsampling_layers+1):
            layers['deconv_{}'.format(i)] = nn.ConvTranspose2d(in_chs, out_chs, 4, stride=2, padding=2)
            layers['relu_{}'.format(i)] = nn.ReLU(inplace=True)
            if i != num_upsampling_layers:
                in_chs = out_chs
                out_chs = int(in_chs / 2)
        layers['conv_last'] = nn.Conv2d(out_chs, 1, kernel_size=1, stride=1, padding=0)
        self.readout_net = nn.Sequential(layers)
        del self.main_net.classifier


class DenseSalSPC(BaseDenseSal):
    def __init__(self, pretrained=True, num_upsampling_layers=3):
        super(DenseSalSPC, self).__init__()
        self.main_net = densenet161(pretrained)
        layers = OrderedDict()
        in_chs = int(4416 / 4)
        for i in range(1, num_upsampling_layers+1):
            layers['pixel_shuffle_{}'.format(i)] = nn.PixelShuffle(upscale_factor=2)
            layers['conv_{}'.format(i)] = nn.Conv2d(in_chs, in_chs, 3, padding=1)
            layers['relu_{}'.format(i)] = nn.ReLU(inplace=True)
            if i != num_upsampling_layers:
                in_chs = int(in_chs / 4)
        layers['conv_last'] = nn.Conv2d(in_chs, 1, kernel_size=1, stride=1, padding=0)
        self.readout_net = nn.Sequential(layers)
        del self.main_net.classifier

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        """Remove global average pooling and classifier since DenseSal needs feature maps before the pooling layer.
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        """
        return out
