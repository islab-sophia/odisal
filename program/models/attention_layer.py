import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    def __init__(self, in_ch=3, ch=512, stride=1):
        raise ValueError("error!")
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv2 = nn.Conv2d(ch, in_ch, kernel_size=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(in_ch)
        
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)
        
        #h += x
        h = self.softmax(h)
        
        output = torch.sum(x.mul(h), dim=1, keepdim=True)

        return output

class AttentionV2(nn.Module):
    def __init__(self, in_ch=3, ch=512, stride=1, expansion=4):
        super(AttentionV2, self).__init__()
        raise ValueError("error!")
        self.conv1 = nn.Conv2d(in_ch, ch, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv3 = nn.Conv2d(ch, ch*expansion, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(ch*expansion)
        
        self.conv4 = nn.Conv2d(ch*expansion, in_ch, kernel_size=1, stride=stride)
        self.bn4 = nn.BatchNorm2d(in_ch)
        
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        
        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        
        h = self.conv4(h)
        h = self.bn4(h)
        
        #h += x
        h = self.softmax(h)
        output = torch.sum(x.mul(h), dim=1, keepdim=True)

        return output
    
    
class AttentionWithFeatures(nn.Module):
    def __init__(self, in_ch=3, ch=512, features_ch=4416, stride=1, expansion=4): #features : densenetの最終出力(norm5)
        super(AttentionWithFeatures, self).__init__()
        raise ValueError("error!")
        self.conv1 = nn.Conv2d(features_ch, ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv2 = nn.Conv2d(ch, in_ch, kernel_size=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(in_ch)
        
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, features):
        if x.shape[2] != features.shape[2]:
            scale = int(x.shape[2]/features.shape[2])
            features = F.interpolate(features, scale_factor=scale, mode='bilinear')
        #features = F.interpolate(features, scale_factor=8, mode='bilinear')

        h = self.conv1(features)
        h = self.bn1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)
        
        #h += x
        h = self.softmax(h)
        
        output = torch.sum(x.mul(h), dim=1, keepdim=True)
        
        return output

class AttentionV2WithFeatures(nn.Module):
    def __init__(self, in_ch=3, ch=512, features_ch=4416, stride=1, expansion=4): #features : densenetの最終出力(norm5)
        super(AttentionV2WithFeatures, self).__init__()
        self.conv1 = nn.Conv2d(features_ch, ch, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv3 = nn.Conv2d(ch, ch * expansion, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(ch * expansion)

        self.conv4 = nn.Conv2d(ch * expansion, in_ch, kernel_size=1, stride=stride)
        self.bn4 = nn.BatchNorm2d(in_ch)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, features):
        if x.shape[2] != features.shape[2]:
            scale = int(x.shape[2]/features.shape[2])
            features = F.interpolate(features, scale_factor=scale, mode='bilinear')
        # features = F.interpolate(features, scale_factor=8, mode='bilinear')
        h = self.conv1(features)
        h = self.bn1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)

        h = self.conv4(h)
        h = self.bn4(h)

#         h += x
        h = self.softmax(h)
        output = torch.sum(x.mul(h), dim=1, keepdim=True)
        
        return output