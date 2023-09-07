import torch
import torch.nn as nn
import torch.nn.functional as F


class ArgumentError(Exception):
    pass


class BaseModel(nn.Module):
    def __init__(self, main_net, readout_net, single_fine_path=False, single_coarse_path=False):
        super(BaseModel, self).__init__()
        if single_fine_path and single_coarse_path:
            raise ArgumentError("Don't set both single_fine_path and single_coarse_path to True.")
        self.main_net = main_net
        self.readout_net = readout_net
        self.single_fine_path = single_fine_path
        self.single_coarse_path = single_coarse_path

    def forward(self, x):
        fine = x
        h_fine = self.main_net(fine)

        if not self.single_fine_path:
            coarse = F.interpolate(x, (x.size(2)//2, x.size(3)//2), mode='bilinear')
            h_coarse = F.interpolate(self.main_net(coarse), (h_fine.size(2), h_fine.size(3)), mode='bilinear')
            if self.single_coarse_path:
                #out = F.relu(self.readout_net(h_coarse), inplace=True)
                out = F.leaky_relu(self.readout_net(h_fine), negative_slope=0.01, inplace=True)
                if out.data[0].min() < 0:
                    out = out - out.min()
                return out
            h_fine = torch.cat([h_fine, h_coarse], dim=1)

        features = h_fine
        #out = F.relu(self.readout_net(h_fine), inplace=True)
        out = F.leaky_relu(self.readout_net(h_fine), negative_slope=0.01, inplace=True)
        if out.data[0].min() < 0:
            out = out - out.min()
        return out, features
