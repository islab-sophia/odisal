import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-7


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        """
        Args:
            size_average (bool, optional): By default, the losses are averaged
               over observations for each minibatch. However, if the field
               size_average is set to False, the losses are instead summed for
               each minibatch. Default: True
         """
        super(_Loss, self).__init__()
        self.size_average = size_average


class KLD(_Loss):

    def forward(self, sm, fm):
        """
        Args:
            sm : Variable(mini_batch_size, channels, height, width)
                Predicted saliency map
            fm : Variable(mini_batch_size, channels, height, width)
                Ground-truth fixation map
        """


        fm = fm.unsqueeze(1)

        fm = F.interpolate(fm, size=(sm.size(2), sm.size(3)), mode='bilinear')

        batchsize = sm.size(0)
        kl_divergence = 0
        for i in range(batchsize):
            sm_i = torch.unsqueeze(sm[i], dim=0)
            fm_i = torch.unsqueeze(fm[i], dim=0)



            # ps = (sm_i + eps) / (torch.sum(torch.sum(sm_i, dim=2), dim=2).expand_as(sm_i) + eps)
            # pf = (fm_i + eps) / (torch.sum(torch.sum(fm_i, dim=2), dim=2).expand_as(fm_i) + eps)

            ps = (sm_i + eps) / (torch.sum(torch.sum(torch.sum(sm_i, dim=1), dim=1), dim=1) + eps)
            pf = (fm_i + eps) / (torch.sum(torch.sum(torch.sum(fm_i, dim=1), dim=1), dim=1) + eps)


            kl_divergence += torch.sum(-pf*torch.log(ps) + pf*torch.log(pf))
        if self.size_average:
            kl_divergence = kl_divergence / sm.size(0)
        return kl_divergence


class KLD_adjust_gtmap_size(_Loss):

    def forward(self, sm, fm):
        """
        Args:
            sm : Variable(mini_batch_size, channels, height, width)
                Predicted saliency map
            fm : Variable(mini_batch_size, channels, height, width)
                Ground-truth fixation map
        """
        sm = F.interpolate(sm, size=(fm.size(1), fm.size(2)), mode='bilinear')
        fm = fm.unsqueeze(1)
        batchsize = sm.size(0)
        kl_divergence = 0
        for i in range(batchsize):
            sm_i = torch.unsqueeze(sm[i], dim=0)
            fm_i = torch.unsqueeze(fm[i], dim=0)
            ps = (sm_i + eps) / (torch.sum(torch.sum(sm_i, dim=2), dim=2).expand_as(sm_i) + eps)
            pf = (fm_i + eps) / (torch.sum(torch.sum(fm_i, dim=2), dim=2).expand_as(fm_i) + eps)
            kl_divergence += torch.sum(-pf*torch.log(ps) + pf*torch.log(pf))
        if self.size_average:
            kl_divergence = kl_divergence / sm.size(0)
        return kl_divergence


class NSS(_Loss):

    def forward(self, sm, fm):
        """
        Args:
            sm : Variable (N,H,W) or (N,1,H,W)
                Predicted saliency map
            fm : Variable (N,H,W) or (N,1,H,W)
                Human fixation map (binary map).
        """

        sm = F.interpolate(sm, size=(fm.size(1), fm.size(2)), mode='bilinear')

        sm = sm.view(sm.size(0), -1)
        fm = fm.view(fm.size(0), -1)

        sm_mean = sm.mean(dim=1, keepdim=True)
        sm_std = sm.std(dim=1, unbiased=False, keepdim=True).add(eps)
        sm_normalized = sm.sub(sm_mean).div(sm_std)
        nss = sm_normalized.mul(fm).sum(dim=1).div(fm.sum(dim=1))
        nss = nss.mean() if self.size_average else nss.sum()

        return nss.mul(-1)


class CC(_Loss):

    def forward(self, sm, fm):
        """
        Args:
            sm : Variable (N,H,W) or (N,1,H,W)
                Predicted saliency map
            fm : Variable (N,H,W) or (N,1,H,W)
                Ground-truth fixation map
        """

        sm = F.interpolate(sm, size=(fm.size(1), fm.size(2)), mode='bilinear')

        sm = sm.view(sm.size(0), -1)
        fm = fm.view(fm.size(0), -1)

        sm = (sm - sm.mean()) / (sm.std() + eps)
        fm = (fm - fm.mean()) / (fm.std() + eps)
        cc = torch.sum(sm * fm, dim=1) / sm.size(1)

        return cc.mul(-1)

class SIM(_Loss):

    def forward(self, sm, fm):
        """
        Args:
            sm : Variable (N,H,W) or (N,1,H,W)
                Predicted saliency map
            fm : Variable (N,H,W) or (N,1,H,W)
                Ground-truth fixation map
        """

        sm = F.interpolate(sm, size=(fm.size(1), fm.size(2)), mode='bilinear')

        sm = sm.view(sm.size(0), -1)
        fm = fm.view(fm.size(0), -1)

        sm = (sm - sm.min()) / (sm.max() - sm.min())
        fm = (fm - fm.min()) / (fm.max() - fm.min())

        sm = sm / (sm.sum(dim=1) + eps)
        fm = fm / (fm.sum(dim=1) + eps)

        sim = torch.sum(sm * (sm < fm).float() + fm * (sm >= fm).float())
        return sim.mul(-1)
