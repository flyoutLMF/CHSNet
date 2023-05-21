import torch
import torch.nn as nn
from einops import rearrange


class DownMSELoss(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=size)
        self.tot = size * size
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, dmap, gt_density):
        gt_density = self.avgpooling(gt_density) * self.tot
        b, c, h, w = dmap.size()
        assert gt_density.size() == dmap.size()
        return self.mse(dmap, gt_density)


class PatchNCELoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.T = temp

    def forward(self, query, feat_k_pos, feat_k_neg):
        """
        query: [M, dim_feat], dim_feat=128
        feat_k_pos: positive features, [num, dim_feat]
        feat_k_neg: negative features, [256, dim_feat]
        """
        feat_k_pos = feat_k_pos.detach()
        feat_k_neg = feat_k_neg.detach()
        pos = query.mm(feat_k_pos.permute(1, 0).contiguous()) / self.T  # [M, num]
        neg = query.mm(feat_k_neg.permute(1, 0).contiguous()) / self.T  # [M, 256]
        exp_pos = pos.exp()  # [N, num]
        exp_neg = neg.exp().sum(dim=1, keepdim=True).repeat(1, pos.shape[1])  # [M, num]
        softmax_term = exp_pos / (exp_pos + exp_neg)  # [M, num]
        NCE_loss = - softmax_term.log().mean()  # cross entropy
        return NCE_loss