import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import random, math
from PIL import Image

def img_show(name, tensor):
    std = torch.Tensor([0.485, 0.456, 0.406])
    mean = torch.Tensor([0.229, 0.224, 0.225])
    img = tensor.cpu().permute(0, 2, 3, 1).contiguous().squeeze(0) * std + mean
    img = np.uint8(np.clip(img.numpy() * 255, 0, 255))
    img = Image.fromarray(img)
    img.show(name)


def find_dis(point):
    try:
        square = np.sum(point*point, axis=1)
        dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
        # 快速排序的划分函数，找出第0,1,2,3近的四个点，第0个是自己
        dis = np.mean(np.partition(dis, 1, axis=1)[:, 1:2])
    except Exception as e:
        print(e)
        dis = 32
    return dis


def getTensorPatch(Tensor, loc, half_h, half_w):
    h, w = Tensor.shape[2], Tensor.shape[3]
    top = loc[0] - half_h if loc[0] - half_h >= 0 else 0
    down = loc[0] + half_h if loc[0] + half_h < h else h - 1
    left = loc[1] - half_w if loc[1] - half_w >= 0 else 0
    right = loc[1] + half_w if loc[1] + half_w < w else w - 1
    if (down - top) % 2:
        if top > 0:
            top -= 1
        else:
            down += 1
    if (right - left) % 2:
        if left > 0:
            left -= 1
        else:
            right += 1
    return Tensor[:, :, top:down, left:right]


def getTensorLocsPatchList(tensor, patch_size, device, centers=None, count=1600):
    assert centers is None or len(centers) == count
    B, C, H, W = tensor.shape
    if isinstance(patch_size, int):
        patch_size = [patch_size, patch_size]
    half_h, half_w = int(patch_size[0] / 2), int(patch_size[1] / 2)
    patches = torch.zeros(size=(B, C, *patch_size), device=device)
    if centers is not None:
        sample_locs = np.array(centers) - np.array(patch_size)
    else:
        sample_locs = np.random.randint([half_h, half_w], [H - half_h, W - half_w], size=(count, 2))
    for loc in sample_locs:
        patches = torch.cat((patches, getTensorPatch(tensor, loc, half_h, half_w)), 0)

    return torch.Tensor(sample_locs), patches[1:]


class PatchSampleNonlocal(nn.Module):
    def __init__(self, nc=256, num_patches=256,
                 sample_method='NonLocal', feat_from='Encoder', feat_get_method='Point'):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleNonlocal, self).__init__()
        self.nc = nc  # hard-coded
        self.num_patches = num_patches

        assert sample_method in ['NonLocal', 'Random']
        self.sample_method = sample_method
        assert feat_from in ['Encoder', 'Attention']
        self.feat_from = feat_from
        if feat_from == 'Encoder':
            input_nc = [256, 512]
        else:
            input_nc = [512, 512]
        assert feat_get_method in ['Point', 'MaxPool']
        self.feat_get_method = feat_get_method

        # self.patch_size = patch_size
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        for mlp_id, inc in enumerate(input_nc):
            mlp = nn.Sequential(*[nn.Linear(inc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

    def forward(self, feats, sample_patch=None, patch_ids=None, use_mlp=True):
        pass

    def sample_neg(self, feats, sample_patch, patch_ids=None, use_mlp=True):
        """
        get feature in points
        TODO: get feature from maxpool
        """
        return_feats = []
        patch_size = [sample_patch.shape[2], sample_patch.shape[3]]
        img = feats[0]
        if patch_ids is None:  # calculate the num_patches non-local keys in raw image
            if self.sample_method == 'NonLocal':
                locs, patches = getTensorLocsPatchList(img, patch_size, img.device)
                diff_pow = 1. - torch.pow((patches - sample_patch), 2)
                diff_sum = torch.sum(torch.sum(torch.sum(diff_pow, 2), 2), 1)
                # diff_sqr = torch.sqrt(diff_sum)  # length * 1
                score, index = torch.topk(diff_sum, self.num_patches, largest=False, sorted=True)
                patch_ids = locs[index]
                patch_ids = patch_ids.to(torch.long)
            elif self.sample_method == 'Random':
                B, C, H, W = img.shape
                half_h, half_w = int(patch_size[0] / 2), int(patch_size[1] / 2)
                patch_ids = torch.Tensor(np.random.randint([half_h, half_w], [H - half_h, W - half_w],
                                                           size=(self.num_patches, 2)))
            else:
                raise NotImplementedError
            # img_show("a", patches[index[0]].unsqueeze(0))
            # img_show("b", patches[index[-1]].unsqueeze(0))

        feats = feats[1:]
        for feat_id, feat in enumerate(feats):
            if self.feat_get_method == 'Point':
                B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
                feat_patch_loc = torch.div((patch_ids * H), img.shape[2], rounding_mode='floor')
                patch_id = feat_patch_loc[:, 0] * W + feat_patch_loc[:, 1]  # 256
                # patch_id shape: num_patches
                patch_id = patch_id.to(torch.long)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            elif self.feat_get_method == 'MaxPool':
                raise NotImplementedError
            else:
                raise NotImplementedError
            if use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            # return_ids.append(patch_id)
            x_sample = F.normalize(x_sample, p=2, dim=1)

            return_feats.append(x_sample)
        return_ids = patch_ids
        return return_feats, return_ids

    def sample_pos(self, feats, points, examplers=None, use_mlp=True):
        """
        get feature in a point. no use scale
        points: [H*W]*n
        """
        img = feats[0]
        if examplers is None:
            dis = math.ceil(find_dis(points) / 2)
            # index = random.randint(0, len(points))
            center = random.choice(points)
            examplers = [getTensorPatch(img, [int(center[0]), int(center[1])], dis, dis)]
        # img_show('ex', examplers[0])

        # max need to le the num of neg patches
        if len(points) > self.num_patches:
            points = points[:self.num_patches, :]
        points_torch = torch.tensor(points).long().cuda()

        return_feats = []

        feats = feats[1:]
        for feat_id, feat in enumerate(feats):
            if self.feat_get_method == 'Point':
                B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
                feat_patch_loc = torch.div((points_torch * H), img.shape[2], rounding_mode='floor')
                patch_id = feat_patch_loc[:, 0] * W + feat_patch_loc[:, 1]
                patch_id = patch_id.to(torch.long)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            elif self.feat_get_method == 'MaxPool':
                raise NotImplementedError
            else:
                raise NotImplementedError

            if use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            x_sample = F.normalize(x_sample, p=2, dim=1)

            return_feats.append(x_sample)

        return examplers[0], return_feats

    def apply_mlp(self, feats):
        return_feats = []
        for feat_id, feat in enumerate(feats):
            feat = self.global_max_pool(feat)
            A = feat.shape[1]
            feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, A)
            mlp = getattr(self, 'mlp_%d' % feat_id)
            feat = mlp(feat)
            feat = F.normalize(feat, p=2, dim=1)
            return_feats.append(feat)
        return return_feats

