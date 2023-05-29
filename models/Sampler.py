import os

import PIL.ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, math
from PIL import Image


def img_show(name, tensor):
    """
    show the normalized img
    """
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
        dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4])
    except Exception as e:
        print(e)
        dis = 32
    return dis


def getTensorPatch(Tensor, loc, half_h, half_w):
    """
    get the patch of the tensor
    Tensor: input img
    loc: the center position for the patch
    half_h: the half height of the patch
    half_w: the half width of the patch
    """
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


def getTensorLocsPatchList(tensor, patch_size, device, centers=None, count=600):
    """
    get a patch list and position list
    tensor: input img
    patch_size: the patch size
    centers: the position list to get
    count: the num of patches
    """
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


class PatchSampler(nn.Module):
    def __init__(self, nc=128, num_patches=256,
                 sample_method='APS', feat_from='Attention', feat_get_method='Point'):
        """
        nc: the output feature dimension
        num_patches: the sampled patches number
        """
        super(PatchSampler, self).__init__()
        self.nc = nc
        self.num_patches = num_patches
        self.d_i = 0  # for debug visualization
        self.std = torch.Tensor([0.485, 0.456, 0.406])
        self.mean = torch.Tensor([0.229, 0.224, 0.225])

        assert sample_method in ['RS', 'LSS', 'APS']
        self.sample_method = sample_method
        assert feat_from in ['Encoder', 'Attention']
        self.feat_from = feat_from
        if feat_from == 'Encoder':  # set for mlp input dimension
            input_nc = [256, 512]
        else:
            input_nc = [512, 512]
        assert feat_get_method in ['Point', 'MaxPool']
        self.feat_get_method = feat_get_method

        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        for mlp_id, inc in enumerate(input_nc):  # mlp
            mlp = nn.Sequential(*[nn.Linear(inc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        for m in self.modules():  # initialize
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

    def forward(self, feats, sample_patch=None, patch_ids=None, use_mlp=True):
        pass

    def sample_neg(self, feats, sample_patch=None, patch_ids=None, use_mlp=True, dis=0):
        """
        get negative samples
        feats: input feature list: [original image, feature 1, feature 2]
        sample_patch: the sample patch, which is the exampler to get the most unsimilar patch
        patch_ids: patch center position, if is None, it will get patch_ids real time; otherwise get from pre-saved .npy
        use_mlp: if use mlp to project feature
        dis: the height and width of the patch
        return: features, position
        """
        assert sample_patch is not None or patch_ids is not None
        return_feats = []
        img = feats[0]  # get the original image
        if patch_ids is None:  # get non-local patch real time
            patch_size = [sample_patch.shape[2], sample_patch.shape[3]] if sample_patch is not None else [32, 32]
            if self.sample_method == 'LSS':  # will take a lot of time
                locs, patches = getTensorLocsPatchList(img, patch_size, img.device)  # get patches and location
                diff_pow = 1. - torch.pow((patches - sample_patch), 2)
                diff_sum = torch.sum(torch.sum(torch.sum(diff_pow, 2), 2), 1)  # calculate the unsimilar score
                # diff_sqr = torch.sqrt(diff_sum)  # length * 1
                score, index = torch.topk(diff_sum, self.num_patches, largest=False, sorted=True)  # get the tops
                patch_ids = locs[index]
                patch_ids = patch_ids.to(torch.long)
            elif self.sample_method == 'RS':
                B, C, H, W = img.shape
                half_h, half_w = int(patch_size[0] / 2), int(patch_size[1] / 2)
                patch_ids = torch.Tensor(np.random.randint([half_h, half_w], [H - half_h, W - half_w],
                                                           size=(self.num_patches, 2)))  # random patch_ids
            else:
                raise NotImplementedError
        elif self.sample_method == 'APS':  # sample from the whole feature map, like similarity loss of BMNet
            assert self.feat_get_method == 'Point'
            ones_map = patch_ids.unsqueeze(0)  # the patch_ids is the ones_map of the points to be counted
            ones_map = F.max_pool2d(ones_map, 16).squeeze(1).squeeze(0)
            img = feats[-1]
            patch_ids = torch.nonzero(ones_map == 0, as_tuple=False)  # get all the negative sample
                # patch_ids = patch_ids[:, 2:]

            # img_show("a", patches[index[0]].unsqueeze(0))
            # img_show("b", patches[index[-1]].unsqueeze(0))

        feats = feats[1:]
        for feat_id, feat in enumerate(feats):
            if self.feat_get_method == 'Point':
                B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
                feat_patch_loc = torch.div((patch_ids * H), img.shape[2], rounding_mode='floor')  # adjust the location
                patch_id = feat_patch_loc[:, 0] * W + feat_patch_loc[:, 1]  # 256
                # patch_id shape: num_patches
                patch_id = patch_id.to(torch.long)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            elif self.feat_get_method == 'MaxPool':
                _, C, H, W = feat.shape
                dis = int(dis * H / img.shape[2])  # adjust dis
                dis = 1 if dis < 1 else dis
                H, W = math.ceil(H / dis), math.ceil(W / dis)  # the output size after maxpool
                feat_maxpool = F.adaptive_max_pool2d(feat, (H, W))  # maxpool from the feature map
                feat_reshape = feat_maxpool.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
                feat_patch_loc = torch.div((patch_ids * H), img.shape[2], rounding_mode='floor')
                patch_id = feat_patch_loc[:, 0] * W + feat_patch_loc[:, 1]  # 256
                # patch_id shape: num_patches
                patch_id = patch_id.to(torch.long)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                raise NotImplementedError
            if use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            # return_ids.append(patch_id)
            x_sample = F.normalize(x_sample, p=2, dim=1)

            return_feats.append(x_sample)
        # if self.d_i <= 6370:
        #     self.debug_visualization(img, patch_ids, dis * 2)
        return_ids = patch_ids
        return return_feats, return_ids

    def sample_pos(self, feats, points, dis=0, examplers=None, use_mlp=True):
        """
        get positive features from the points
        return the query and positive sample
        """
        img = feats[0]
        # if examplers is None:
        #     dis = math.ceil(find_dis(points) / 2)
        #     # index = random.randint(0, len(points))
        #     center = random.choice(points)
        #     examplers = [getTensorPatch(img, [int(center[0]), int(center[1])], dis, dis)]
        # img_show('ex', examplers[0])

        # max need to le the num of neg patches
        if len(points) > self.num_patches and self.sample_method != 'APS':
            points = points[:self.num_patches, :]
        exampler_pos = random.randint(0, len(points) - 1)
        points_torch = torch.tensor(points).long().cuda()

        return_feats = []
        exampler_feats = []

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
                _, C, H, W = feat.shape
                dis = int(dis * H / img.shape[2])
                dis = 1 if dis < 1 else dis
                H, W = math.ceil(H / dis), math.ceil(W / dis)
                feat_maxpool = F.adaptive_max_pool2d(feat, (H, W))
                feat_reshape = feat_maxpool.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
                feat_patch_loc = torch.div((points_torch * H), img.shape[2], rounding_mode='floor')
                patch_id = feat_patch_loc[:, 0] * W + feat_patch_loc[:, 1]  # 256
                # patch_id shape: num_patches
                patch_id = patch_id.to(torch.long)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                raise NotImplementedError

            if use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            x_sample = F.normalize(x_sample, p=2, dim=1)

            return_feats.append(x_sample)
            exampler_feats.append(x_sample[exampler_pos, :].clone().unsqueeze(0))

        # if self.d_i <= 6370:
        #     self.debug_visualization(img, points_torch, dis * 2)

        return exampler_feats, return_feats

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

    def debug_visualization(self, img, points, dis):
        self.d_i += 1
        dis = dis // 2
        points = points.cpu().numpy()
        denormed_img = img.cpu().permute(0, 2, 3, 1).contiguous().squeeze(0) * self.std + self.mean
        denormed_img = np.uint8(np.clip(denormed_img.numpy() * 255, 0, 255))
        img1 = Image.fromarray(denormed_img)
        draw = PIL.ImageDraw.Draw(img1)
        for point in points:
            pos = (point[1] - dis, point[0] - dis, point[1] + dis, point[0] + dis)
            draw.rounded_rectangle(pos, width=2)
        if not os.path.exists('visualization'):
            os.mkdir('visualization')
        img2 = Image.fromarray(denormed_img)
        img2.save(os.path.join('visualization', str(self.d_i) + '_ori.png'))
        img1.save(os.path.join('visualization', str(self.d_i) + '_all.png'))
