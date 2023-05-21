import json
import math
import random

import numpy as np
import torchvision.transforms
from tqdm import tqdm

import torch
import os
from PIL import Image
# import pydevd_pycharm
# # 本机cmd命令窗口通过ipconfig指令查询本机ipv4地址
# # 端口号可以随便改，可用即可。端口范围一般用到的是1到65535，其中0一般不使用。
# pydevd_pycharm.settrace('10.117.85.112', port=65534, stdoutToServer=True,
#                         stderrToServer=True)


torch.cuda.set_device(1)

def img_show(name, tensor):
    img = tensor.cpu().permute(0, 2, 3, 1).contiguous().squeeze(0)
    img = np.uint8(np.clip(img.numpy() * 255, 0, 255))
    img = Image.fromarray(img)
    img.show(name)


def find_dis(point):
    square = np.sum(point*point, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    # 快速排序的划分函数，找出第0,1,2,3近的四个点，第0个是自己
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4])
    return dis


def getTensorPatch(Tensor, loc, half_h, half_w):
    return Tensor[:, :, loc[0] - half_h:loc[0] + half_h, loc[1] - half_w:loc[1] + half_w]


# def getTensorLocsPatchList(tensor, patch_size, device, centers=None, count=16000):
#     assert centers is None or len(centers) == count
#     B, C, H, W = tensor.shape
#     if isinstance(patch_size, int):
#         patch_size = [patch_size, patch_size]
#     half_h, half_w = int(patch_size[0] / 2), int(patch_size[1] / 2)
#     patches = torch.zeros(size=(B, C, *patch_size), device=device)
#     locs = torch.zeros(size=(1, 2), device=device)
#     for h in range(half_h, H - half_h):
#         for w in range(half_w, W - half_w):
#             patches = torch.cat((patches, getTensorPatch(tensor, [h, w], half_h, half_w)), 0)
#             locs = torch.cat((locs, torch.Tensor([[h, w]]).cuda()), 0)
#
#     return locs[1:], patches[1:]


def getTensorLocsPatchList(tensor, patch_size, device, centers=None, count=10000):
    assert centers is None or len(centers) == count
    B, C, H, W = tensor.shape
    half_h, half_w = patch_size, patch_size
    patches = torch.zeros(size=(B, C, half_h * 2, half_w * 2), device=device)
    if centers is not None:
        sample_locs = np.array(centers) - np.array(patch_size)
    else:
        sample_locs = np.random.randint([half_h, half_w], [H - half_h, W - half_w], size=(count, 2))
    for loc in sample_locs:
        patches = torch.cat((patches, getTensorPatch(tensor, loc, half_h, half_w)), 0)

    return torch.Tensor(sample_locs), patches[1:]


def write_info(dis, centers, path):
    np.save(path, {
        'dis': dis,
        'negs': centers
    })

num_samples = 1024
root = '../datasets/FSC'
anno_file = os.path.join(root, 'annotation_FSC147_384.json')
data_split_file = os.path.join(root, 'Train_Test_Val_FSC_147.json')
im_dir = os.path.join(root, 'images_384_VarV2')
gt_dir = os.path.join(root, 'gt_density_map_adaptive_384_VarV2')
info_dir = os.path.join(root, 'samples')
if not os.path.exists(info_dir):
    os.makedirs(info_dir)
exist_info = os.listdir(info_dir)
exist_info = [i.replace('_sample.npy', '.jpg') for i in exist_info]
with open(anno_file) as f:
    annotations = json.load(f)
with open(data_split_file) as f:
    data_split = json.load(f)
im_ids = data_split['train']
trans = torchvision.transforms.ToTensor()
for im_id in tqdm(im_ids):
    if im_id in exist_info:
        continue
    img_path = os.path.join(im_dir, im_id)
    gd_path = os.path.join(gt_dir, im_id).replace('.jpg', '.npy')
    anno = annotations[im_id]
    img = trans(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
    points = np.array(anno['points'])  # W H
    try:
        dis = math.ceil(find_dis(points) / 2)
    except:
        dis = 0
    if dis == 0:
        write_info(0, np.int64(np.zeros((num_samples, 2))), os.path.join(info_dir, im_id.replace('.jpg', '_sample.npy')))
        continue

    pos_example_dis = -1
    sample_patch = None
    count = 0
    while 1:
        if pos_example_dis != dis:
            center = random.choice(points)
            sample_patch = getTensorPatch(img, [int(center[1]), int(center[0])], dis, dis)
            pos_example_dis = sample_patch.shape[2] // 2
            count += 1
            if count > len(points):
                dis = 0
                break
        else:
            break
    if dis == 0:
        write_info(0, np.int64(np.zeros((num_samples, 2))), os.path.join(info_dir, im_id.replace('.jpg', '_sample.npy')))
        continue
    # img_show("", sample_patch)
    try:
        locs, patches = getTensorLocsPatchList(img, dis, img.device)
        diff_pow = 1. - torch.pow((patches - sample_patch), 2)
        diff_sum = torch.sum(torch.sum(torch.sum(diff_pow, 2), 2), 1)
        # diff_sqr = torch.sqrt(diff_sum)  # length * 1
        score, index = torch.topk(diff_sum, num_samples, largest=False, sorted=True)
        patch_ids = locs[index.cpu()]
        patch_ids = patch_ids.to(torch.long)
        patch_ids = patch_ids.cpu().numpy()
        write_info(dis, patch_ids, os.path.join(info_dir, im_id.replace('.jpg', '_sample.npy')))
    except:
        write_info(0, np.int64(np.zeros((num_samples, 2))),
                   os.path.join(info_dir, im_id.replace('.jpg', '_sample.npy')))