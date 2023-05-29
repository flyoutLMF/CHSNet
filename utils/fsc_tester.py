import os
import time
import logging
from math import ceil

import cv2
import torch
import torchvision.io.image
from torch.utils.data import DataLoader

from datasets.fsc_data import FSCData
from models.convtrans import VGG16Trans
from utils.trainer import Trainer

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import json


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    dmaps = torch.stack(transposed_batch[1], 0)
    ex_list = transposed_batch[2]
    return images, dmaps, ex_list


class FSCTester(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise Exception("gpu is not available")

        val_datasets = FSCData(args.data_dir, method='val')
        val_dataloaders = DataLoader(val_datasets, 1, shuffle=False,
                                                      num_workers=args.num_workers, pin_memory=True)
        test_datasets = FSCData(args.data_dir, method='test')
        test_dataloaders = DataLoader(test_datasets, 1, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
        self.dataloaders = {'val': val_dataloaders, 'test': test_dataloaders}

        self.model = VGG16Trans(dcsize=args.dcsize, load_weights=True, args=args)
        self.model.to(self.device)

        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model_resume(checkpoint['model_state_dict'])
            elif suf == 'pth':
                # self.model.load_state_dict(torch.load(args.resume, self.device))
                self.model_resume(torch.load(args.resume, self.device))


    def test(self):
        self.train()

    def train(self):
        self.val_epoch('val')
        self.val_epoch('test')

    def val_epoch(self, mode):
        assert mode in ['val', 'test']
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []

        root = os.path.join(self.save_dir, mode)
        if not os.path.exists(root):
            os.makedirs(root)
        result_json = {}

        for inputs, count, ex_list, name in tqdm(self.dataloaders[mode]):
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'

            max_size = 2000
            if h > max_size or w > max_size:
                h_stride = int(ceil(1.0 * h / max_size))
                w_stride = int(ceil(1.0 * w / max_size))
                h_step = h // h_stride
                w_step = w // w_stride
                pre_count = 0.0
                output = torch.zeros(size=(1, 1, h, w)).cuda()
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w

                        with torch.set_grad_enabled(False):
                            result = self.model(inputs[:, :, h_start:h_end, w_start:w_end])
                            output[:, :, h_start:h_end, w_start:w_end] = result
                            pre_count += torch.sum(result)
            else:
                with torch.set_grad_enabled(False):
                    output = self.model(inputs)
                    pre_count = torch.sum(output) / self.args.log_param
            if True:
                output = F.interpolate(output, size=(h, w), mode='bicubic', align_corners=True) / self.args.log_param
                output = output / torch.max(output)
                np_img = np.clip(output.cpu().permute(0, 2, 3, 1).contiguous().squeeze(0).squeeze(2).numpy() * 255, 0., 255.)
                heatmap = cv2.applyColorMap(np.uint8(np_img), cv2.COLORMAP_JET)
                # cv2.imshow(name[0], heatmap)
                # cv2.waitKey()
                np_img = np.clip(inputs.cpu().permute(0, 2, 3, 1).contiguous().squeeze(0).numpy() * 255, 0., 255.)
                img = cv2.cvtColor(np.uint8(np_img), cv2.COLOR_RGB2BGR)
                result_img = np.uint8(heatmap * 0.4 + img * 0.6)
                cv2.imwrite(os.path.join(root, name[0]), result_img)
                torchvision.io.image.write_png(output.cpu(), os.path.join(root, name[0]))
            result_json[name[0]] = {'gt': count[0].item(), 'pre': pre_count.item()}
            epoch_res.append(count[0].item() - pre_count.item())
            # epoch_res.append(count[0].item())

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        result_json['final_result'] = {'mse': mse, 'mae': mae}
        with open(os.path.join(root, 'result.json'), 'w') as f:
            json.dump(result_json, f)

        logging.info('{}: MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(mode, mse, mae, time.time() - epoch_start))

    def model_resume(self, ckpt):
        misses = ["sampler.mlp_0.0.weight", "sampler.mlp_0.0.bias", "sampler.mlp_0.2.weight", "sampler.mlp_0.2.bias",
               "sampler.mlp_1.0.weight", "sampler.mlp_1.0.bias", "sampler.mlp_1.2.weight", "sampler.mlp_1.2.bias"]
        our_ckpt = self.model.state_dict()
        for miss in misses:
            ckpt[miss] = our_ckpt[miss]
        self.model.load_state_dict(ckpt)
