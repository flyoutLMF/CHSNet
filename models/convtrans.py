import numpy as np
import torch
import torch.nn as nn
import torchvision
import collections
from models.transformer_module import Transformer
from models.convolution_module import Encoder, OutputNet, ConvBlock, NonLocalAttention
from models.Sampler import PatchSampler
from PIL.Image import BICUBIC
import time

class VGG16Trans(nn.Module):
    def __init__(self, dcsize, batch_norm=True, load_weights=False, args=None):
        super().__init__()
        self.args = args
        self.scale_factor = 16 // dcsize
        self.encoder = Encoder()
        # self.encoder = nn.Sequential(
        #     ConvBlock(cin=3, cout=64),
        #     ConvBlock(cin=64, cout=64),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     ConvBlock(cin=64, cout=128),
        #     ConvBlock(cin=128, cout=128),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     ConvBlock(cin=128, cout=256),
        #     ConvBlock(cin=256, cout=256),
        #     ConvBlock(cin=256, cout=256),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     ConvBlock(cin=256, cout=512),
        #     ConvBlock(cin=512, cout=512),
        #     ConvBlock(cin=512, cout=512),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     ConvBlock(cin=512, cout=512),
        #     ConvBlock(cin=512, cout=512),
        #     ConvBlock(cin=512, cout=512),
        # )
        self.tran_decoder = Transformer(layers=4)
        self.tran_decoder_p2 = OutputNet(dim=512)

        try:
            self.sampler = PatchSampler(sample_method=self.args.sample_method,
                                        feat_from=self.args.feat_from, feat_get_method=self.args.feat_get_method)
        except:  # for test
            self.sampler = PatchSampler()

        # self.conv_decoder = nn.Sequential(
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        #     ConvBlock(512, 512, 3, d_rate=2),
        # )
        # self.conv_decoder_p2 = OutputNet(dim=512)

        self._initialize_weights()
        if not load_weights:
            if batch_norm:
                mod = torchvision.models.vgg16_bn(pretrained=True)
            else:
                mod = torchvision.models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.encoder.state_dict().items())):
                temp_key = list(self.encoder.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.encoder.load_state_dict(fsd)

        # self.resize = torchvision.transforms.Resize((32, 32), interpolation=BICUBIC)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, raw_x = self.encoder(x)
        bs, c, h, w = raw_x.shape

        # path-transformer
        x = raw_x.flatten(2).permute(2, 0, 1)  # -> bs c hw -> hw b c
        x = self.tran_decoder(x, (h, w))
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        y = self.tran_decoder_p2(x)

        return y

    def calculate_loss(self, x, dmap, ones_map, criterionMSE, criterionNCE, dis, neg_points, examplers=None):
        """
        calculate loss including mse_loss and nce_loss
        x: input image
        dmap: input density map
        ones_map: input points map
        criterionMSE: the loss function for mse
        criterionNCE: the loss function for nce
        dis: the half height and half width of a patch
        neg_points: the location of negative samples got from the pre-saved .npy
        examplers: no use
        """
        x0, x1 = self.encoder(x)  # two scale feature
        bs, c, h, w = x1.shape

        # path-transformer
        x2 = x1.flatten(2).permute(2, 0, 1)  # -> bs c hw -> hw b c
        x2 = self.tran_decoder(x2, (h, w))
        x2 = x2.permute(1, 2, 0).view(bs, c, h, w)
        x3 = nn.functional.interpolate(x2, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        y = self.tran_decoder_p2(x3)

        mse_loss = criterionMSE(y, dmap)  # calculate mse loss

        # calculate nce loss
        total_nce_loss = 0.0
        for i in range(x.shape[0]):  # sample for single img
            points = self.ones2points(ones_map[i])
            if len(points) < 4 or (self.args.samples and dis[i].item() == 0):
                continue
            if self.args.feat_from == 'Attention':  # get the feature
                feats = [x[i].unsqueeze(0), x1[i].unsqueeze(0), x2[i].unsqueeze(0)]
            else:
                feats = [x[i].unsqueeze(0), x0[i].unsqueeze(0), x1[i].unsqueeze(0)]
            if examplers is not None:  # no use
                examplers = [e[i].unsqueeze(0) for e in examplers]

            # sample for pos
            query, feat_pos = self.sampler.sample_pos(feats, points, dis=dis[i].item(), examplers=examplers)
            if self.args.sample_method == 'APS':
                patch_ids = ones_map[i]
            else:
                patch_ids = torch.Tensor(neg_points[i])
            feat_neg, _ = self.sampler.sample_neg(feats, patch_ids=patch_ids, dis=dis[i].item())  # sample for neg
            if self.args.num_query == 'all':
                query = [i.clone() for i in feat_pos]
            for q, f_k_pos, f_k_neg, crit in zip(query, feat_pos, feat_neg, criterionNCE):  # calculate the nce_loss
                loss = crit(q, f_k_pos, f_k_neg)
                total_nce_loss += loss.mean()

        return total_nce_loss * self.args.lambda_nce + mse_loss, mse_loss.item(), \
               total_nce_loss.item() if isinstance(total_nce_loss, torch.Tensor) else total_nce_loss, y

    def ones2points(self, ones_map):
        """ones_map to points, return [H, W]*n """
        ones_map = ones_map.long().squeeze(0)
        points = torch.nonzero(ones_map)
        return np.array([i.cpu().numpy() for i in points])
