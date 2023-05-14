import numpy as np
import torch
import torch.nn as nn
import torchvision
import collections
from models.transformer_module import Transformer
from models.convolution_module import Encoder, OutputNet
from models.Sampler import PatchSampleNonlocal
from PIL.Image import BICUBIC


class VGG16Trans(nn.Module):
    def __init__(self, dcsize, batch_norm=True, load_weights=False):
        super().__init__()
        self.scale_factor = 16//dcsize
        self.encoder = Encoder()
        self.tran_decoder = Transformer(layers=4)
        self.tran_decoder_p2 = OutputNet(dim=512)

        self.sampler = PatchSampleNonlocal(input_nc=[3, 256, 512])

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

        self.resize = torchvision.transforms.Resize((32, 32), interpolation=BICUBIC)

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

    def extract_feat(self, x, points, examplers=None):
        x0, x1 = self.encoder(x)
        feats = [x, x0, x1]
        exampler, feat_pos = self.sampler.sample_pos(feats, points, examplers)
        feat_neg, _ = self.sampler.sample_neg(feats, sample_patch=exampler)

        if exampler.shape[2] < 32 or exampler.shape[3] < 32:
            exampler = self.resize(exampler)
        exampler_x0, exampler_x1 = self.encoder(exampler)
        query = self.sampler.apply_mlp([exampler, exampler_x0, exampler_x1])
        return query, feat_pos, feat_neg


    def calculate_NCE_loss(self, x, ones_map, criterionNCE, examplers=None, nce_weight=1.):
        total_nce_loss = 0.0
        for i in range(x.shape[0]):
            points = self.ones2points(ones_map[i])
            if len(points) < 3:
                continue
            exp_ = None
            if examplers is not None:
                exp_ = [e[i].unsqueeze(0) for e in examplers]
            query, feat_pos, feat_neg = self.extract_feat(x[i].unsqueeze(0), points, examplers=exp_)
            for q, f_k_pos, f_k_neg, crit in zip(query, feat_pos, feat_neg, criterionNCE):
                loss = crit(q, f_k_pos, f_k_neg) * nce_weight
                total_nce_loss += loss.mean()
        return total_nce_loss

    def ones2points(self, ones_map):
        """ones_map to points, [H, W]*n """
        ones_map = ones_map.long().squeeze(0)
        points = torch.nonzero(ones_map)
        return np.array([i.cpu().numpy() for i in points])