from .layers import Conv2d

import torch.nn as nn
from torch.nn.functional import avg_pool2d
import torch
from math import log2
from prosr.logger import error


class ProSRD(nn.Module):
    """SRGAN discriminator"""

    def __init__(self, ndf, max_scale, scale_overhead, **kwargs):
        super(ProSRD, self).__init__()

        self.init_conv = Conv2d(3, ndf, 3)
        self.n_pyramids = int(log2(max_scale))
        self.scale_overhead = scale_overhead

        # used in curriculum learning, initially set to the last scale
        self.current_scale_idx = self.n_pyramids - 1

        if scale_overhead:
            for i in range(1, self.n_pyramids):
                layers = []
                layers += [Conv2d(ndf, ndf, 3),
                           nn.LeakyReLU(negative_slope=0.2),
                           Conv2d(ndf, ndf, 3),
                           nn.LeakyReLU(negative_slope=0.2),
                           nn.AvgPool2d(2)]
                self.add_module('pyramid_level_%d' % i, nn.Sequential(*layers))

        num_features = ndf
        layers = []
        layers += [Conv2d(num_features, num_features*2, 3),
                   nn.LeakyReLU(negative_slope=0.2),
                   Conv2d(num_features*2, num_features*2, 3),
                   nn.LeakyReLU(negative_slope=0.2),
                   nn.AvgPool2d(2),
                   Conv2d(num_features*2, num_features*4, 3),
                   nn.LeakyReLU(negative_slope=0.2),
                   Conv2d(num_features*4, num_features*4, 3),
                   nn.LeakyReLU(negative_slope=0.2),
                   nn.AvgPool2d(2),
                   Conv2d(num_features*4, num_features*8, 3),
                   nn.LeakyReLU(negative_slope=0.2),
                   nn.AvgPool2d(2),
                   Conv2d(num_features*8, num_features*8, 3),
                   nn.LeakyReLU(negative_slope=0.2),
                   nn.AvgPool2d(2),
                   Conv2d(num_features*8, num_features*8, 3),
                   Conv2d(num_features*8, 1, 3),
                   ]
        self.base = nn.Sequential(*layers)

    def forward(self, x, upscale_factor=None, blend=1.0):
        if upscale_factor is None:
            upscale_factor = self.max_scale
        else:
            valid_upscale_factors = [
                2**(i + 1) for i in range(self.n_pyramids)
            ]
            if upscale_factor not in valid_upscale_factors:
                error("Invalid upscaling factor {}: choose one of: {}".format(
                    upscale_factor, valid_upscale_factors))
                raise SystemExit(1)

        features = self.init_conv(x)

        if self.scale_overhead:
            # 3 -> 2 -> 1
            for s in range(int(log2(upscale_factor)), 1, -1):
                features = blend * getattr(self, 'pyramid_level_%d' % s)(features)
                if (s - 1) == self.current_scale_idx and blend != 1:
                    tmp = self.init_conv(avg_pool2d(x, 2)) * (1 - blend)
                    features += tmp
        pred = self.base(features)
        return pred

    def class_name(self):
        return 'ProSRD'


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = torch.cuda.FloatTensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = torch.cuda.FloatTensor(input.size()).fill_(self.fake_label)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.detach())
