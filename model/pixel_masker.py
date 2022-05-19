import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.common_utils import np2torch


class PixelMasker(nn.Module):
    def __init__(self, args):
        super(PixelMasker, self).__init__()
        self.p_m = args.p_m
        self.h_pad = args.h_pad
        self.w_pad = args.w_pad
        self.dropout = nn.Dropout(p=self.p_m)
        kernel = np.array([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], dtype=np.float32) / 8.
        self.kernel = np2torch(kernel[np.newaxis, np.newaxis, :, :], True)
        self.mask = torch.ones([1, args.c * args.n_file, args.h - 2 * args.h_pad, args.w - 2 * args.w_pad]).cuda()

    def forward(self, x):
        mask = self.dropout(self.mask) * (1. - self.p_m)
        padded_mask = F.pad(mask, (self.h_pad, self.h_pad, self.w_pad, self.w_pad), 'constant', value=1)
        aver_x = F.conv2d(x.permute(1, 0, 2, 3), self.kernel, stride=1, padding=1)
        aver_x = aver_x.permute(1, 0, 2, 3)
        x = padded_mask * x + (1. - padded_mask) * aver_x
        return x.detach(), mask.detach()
