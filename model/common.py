import torch
import torch.nn as nn


class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, n_file, is_dropout=True, act='LReLU', p=0.075):
        super(conv_layer, self).__init__()
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(p=p)
        self.conv = nn.Conv2d(in_channels * n_file, out_channels * n_file, kernel_size=3, stride=1, groups=n_file,
                              padding=(1, 1))
        if act == 'LReLU':
            self.act = nn.LeakyReLU()
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Linear':
            self.act = nn.Identity()
        elif act == 'ReLU_Threshold':
            self.act = nn.Threshold(1e-7, 1e-7)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.act(x)
        return x


def concat(x, y, n_file):
    lx = list(x.shape)
    ly = list(y.shape)
    h = min(lx[2], ly[2])
    w = min(lx[3], ly[3])
    x = x.view(lx[0], n_file, -1, lx[2], lx[3])
    y = y.view(ly[0], n_file, -1, ly[2], ly[3])
    x = torch.cat([x[:, :, :, 0:h, 0:w], y[:, :, :, 0:h, 0:w]], dim=2)
    x = x.view(lx[0], -1, h, w)
    return x
