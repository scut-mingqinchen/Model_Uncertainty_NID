import torch.nn as nn
import torch.nn.functional as F
from model.common import conv_layer, concat


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        self.in_channels = args.c
        self.out_channels = args.c
        self.n_file = args.n_file
        self.conv1 = conv_layer(self.in_channels, 48, self.n_file, p=0.075)
        self.conv2 = conv_layer(48, 48, self.n_file, p=0.075)
        self.conv3 = conv_layer(48, 48, self.n_file, p=0.15)
        self.conv4 = conv_layer(48, 48, self.n_file, p=0.225)
        self.conv5 = conv_layer(48, 48, self.n_file, p=0.3)
        self.conv6 = conv_layer(48, 48, self.n_file, p=0.375)
        self.conv7 = conv_layer(48, 48, self.n_file, p=0.45)
        self.conv8 = conv_layer(96, 96, self.n_file, p=0.375)
        self.conv9 = conv_layer(96, 96, self.n_file, p=0.375)
        self.conv10 = conv_layer(144, 96, self.n_file, p=0.3)
        self.conv11 = conv_layer(96, 96, self.n_file, p=0.3)
        self.conv12 = conv_layer(144, 96, self.n_file, p=0.225)
        self.conv13 = conv_layer(96, 96, self.n_file, p=0.225)
        self.conv14 = conv_layer(144, 96, self.n_file, p=0.15)
        self.conv15 = conv_layer(96, 96, self.n_file, p=0.15)
        self.conv16 = conv_layer(96 + self.in_channels, 64, self.n_file, p=0.075)
        self.conv17 = conv_layer(64, 32, self.n_file, p=0.075)
        self.conv18 = conv_layer(32, self.in_channels, self.n_file, act=args.nn_last_act, p=0.075)

    def forward(self, x):
        skips = [x]
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)
        skips.append(x)

        x = self.conv3(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)
        skips.append(x)

        x = self.conv4(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)
        skips.append(x)

        x = self.conv5(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)
        skips.append(x)

        x = self.conv6(x)
        x = nn.MaxPool2d(kernel_size=2, ceil_mode=True)(x)
        x = self.conv7(x)

        # -----------------------------------------------
        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv8(x)
        x = self.conv9(x)

        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv10(x)
        x = self.conv11(x)

        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv12(x)
        x = self.conv13(x)

        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv14(x)
        x = self.conv15(x)

        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = concat(x, skips.pop(), self.n_file)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x
