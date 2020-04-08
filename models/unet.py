import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn

__all__ = ['UNet', 'UPPNet']


class Down(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Down(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Down(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Down(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Down(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Down(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = Down(
            nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = Down(
            nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = Down(
            nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = Down(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class UPPNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Down(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = Down(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = Down(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Down(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Down(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = Down(
            nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = Down(
            nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = Down(
            nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = Down(
            nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = Down(
            nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = Down(
            nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = Down(
            nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = Down(
            nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = Down(
            nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = Down(
            nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


def Conv1(input_channels, nb_filter, act_fn):
    model = nn.Sequential(
        nn.Conv2d(input_channels, nb_filter, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(nb_filter),
        act_fn,
    )
    return model


def ConvTranspose(input_channels, nb_filter, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(input_channels, nb_filter, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(nb_filter),
        act_fn,
    )
    return model


def Maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def Conv3(input_channels, nb_filter, act_fn):
    model = nn.Sequential(
        Conv1(input_channels, nb_filter, act_fn),
        Conv1(nb_filter, nb_filter, act_fn),
        nn.Conv2d(nb_filter, nb_filter, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(nb_filter),
    )
    return model


class ConvResidual(nn.Module):

    def __init__(self, input_channels, nb_filter, act_fn):
        super(ConvResidual, self).__init__()

        self.conv_1 = Conv1(input_channels, nb_filter, act_fn)
        self.conv_2 = Conv3(nb_filter, nb_filter, act_fn)
        self.conv_3 = Conv1(nb_filter, nb_filter, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class FusionNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, nb_filter=32, **kwargs):
        super().__init__()
        act_fn_1 = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        self.down_1 = ConvResidual(input_channels, nb_filter, act_fn_1)
        self.pool_1 = Maxpool()
        self.down_2 = ConvResidual(nb_filter, nb_filter*2, act_fn_1)
        self.pool_2 = Maxpool()
        self.down_3 = ConvResidual(nb_filter*2, nb_filter*4, act_fn_1)
        self.pool_3 = Maxpool()
        self.down_4 = ConvResidual(nb_filter*4, nb_filter*8, act_fn_1)
        self.pool_4 = Maxpool()

        # bridge
        self.bridge = ConvResidual(nb_filter*8, nb_filter*16, act_fn_1)

        # decoder
        self.deconv_1 = ConvTranspose(nb_filter*16, nb_filter*8, act_fn_2)
        self.up_1 = ConvResidual(nb_filter*8, nb_filter*8, act_fn_2)
        self.deconv_2 = ConvTranspose(nb_filter*8, nb_filter*4, act_fn_2)
        self.up_2 = ConvResidual(nb_filter*4, nb_filter*4, act_fn_2)
        self.deconv_3 = ConvTranspose(nb_filter*4, nb_filter*2, act_fn_2)
        self.up_3 = ConvResidual(nb_filter*2, nb_filter*2, act_fn_2)
        self.deconv_4 = ConvTranspose(nb_filter*2, nb_filter, act_fn_2)
        self.up_4 = ConvResidual(nb_filter, nb_filter, act_fn_2)

        # output
        self.out = nn.Conv2d(nb_filter, num_classes, kernel_size=3, stride=1, padding=1)

        # initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(0.0, 0.02)
        #         m.bias.data.fill_(0)

        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(1.0, 0.02)
        #         m.bias.data.fill_(0)

    def forward(self, inputs):
        down_1 = self.down_1(inputs)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)/2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)/2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)/2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)/2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)
        return out
  
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.5),
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             # nn.Dropout(0.5),
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels),
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class UpCat(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)


#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#         diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_channels=1, n_classes=1, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = UpCat(1024, 512, bilinear)
#         self.up2 = UpCat(512, 256, bilinear)
#         self.up3 = UpCat(256, 128, bilinear)
#         self.up4 = UpCat(128, 64 * factor, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
