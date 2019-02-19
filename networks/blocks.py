
import torch
import torch.nn as nn

from networks.layers import GroupNorm2d

# 2D convolution layer with batch normalization and relu activation
class ConvBatchNormRelu2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding='SAME', bias=True, dilation=1):
        super(ConvBatchNormRelu2D, self).__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else: # VALID (no) padding
            p = 0

        self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation),
                                  nn.BatchNorm2d(int(out_channels)),
                                  nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 2D convolution layer with group normalization and relu activation
class ConvGroupNormRelu2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  stride=1, padding='SAME', bias=True, dilation=1):
        super(ConvGroupNormRelu2D, self).__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else: # VALID (no) padding
            p = 0

        self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation),
                                  GroupNorm2d(int(out_channels)),
                                  nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return outputs

# 2D convolution block of the classical unet
class UNetConvBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', group_norm=True):
        super(UNetConvBlock2D, self).__init__()

        if group_norm:
            self.conv1 = ConvGroupNormRelu2D(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            self.conv2 = ConvGroupNormRelu2D(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        else:
            self.conv1 = ConvBatchNormRelu2D(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            self.conv2 = ConvBatchNormRelu2D(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

# 2D upsampling block of the classical unet:
# upsamples the input and concatenates with another input
class UNetUpSamplingBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock2D, self).__init__()

        if deconv: # use transposed convolution
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        else: # use bilinear upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, *arg):
        if len(arg) == 2:
            return self.forward_concat(arg[0], arg[1])
        else:
            return self.forward_standard(arg[0])

    def forward_concat(self, inputs1, inputs2):

        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):

        return self.up(inputs)