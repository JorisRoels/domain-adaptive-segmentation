
import torch.nn as nn
import torch.nn.functional as F

from networks.blocks import ConvBatchNormRelu2D, ConvGroupNormRelu2D

# classical convolutional neural network implementation
class CNN(nn.Module):

    def __init__(self, input_size, conv_channels, fc_channels, kernel_size=3, group_norm=True):
        super(CNN, self).__init__()

        self.input_size = input_size
        self.conv_channels = conv_channels
        self.fc_channels = fc_channels
        self.kernel_size = kernel_size

        self.conv_features = nn.Sequential()
        self.fc_features = nn.Sequential()

        # convolutional layers
        in_channels = input_size[0]
        data_size = input_size[1]
        for i, out_channels in enumerate(conv_channels):
            if group_norm:
                self.conv_features.add_module('conv%d' % (i + 1), ConvGroupNormRelu2D(in_channels, out_channels, kernel_size=kernel_size))
            else:
                self.conv_features.add_module('conv%d' % (i + 1), ConvBatchNormRelu2D(in_channels, out_channels, kernel_size=kernel_size))
            in_channels = out_channels
            data_size /= 2

        # full connections
        in_channels = conv_channels[-1]*data_size*data_size
        for i, out_channels in enumerate(fc_channels):
            if i==len(fc_channels)-1:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels))
            else:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels),
                                   nn.BatchNorm1d(out_channels),
                                   nn.ReLU())
            self.fc_features.add_module('linear%d' % (i + 1), fc)
            in_channels = out_channels

    def forward(self, inputs):

        outputs = inputs
        for i in range(len(self.conv_channels)):
            outputs = getattr(self.conv_features, 'conv%d' % (i + 1))(outputs)
            outputs = F.max_pool2d(outputs, kernel_size=2)

        outputs = outputs.view(outputs.size(0),-1)
        for i in range(len(self.fc_channels)):
            outputs = getattr(self.fc_features, 'linear%d' % (i + 1))(outputs)

        return outputs