
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from networks.blocks import UNetConvBlock, UNetUpSamplingBlock

# original 2D unet encoder
class UNetEncoder(nn.Module):

    def __init__(self, in_channels=1, feature_maps=64, levels=4, group_norm=True):
        super(UNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps

            # convolutional block
            conv_block = UNetConvBlock(in_features, out_features, group_norm=group_norm)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

        # center (lowest) block
        self.center_conv = UNetConvBlock(2**(levels-1) * feature_maps, 2**levels * feature_maps)

    def forward(self, inputs):

        encoder_outputs = []  # for decoder skip connections

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features,'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            outputs = getattr(self.features,'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)

        return encoder_outputs, outputs

# original 2D unet decoder
class UNetDecoder(nn.Module):

    def __init__(self, out_channels=2, feature_maps=64, levels=4, skip_connections=True, group_norm=True):
        super(UNetDecoder, self).__init__()

        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.features = nn.Sequential()

        for i in range(levels):

            # upsampling block
            upconv = UNetUpSamplingBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=True)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            # convolutional block
            if skip_connections:
                conv_block = UNetConvBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, group_norm=group_norm)
            else:
                conv_block = UNetConvBlock(2**(levels-i-1) * feature_maps, 2**(levels-i-1) * feature_maps, group_norm=group_norm)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        # output layer
        self.output = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, encoder_outputs):

        decoder_outputs = []

        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                outputs = getattr(self.features,'upconv%d' % (i + 1))(encoder_outputs[i], outputs)  # also deals with concat
            else:
                outputs = getattr(self.features,'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)

        return decoder_outputs, self.output(outputs)

# original 2D unet model
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, group_norm=True):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels

        # contractive path
        self.encoder = UNetEncoder(in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)
        # expansive path
        self.decoder = UNetDecoder(out_channels, feature_maps=feature_maps, levels=levels, skip_connections=True, group_norm=group_norm)

    def forward(self, inputs):

        # contractive path
        encoder_outputs, final_output = self.encoder(inputs)

        # expansive path
        decoder_outputs, outputs = self.decoder(final_output, encoder_outputs)

        return outputs

    # returns the basic segmentation network
    def get_segmentation_net(self):
        return self

    # trains the network for one epoch
    def train_epoch(self, loader, loss_fn, optimizer, epoch, print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        cnt = 0

        # start epoch
        for i, data in enumerate(loader):

            # get the inputs
            x, y = data[0].cuda(), data[1].cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            y_pred = self(x)

            # compute loss
            loss = loss_fn(y_pred, y)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics if necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader.dataset), loss))

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Average train loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('train/loss', loss_avg, epoch)

            if write_images:
                # write images
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                writer.add_image('train/x', x, epoch)
                writer.add_image('train/y', y, epoch)
                writer.add_image('train/y_pred', y_pred, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader_src, loader_tar, loss_fn, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        loss_tar_cum = 0.0
        cnt = 0

        # test loss
        for i, data in enumerate(loader_src):

            # get the inputs
            x, y = data[0].cuda(), data[1].cuda()

            # forward prop
            y_pred = self(x)

            # compute loss
            loss = loss_fn(y_pred, y)
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

        # test loss on target
        if loader_tar is not None:
            cnt = 0
            for i, data in enumerate(loader_tar):

                # get the inputs
                x, y = data[0].cuda(), data[1].cuda()

                # forward prop
                y_pred = self(x)

                # compute loss
                loss = loss_fn(y_pred, y)
                loss_tar_cum += loss.data.cpu().numpy()
                cnt += 1
        else:
            loss_tar_cum = loss_cum

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        loss_tar_avg = loss_tar_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f - Average test loss on target: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg, loss_tar_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss', loss_avg, epoch)
            writer.add_scalar('test/loss_target', loss_tar_avg, epoch)

            if write_images:
                # write images
                x = vutils.make_grid(x, normalize=True, scale_each=True)
                y = vutils.make_grid(y, normalize=y.max()-y.min()>0, scale_each=True)
                y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:,1:2,:,:].data, normalize=y_pred.max()-y_pred.min()>0, scale_each=True)
                writer.add_image('test/x', x, epoch)
                writer.add_image('test/y', y, epoch)
                writer.add_image('test/y_pred', y_pred, epoch)

        return loss_avg, loss_tar_avg

    # trains the network
    def train_net(self, train_loader_src, test_loader_src, test_loader_tar, loss_fn, optimizer, scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader=train_loader_src, loss_fn=loss_fn, optimizer=optimizer, epoch=epoch,
                             print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss, test_loss_tar = self.test_epoch(loader_src=test_loader_src, loader_tar=test_loader_tar, loss_fn=loss_fn, epoch=epoch, writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss_tar < test_loss_min:
                    test_loss_min = test_loss_tar
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()

# returns a unet model from a given encoder and decoder
def unet_from_encoder_decoder(encoder, decoder):

    net = UNet(in_channels=encoder.in_channels, out_channels=decoder.out_channels, feature_maps=encoder.feature_maps, levels=encoder.levels)

    net.encoder = encoder
    net.decoder = decoder

    return net