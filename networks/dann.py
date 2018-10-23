
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from networks.cnn import CNN
from networks.blocks import UNetConvBlock, UNetUpSamplingBlock
from networks.unet import UNetEncoder, UNetDecoder, UNet, unet_from_encoder_decoder

# gradient reversal
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()

        return output, None

# original 2D unet encoder
class DANNUNetEncoder(nn.Module):

    def __init__(self, lambdas, in_channels=1, feature_maps=64, levels=4, group_norm=True):
        super(DANNUNetEncoder, self).__init__()

        self.lambdas = lambdas
        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.return_gradient_reversal = True

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
        self.center_conv = UNetConvBlock(2**(levels-1) * feature_maps, 2**levels * feature_maps, group_norm=group_norm)

    def forward(self, inputs):

        encoder_outputs = []  # for decoder skip connections

        reverse_outputs = []
        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features,'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(outputs)
            if self.lambdas[i] > 0:
                reverse_outputs.append(ReverseLayerF.apply(outputs)) # gradient reversal layer
            else:
                reverse_outputs.append(None)
            outputs = getattr(self.features,'pool%d' % (i + 1))(outputs)

        outputs = self.center_conv(outputs)
        if self.lambdas[self.levels] > 0:
            reverse_outputs.append(ReverseLayerF.apply(outputs)) # gradient reversal layer
        else:
            reverse_outputs.append(None)

        if self.return_gradient_reversal:
            return encoder_outputs, reverse_outputs, outputs
        else:
            return encoder_outputs, outputs

# original 2D unet decoder
class DANNUNetDecoder(nn.Module):

    def __init__(self, lambdas, out_channels=2, feature_maps=64, levels=4, skip_connections=True, group_norm=True):
        super(DANNUNetDecoder, self).__init__()

        self.lambdas = lambdas
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.skip_connections = skip_connections
        self.features = nn.Sequential()
        self.return_gradient_reversal = True

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

        reverse_outputs = []
        outputs = inputs
        for i in range(self.levels):
            if self.skip_connections:
                outputs = getattr(self.features,'upconv%d' % (i + 1))(encoder_outputs[i], outputs)  # also deals with concat
            else:
                outputs = getattr(self.features,'upconv%d' % (i + 1))(outputs)  # no concat
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            decoder_outputs.append(outputs)
            if self.lambdas[self.levels + i + 1] > 0:
                reverse_outputs.append(ReverseLayerF.apply(outputs)) # gradient reversal layer
            else:
                reverse_outputs.append(None)

        if self.return_gradient_reversal:
            return encoder_outputs, reverse_outputs, outputs
        else:
            return encoder_outputs, outputs

# DANN U-Net model
class UNet_DANN(nn.Module):

    def __init__(self, n, lambdas, in_channels=1, out_channels=2, feature_maps=64, levels=4, group_norm=True):
        super(UNet_DANN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.lambdas = lambdas

        # encoder
        self.encoder = DANNUNetEncoder(lambdas=lambdas, in_channels=in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

        # segmentation decoder
        self.segmentation_decoder = DANNUNetDecoder(lambdas=lambdas, out_channels=out_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

        # domain classifiers
        self.encoder_classifiers = []
        self.decoder_classifiers = []
        for l in range(levels):
            nl = n // 2**l  # size of the data at this level
            fml = feature_maps * 2**l  # number of feature maps at this level
            nb = 2+(levels-l)  # number of convolution-pool blocks in the discriminator
            conv_channels = np.ones(nb, dtype=int)*48
            fc_channels = [48, 24, 2]   # number of feature maps in each fully connected layer
            # encoder classifiers
            if lambdas[l] > 0:
                self.encoder_classifiers.append(CNN(input_size=(fml, nl, nl), conv_channels=conv_channels, fc_channels=fc_channels, group_norm=group_norm).cuda())
            else:
                self.encoder_classifiers.append(None)
            # decoder classifiers
            if lambdas[levels+l+1] > 0:
                self.decoder_classifiers.append(CNN(input_size=(fml, nl, nl), conv_channels=conv_channels, fc_channels=fc_channels, group_norm=group_norm).cuda())
            else:
                self.decoder_classifiers.append(None)
        # encoded classifier
        if lambdas[levels] > 0:
            nl = n // 2 ** levels  # size of the data at this level
            fml = feature_maps * 2 ** levels  # number of feature maps at this level
            nb = 2  # number of convolution-pool blocks in the discriminator
            conv_channels = np.ones(nb, dtype=int) * 48
            fc_channels = [48, 24, 2]  # number of feature maps in each fully connected layer
            self.encoded_classifier = CNN(input_size=(fml, nl, nl), conv_channels=conv_channels, fc_channels=fc_channels, group_norm=group_norm).cuda()
        else:
            self.encoded_classifier = None

    def forward(self, inputs):

        # encoder
        encoder_outputs, encoder_outputs_reversed, encoded = self.encoder(inputs)

        # segmentation decoder
        decoder_outputs, decoder_outputs_reversed, segmentation_outputs = self.segmentation_decoder(encoded, encoder_outputs)

        # domain predictions
        domain_outputs = [None] * (2*self.levels+1)
        for l in range(self.levels):
            # encoder classifiers
            if self.lambdas[l] > 0:
                domain_outputs[l] = self.encoder_classifiers[l](encoder_outputs_reversed[l])
            # decoder classifiers
            if self.lambdas[self.levels+l+1] > 0:
                domain_outputs[self.levels+l+1] = self.decoder_classifiers[l](decoder_outputs_reversed[self.levels+l+1])
        # encoded classifier
        if self.lambdas[self.levels] > 0:
            domain_outputs[self.levels] = self.encoded_classifier(encoder_outputs_reversed[self.levels])

        return domain_outputs, segmentation_outputs

    # returns the basic segmentation network
    def get_segmentation_net(self):

        encoder = self.encoder
        segmentation_decoder = self.segmentation_decoder

        encoder.return_gradient_reversal = False
        segmentation_decoder.return_gradient_reversal = False

        return unet_from_encoder_decoder(encoder, segmentation_decoder)

    # trains the network for one epoch
    def train_epoch(self, loader_src, loader_tar, loss_seg_fn, lambdas, optimizer, epoch, print_stats=1, writer=None,
                    write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average losses during the epoch
        loss_seg_src_cum = 0.0
        losses_dom_src_cum = np.zeros(len(lambdas))
        loss_dom_src_cum = 0.0

        losses_dom_tar_cum = np.zeros(len(lambdas))
        loss_dom_tar_cum = 0.0

        loss_cum = 0.0
        cnt = 0

        # domain loss function
        loss_dom_fn = nn.CrossEntropyLoss()

        # list of the target data
        list_tar = list(enumerate(loader_tar))

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src, y_src = data[0].cuda(), data[1].cuda()
            x_tar = list_tar[i][1].cuda()

            # prep domain labels
            domain_label_src = torch.zeros((x_src.size(0))).cuda().long()
            domain_label_tar = torch.ones((x_tar.size(0))).cuda().long()

            # zero the gradient buffers
            optimizer.zero_grad()

            # forward prop
            domain_pred_src, y_src_pred = self(x_src)
            domain_pred_tar, y_tar_pred = self(x_tar)

            # compute segmentation loss on source predictions
            loss_seg_src = loss_seg_fn(y_src_pred, y_src)
            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()

            # compute domain loss
            loss_dom_src = 0
            loss_dom_tar = 0
            for j, lambda_dom in enumerate(lambdas):
                if lambda_dom>0:
                    # source loss
                    ld = lambda_dom * loss_dom_fn(domain_pred_src[j], domain_label_src)
                    losses_dom_src_cum[j] += ld.data.cpu().numpy()
                    loss_dom_src += ld
                    # target loss
                    ld = lambda_dom * loss_dom_fn(domain_pred_tar[j], domain_label_tar)
                    losses_dom_tar_cum[j] += ld.data.cpu().numpy()
                    loss_dom_tar += ld
            loss_dom_src_cum += loss_dom_src.data.cpu().numpy()
            loss_dom_tar_cum += loss_dom_tar.data.cpu().numpy()
            cnt += 1

            # compute total loss
            loss = loss_seg_src + loss_dom_src + loss_dom_tar
            loss_cum += loss.data.cpu().numpy()

            # backward prop
            loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss segmentation src: %.6f - Loss domain src: %.6f - Loss domain tar: %.6f - Total loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader_src.dataset), loss_seg_src, loss_dom_src, loss_dom_tar, loss))

        # don't forget to compute the average and print it
        loss_seg_src_avg = loss_seg_src_cum / cnt
        losses_dom_src_avg = losses_dom_src_cum / cnt
        loss_dom_src_avg = loss_dom_src_cum / cnt

        losses_dom_tar_avg = losses_dom_tar_cum / cnt
        loss_dom_tar_avg = loss_dom_tar_cum / cnt

        loss_avg = loss_cum / cnt

        print('[%s] Epoch %5d - Train averages: Loss segmentation src: %.6f - Loss domain src: %.6f - Loss domain tar: %.6f - Total loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_src_avg, loss_dom_src_avg, loss_dom_tar_avg, loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('train_src/loss_seg', loss_seg_src_avg, epoch)
            writer.add_scalar('train_src/loss_dom', loss_dom_src_avg, epoch)
            writer.add_scalar('train_tar/loss_dom', loss_dom_tar_avg, epoch)
            writer.add_scalar('train/loss', loss_avg, epoch)
            for k in range(len(losses_dom_src_avg)):
                if lambdas[k] > 0:
                    writer.add_scalar('train_src/loss_dom_'+str(k), losses_dom_src_avg[k], epoch)
                    writer.add_scalar('train_tar/loss_dom_' + str(k), losses_dom_tar_avg[k], epoch)

            if write_images:
                # write images
                x_src = vutils.make_grid(x_src, normalize=True, scale_each=True)
                x_tar = vutils.make_grid(x_tar, normalize=True, scale_each=True)
                y_src = vutils.make_grid(y_src, normalize=y_src.max()-y_src.min()>0, scale_each=True)
                y_src_pred = vutils.make_grid(F.softmax(y_src_pred, dim=1)[:,1:2,:,:].data, normalize=y_src_pred.max()-y_src_pred.min()>0, scale_each=True)
                y_tar_pred = vutils.make_grid(F.softmax(y_tar_pred, dim=1)[:,1:2,:,:].data, normalize=y_tar_pred.max()-y_tar_pred.min()>0, scale_each=True)
                writer.add_image('train_src/x', x_src, epoch)
                writer.add_image('train_tar/x', x_tar, epoch)
                writer.add_image('train_src/y', y_src, epoch)
                writer.add_image('train_src/y_pred', y_src_pred, epoch)
                writer.add_image('train_tar/y_pred', y_tar_pred, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader_src, loader_tar, loss_fn, lambdas, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        loss_seg_src_cum = 0.0
        losses_dom_src_cum = np.zeros(len(lambdas))
        loss_dom_src_cum = 0.0
        losses_dom_tar_cum = np.zeros(len(lambdas))
        loss_dom_tar_cum = 0.0
        loss_tar_cum = 0.0
        cnt = 0

        # domain loss function
        loss_dom_fn = nn.CrossEntropyLoss()

        # list of the target data
        list_tar = list(enumerate(loader_tar))

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src, y_src = data[0].cuda(), data[1].cuda()
            x_tar, y_tar = list_tar[i][1][0].cuda(), list_tar[i][1][1].cuda()

            # prep domain labels
            domain_label_src = torch.zeros((x_src.size(0))).cuda().long()
            domain_label_tar = torch.ones((x_tar.size(0))).cuda().long()

            # forward prop
            domain_pred_src, y_src_pred = self(x_src)
            domain_pred_tar, y_tar_pred = self(x_tar)

            # compute segmentation loss on source predictions
            loss_seg_src = loss_fn(y_src_pred, y_src)
            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()

            # compute domain loss
            loss_dom_src = 0
            loss_dom_tar = 0
            for j, lambda_dom in enumerate(lambdas):
                if lambda_dom>0:
                    # source loss
                    ld = lambda_dom * loss_dom_fn(domain_pred_src[j], domain_label_src)
                    losses_dom_src_cum[j] += ld.data.cpu().numpy()
                    loss_dom_src += ld
                    # target loss
                    ld = lambda_dom * loss_dom_fn(domain_pred_tar[j], domain_label_tar)
                    losses_dom_tar_cum[j] += ld.data.cpu().numpy()
                    loss_dom_tar += ld
            loss_dom_src_cum += loss_dom_src.data.cpu().numpy()
            loss_dom_tar_cum += loss_dom_tar.data.cpu().numpy()
            cnt += 1

            # compute total loss
            loss = loss_seg_src + loss_dom_src + loss_dom_tar
            loss_cum += loss.data.cpu().numpy()

        # test loss on target
        cnt = 0
        for i, data in enumerate(loader_tar):

            # get the inputs
            x, y = data[0].cuda(), data[1].cuda()

            # forward prop
            _, y_pred = self(x)

            # compute loss
            loss = loss_fn(y_pred, y)
            loss_tar_cum += loss.data.cpu().numpy()
            cnt += 1

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        loss_tar_avg = loss_tar_cum / cnt
        print('[%s] Epoch %5d - Average test loss: %.6f - Average test loss on target: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg, loss_tar_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('test/loss_target', loss_avg, epoch)
            writer.add_scalar('test/loss_target', loss_tar_avg, epoch)

            if write_images:
                # write images
                x_tar = vutils.make_grid(x_tar, normalize=True, scale_each=True)
                y_tar = vutils.make_grid(y_tar, normalize=y_tar.max() - y_tar.min() > 0, scale_each=True)
                y_tar_pred = vutils.make_grid(F.softmax(y_tar_pred, dim=1)[:, 1:2, :, :].data,
                                              normalize=y_tar_pred.max() - y_tar_pred.min() > 0, scale_each=True)
                writer.add_image('test/x_tar', x_tar, epoch)
                writer.add_image('test/y_tar', y_tar, epoch)
                writer.add_image('test/y_tar_pred', y_tar_pred, epoch)

        return loss_avg, loss_tar_avg

    # trains the network
    def train_net(self, train_loader_source, test_loader_source, train_loader_target, test_loader_target, loss_fn, lambdas, optimizer,
                  scheduler=None, epochs=100, test_freq=1, print_stats=1, log_dir=None, write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader_src=train_loader_source, loader_tar=train_loader_target, loss_seg_fn=loss_fn,
                             lambdas=lambdas, optimizer=optimizer, epoch=epoch,
                             print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss, test_loss_tar = self.test_epoch(loader_src=test_loader_source, loader_tar=test_loader_target,
                                                           loss_fn=loss_fn, lambdas=lambdas, epoch=epoch,
                                                           writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss_tar < test_loss_min:
                    test_loss_min = test_loss_tar
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()