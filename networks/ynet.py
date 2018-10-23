
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from networks.unet import UNetEncoder, UNetDecoder, UNet, unet_from_encoder_decoder

# Y-Net model
class YNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, group_norm=True):
        super(YNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels

        # encoder
        self.encoder = UNetEncoder(in_channels=in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder(out_channels=out_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

        # reconstruction decoder
        self.reconstruction_decoder = UNetDecoder(out_channels=in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm, skip_connections=False)

    def forward(self, inputs):

        # encoder
        encoder_outputs, encoded = self.encoder(inputs)

        # segmentation decoder
        _, segmentation_outputs = self.segmentation_decoder(encoded, encoder_outputs)

        # reconstruction decoder
        _, reconstruction_outputs = self.reconstruction_decoder(encoded, encoder_outputs)

        return reconstruction_outputs, segmentation_outputs

    # returns the basic segmentation network
    def get_segmentation_net(self):

        return unet_from_encoder_decoder(self.encoder, self.segmentation_decoder)

    # trains the network for one epoch
    def train_epoch_ynet(self, loader_src, loader_tar,
                         lambda_reg, optimizer, loss_seg_fn, loss_rec_fn, epoch,
                         print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_seg_src_cum = 0.0
        loss_rec_src_cum = 0.0
        loss_rec_tar_cum = 0.0
        total_loss_cum = 0.0

        # list of the target data
        list_tar = list(enumerate(loader_tar))

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src, y_src = data[0].cuda(), data[1].cuda()
            x_tar = list_tar[i][1].cuda()

            # zero the gradient buffers
            self.zero_grad()

            # forward prop
            x_src_pred, y_src_pred = self(x_src)
            x_tar_pred, y_tar_pred = self(x_tar)

            # compute loss
            loss_seg_src = loss_seg_fn(y_src_pred, y_src)
            loss_rec_src = loss_rec_fn(x_src_pred, x_src)
            loss_rec_tar = loss_rec_fn(x_tar_pred, x_tar)
            total_loss = loss_seg_src + lambda_reg * (loss_rec_src + loss_rec_tar)
            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_rec_src_cum += loss_rec_src.data.cpu().numpy()
            loss_rec_tar_cum += loss_rec_tar.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('    [%5d] Jaccard src: %.6f - MSE src: %.6f - MSE tar: %.6f - Total loss: %.6f'
                      % (i, loss_seg_src, loss_rec_src, loss_rec_tar, total_loss))

        # don't forget to compute the average and print it
        loss_seg_src_avg = loss_seg_src_cum / len(loader_src.dataset)
        loss_rec_src_avg = loss_rec_src_cum / len(loader_src.dataset)
        loss_rec_tar_avg = loss_rec_tar_cum / len(loader_src.dataset)
        total_loss_avg = total_loss_cum / len(loader_src.dataset)
        print('TRAIN: Jaccard src: %.6f - MSE src: %.6f - MSE tar: %.6f - Total loss: %.6f'
              % (loss_seg_src_avg, loss_rec_src_avg, loss_rec_tar_avg, total_loss_avg))

        # scalars
        writer.add_scalar('train/jaccard_src', loss_seg_src_avg, epoch)
        writer.add_scalar('train/mse_src', loss_rec_src_avg, epoch)
        writer.add_scalar('train/mse_tar', loss_rec_tar_avg, epoch)
        writer.add_scalar('train/total_loss', total_loss_avg, epoch)

        # log everything
        if writer is not None and write_images:

            # images
            xs = vutils.make_grid(x_src, normalize=True, scale_each=True)
            xt = vutils.make_grid(x_tar, normalize=True, scale_each=True)
            ys = vutils.make_grid(y_src, normalize=y_src.max() - y_src.min() > 0, scale_each=True)
            xs_pred = vutils.make_grid(x_src_pred.data, normalize=True, scale_each=True)
            xt_pred = vutils.make_grid(x_tar_pred.data, normalize=True, scale_each=True)
            ys_pred = vutils.make_grid(F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data,
                                       normalize=y_src_pred.max() - y_src_pred.min() > 0, scale_each=True)
            yt_pred = vutils.make_grid(F.softmax(y_tar_pred, dim=1)[:, 1:2, :, :].data,
                                       normalize=y_tar_pred.max() - y_tar_pred.min() > 0, scale_each=True)
            writer.add_image('train/xs', xs, epoch)
            writer.add_image('train/xt', xt, epoch)
            writer.add_image('train/ys', ys, epoch)
            writer.add_image('train/xs_pred', xs_pred, epoch)
            writer.add_image('train/xt_pred', xt_pred, epoch)
            writer.add_image('train/ys_pred', ys_pred, epoch)
            writer.add_image('train/yt_pred', yt_pred, epoch)

        return total_loss_avg

    # tests the network over one epoch
    def test_epoch_ynet(self, loader_src, loader_tar, lambda_reg, loss_seg_fn, loss_rec_fn, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss during the epoch
        loss_seg_src_cum = 0.0
        loss_seg_tar_cum = 0.0
        loss_rec_src_cum = 0.0
        loss_rec_tar_cum = 0.0
        total_loss_cum = 0.0

        # list of the target data
        list_tar = list(enumerate(loader_tar))

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src, y_src = data[0].cuda(), data[1].cuda()
            x_tar, y_tar = list_tar[i][1][0].cuda(), list_tar[i][1][1].cuda()

            # forward prop
            x_src_pred, y_src_pred = self(x_src)
            x_tar_pred, y_tar_pred = self(x_tar)

            # compute loss
            loss_seg_src = loss_seg_fn(y_src_pred, y_src)
            loss_seg_tar = loss_seg_fn(y_tar_pred, y_tar)
            loss_rec_src = loss_rec_fn(x_src_pred, x_src)
            loss_rec_tar = loss_rec_fn(x_tar_pred, x_tar)
            total_loss = loss_seg_src + lambda_reg * (loss_rec_src + loss_rec_tar)
            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_rec_src_cum += loss_rec_src.data.cpu().numpy()
            loss_rec_tar_cum += loss_rec_tar.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()

        # don't forget to compute the average and print it
        loss_seg_src_avg = loss_seg_src_cum / len(loader_src.dataset)
        loss_seg_tar_avg = loss_seg_tar_cum / len(loader_tar.dataset)
        loss_rec_src_avg = loss_rec_src_cum / len(loader_src.dataset)
        loss_rec_tar_avg = loss_rec_tar_cum / len(loader_tar.dataset)
        total_loss_avg = total_loss_cum / len(loader_src.dataset)
        print('TEST: Jaccard src: %.6f - Jaccard tar: %.6f - MSE src: %.6f - MSE tar: %.6f - Total loss: %.6f'
              % (loss_seg_src_avg, loss_seg_tar_avg, loss_rec_src_avg, loss_rec_tar_avg, total_loss_avg))

        # scalars
        writer.add_scalar('test/jaccard_src', loss_seg_src_avg, epoch)
        writer.add_scalar('test/jaccard_tar', loss_seg_tar_avg, epoch)
        writer.add_scalar('test/mse_src', loss_rec_src_avg, epoch)
        writer.add_scalar('test/mse_tar', loss_rec_tar_avg, epoch)
        writer.add_scalar('test/total_loss', total_loss_avg, epoch)

        # log everything
        if writer is not None and write_images:

            # images
            xs = vutils.make_grid(x_src, normalize=True, scale_each=True)
            xt = vutils.make_grid(x_tar, normalize=True, scale_each=True)
            ys = vutils.make_grid(y_src, normalize=y_src.max() - y_src.min() > 0, scale_each=True)
            yt = vutils.make_grid(y_tar, normalize=y_tar.max() - y_tar.min() > 0, scale_each=True)
            xs_pred = vutils.make_grid(x_src_pred.data, normalize=True, scale_each=True)
            xt_pred = vutils.make_grid(x_tar_pred.data, normalize=True, scale_each=True)
            ys_pred = vutils.make_grid(F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data,
                                       normalize=y_src_pred.max() - y_src_pred.min() > 0, scale_each=True)
            yt_pred = vutils.make_grid(F.softmax(y_tar_pred, dim=1)[:, 1:2, :, :].data,
                                       normalize=y_tar_pred.max() - y_tar_pred.min() > 0, scale_each=True)
            writer.add_image('test/xs', xs, epoch)
            writer.add_image('test/xt', xt, epoch)
            writer.add_image('test/ys', ys, epoch)
            writer.add_image('test/yt', yt, epoch)
            writer.add_image('test/xs_pred', xs_pred, epoch)
            writer.add_image('test/xt_pred', xt_pred, epoch)
            writer.add_image('test/ys_pred', ys_pred, epoch)
            writer.add_image('test/yt_pred', yt_pred, epoch)

        return total_loss_avg, loss_seg_tar_avg

    # trains the network
    def train_net(self, train_loader_source, train_loader_target, test_loader_source, test_loader_target,
                  lambda_reg, optimizer, loss_seg_fn, loss_rec_fn,
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
            self.train_epoch_ynet(loader_src=train_loader_source, loader_tar=train_loader_target,
                                  lambda_reg=lambda_reg, optimizer=optimizer, loss_seg_fn=loss_seg_fn, loss_rec_fn=loss_rec_fn, epoch=epoch,
                                  print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss, test_loss_tar = self.test_epoch_ynet(loader_src=test_loader_source, loader_tar=test_loader_target,
                                                 lambda_reg=lambda_reg, loss_seg_fn=loss_seg_fn, loss_rec_fn=loss_rec_fn, epoch=epoch,
                                                 writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss_tar < test_loss_min:
                    test_loss_min = test_loss_tar
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()