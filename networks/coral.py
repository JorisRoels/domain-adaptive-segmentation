
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from networks.unet import UNetEncoder, UNetDecoder, UNet, unet_from_encoder_decoder

# CORAL U-Net model
class UNet_CORAL(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, group_norm=True):
        super(UNet_CORAL, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels

        # encoder
        self.encoder = UNetEncoder(in_channels=in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder(out_channels=out_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

    def forward(self, inputs):

        # encoder
        encoder_outputs, encoded = self.encoder(inputs)

        # segmentation decoder
        decoder_outputs, segmentation_outputs = self.segmentation_decoder(encoded, encoder_outputs)

        # group feature activations
        features = []
        for encoder_output in encoder_outputs:
            features.append(encoder_output)
        features.append(encoded)
        for decoder_output in decoder_outputs:
            features.append(decoder_output)

        return features, segmentation_outputs

    # returns the basic segmentation network
    def get_segmentation_net(self):

        return unet_from_encoder_decoder(self.encoder, self.segmentation_decoder)

    # trains the network for one epoch
    def train_epoch(self, loader_src, loader_tar, loss_fn, lambdas, optimizer, epoch, print_stats=1, writer=None,
                    write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        losses_coral_cum = np.zeros(len(lambdas))
        total_loss_coral_cum = 0.0
        total_loss_cum = 0.0
        cnt = 0

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
            src_features, y_src_pred = self(x_src)
            tar_features, y_tar_pred = self(x_tar)

            # compute loss
            loss = loss_fn(y_src_pred, y_src)
            total_loss_coral = 0
            for j, lambda_coral in enumerate(lambdas):
                if lambda_coral>0:
                    f_j_src = src_features[j]
                    f_j_tar = tar_features[j]
                    sz = f_j_src.size()
                    for s in range(sz[1]):
                        loss_coral = coral(f_j_src[:,s,:,:].view(sz[0],sz[2]*sz[3]),
                                           f_j_tar[:,s,:,:].view(sz[0],sz[2]*sz[3]))
                        losses_coral_cum[j] = losses_coral_cum[j] + loss_coral.data.cpu().numpy()
                        total_loss_coral = total_loss_coral + lambda_coral*loss_coral
            total_loss = loss + total_loss_coral
            loss_cum += loss.data.cpu().numpy()
            total_loss_coral_cum += total_loss_coral.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss: %.6f - Loss CORAL: %.6f - Total loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader_src.dataset), loss, total_loss_coral, total_loss))

        # don't forget to compute the average and print it
        loss_avg = loss_cum / cnt
        losses_coral_avg = losses_coral_cum / cnt
        total_loss_coral_avg = total_loss_coral_cum / cnt
        total_loss_avg = total_loss_cum / cnt
        print('[%s] Epoch %5d - Train averages: Loss: %.6f - Loss CORAL: %.6f - Total loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_avg, total_loss_coral_avg, total_loss_avg))

        # log everything
        if writer is not None:

            # always log scalars
            writer.add_scalar('train/loss', loss_avg, epoch)
            for i in range(len(lambdas)):
                writer.add_scalar('train/loss_coral_level_' + str(i), losses_coral_avg[i], epoch)
            writer.add_scalar('train/total_loss_coral', total_loss_coral_avg, epoch)
            writer.add_scalar('train/total_loss', total_loss_avg, epoch)

            if write_images:
                # write images
                x_src = vutils.make_grid(x_src, normalize=True, scale_each=True)
                x_tar = vutils.make_grid(x_tar, normalize=True, scale_each=True)
                y_src = vutils.make_grid(y_src, normalize=y_src.max() - y_src.min() > 0, scale_each=True)
                y_src_pred = vutils.make_grid(F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data,
                                              normalize=y_src_pred.max() - y_src_pred.min() > 0, scale_each=True)
                writer.add_image('train/x_src', x_src, epoch)
                writer.add_image('train/x_tar', x_tar, epoch)
                writer.add_image('train/y_src', y_src, epoch)
                writer.add_image('train/y_src_pred', y_src_pred, epoch)

        return total_loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader_src, loader_tar, loss_fn, lambdas, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss during the epoch
        loss_cum = 0.0
        losses_coral_cum = np.zeros(len(lambdas))
        total_loss_coral_cum = 0.0
        total_loss_cum = 0.0
        loss_tar_cum = 0.0
        cnt = 0

        # list of the target data
        list_tar = list(enumerate(loader_tar))

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src, y_src = data[0].cuda(), data[1].cuda()
            x_tar, y_tar = list_tar[i][1][0].cuda(), list_tar[i][1][1].cuda()

            # forward prop
            src_features, y_src_pred = self(x_src)
            tar_features, y_tar_pred = self(x_tar)

            # compute loss
            loss = loss_fn(y_src_pred, y_src)
            total_loss_coral = 0
            for j, lambda_coral in enumerate(lambdas):
                if lambda_coral>0:
                    f_j_src = src_features[j]
                    f_j_tar = tar_features[j]
                    sz = f_j_src.size()
                    for s in range(sz[1]):
                        loss_coral = coral(f_j_src[:,s,:,:].view(sz[0],sz[2]*sz[3]),
                                           f_j_tar[:,s,:,:].view(sz[0],sz[2]*sz[3]))
                        losses_coral_cum[j] = losses_coral_cum[j] + loss_coral.data.cpu().numpy()
                        total_loss_coral = total_loss_coral + lambda_coral*loss_coral
            total_loss = loss + total_loss_coral
            loss_cum += loss.data.cpu().numpy()
            total_loss_coral_cum += total_loss_coral.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()
            cnt += 1

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
        loss_avg = total_loss_cum / cnt
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
            self.train_epoch(loader_src=train_loader_source, loader_tar=train_loader_target, loss_fn=loss_fn,
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

def coral(source, target):

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.mm(xm.view(xm.size(1),xm.size(0)), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.mm(xmt.view(xmt.size(1),xmt.size(0)), xmt)

    # frobenius norm between source and target
    loss = torch.mean(torch.pow(xc - xct, 2))

    return loss