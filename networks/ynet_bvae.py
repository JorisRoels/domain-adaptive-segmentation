
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from networks.unet import unet_from_encoder_decoder
from networks.unet_bvae import UNetEncoder, UNetDecoder
from util.losses import MSELoss, KLDLoss

def reparametrise(mu, logvar):

    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())

    return mu + std*eps

# Y-Net model
class YNetBVAE(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, group_norm=False, lambda_rec=1e-3, beta=1):
        super(YNetBVAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.lambda_rec = lambda_rec
        self.beta = beta

        # encoder
        self.encoder = UNetEncoder(in_channels=in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder(out_channels=out_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

        # reconstruction decoder
        self.reconstruction_decoder = UNetDecoder(out_channels=in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm, skip_connections=False)

    def forward(self, inputs):

        # encoder
        encoder_outputs, encoded = self.encoder(inputs)

        # reparameterisation trick
        n_features_split = encoded.size(1)//2
        mu = encoded[:, :n_features_split]
        logvar = encoded[:, n_features_split:]
        z = reparametrise(mu, logvar)

        # segmentation decoder
        _, segmentation_outputs = self.segmentation_decoder(z, encoder_outputs)

        # reconstruction decoder
        _, reconstruction_outputs = self.reconstruction_decoder(z, encoder_outputs)

        return reconstruction_outputs, segmentation_outputs, mu, logvar

    # returns the basic segmentation network
    def get_segmentation_net(self):

        return unet_from_encoder_decoder(self.encoder, self.segmentation_decoder)

    # trains the network for one epoch
    def train_epoch(self, loader_src, loader_tar,
                    optimizer, loss_seg_fn, loss_rec_fn, epoch,
                    print_stats=1, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.train()

        # define losses
        kld_loss = KLDLoss()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_rec_cum = 0.0
        loss_kld_cum = 0.0
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
            x_src_pred, y_src_pred, mu_src, logvar_src = self(x_src)
            x_tar_pred, y_tar_pred, mu_tar, logvar_tar = self(x_tar)

            # compute loss
            loss_seg = loss_seg_fn(y_src_pred, y_src)
            loss_rec = 0.5 * (loss_rec_fn(x_src_pred, x_src) + loss_rec_fn(x_tar_pred, x_tar))
            mu = torch.cat((mu_src, mu_tar), dim=0)
            logvar = torch.cat((logvar_src, logvar_tar), dim=0)
            _, _, loss_kld = kld_loss(mu.view(mu.size(0), -1), logvar.view(logvar.size(0), -1))
            total_loss = loss_seg + self.lambda_rec * loss_rec + self.beta * loss_kld
            loss_seg_cum += loss_seg.data.cpu().numpy()
            loss_rec_cum += loss_rec.data.cpu().numpy()
            loss_kld_cum += loss_kld.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()

            # backward prop
            total_loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss seg: %.6f - Loss rec: %.6f - Loss kld: %.6f - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader_src.dataset)/loader_src.batch_size, loss_seg, loss_rec, loss_kld, total_loss))

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / len(loader_src.dataset)
        loss_rec_avg = loss_rec_cum / len(loader_src.dataset)
        loss_kld_avg = loss_kld_cum / len(loader_src.dataset)
        total_loss_avg = total_loss_cum / len(loader_src.dataset)
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss kld: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, loss_kld_avg, total_loss_avg))

        # scalars
        writer.add_scalar('train/loss-seg', loss_seg_avg, epoch)
        writer.add_scalar('train/loss-rec', loss_rec_avg, epoch)
        writer.add_scalar('train/loss-kld', loss_kld_avg, epoch)
        writer.add_scalar('train/loss', total_loss_avg, epoch)

        # log everything
        if writer is not None and write_images:

            # images
            x = torch.cat((x_src, x_tar), dim=0)
            x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0)
            y_pred = torch.cat((y_src_pred, y_tar_pred), dim=0)
            x = vutils.make_grid(x, normalize=True, scale_each=True)
            ys = vutils.make_grid(y_src, normalize=y_src.max() - y_src.min() > 0, scale_each=True)
            x_pred = vutils.make_grid(x_pred.data, normalize=True, scale_each=True)
            y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:, 1:2, :, :].data,
                                      normalize=y_pred.max() - y_pred.min() > 0, scale_each=True)
            writer.add_image('train/x', x, epoch)
            writer.add_image('train/y', ys, epoch)
            writer.add_image('train/x-pred', x_pred, epoch)
            writer.add_image('train/y-pred', y_pred, epoch)

        return total_loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader_src, loader_tar, loss_seg_fn, loss_rec_fn, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # define losses
        kld_loss = KLDLoss()

        # keep track of the average loss during the epoch
        loss_seg_cum = 0.0
        loss_seg_tar_cum = 0.0
        loss_rec_cum = 0.0
        loss_kld_cum = 0.0
        total_loss_cum = 0.0

        # list of the target data
        list_tar = list(enumerate(loader_tar))

        # start epoch
        for i, data in enumerate(loader_src):

            # get the inputs
            x_src, y_src = data[0].cuda(), data[1].cuda()
            x_tar, y_tar = list_tar[i][1][0].cuda(), list_tar[i][1][1].cuda()

            # forward prop
            x_src_pred, y_src_pred, mu_src, logvar_src = self(x_src)
            x_tar_pred, y_tar_pred, mu_tar, logvar_tar = self(x_tar)

            # compute loss
            loss_seg = loss_seg_fn(y_src_pred, y_src)
            loss_seg_tar = loss_seg_fn(y_tar_pred, y_tar)
            loss_rec = 0.5 * (loss_rec_fn(x_src_pred, x_src) + loss_rec_fn(x_tar_pred, x_tar))
            mu = torch.cat((mu_src, mu_tar), dim=0)
            logvar = torch.cat((logvar_src, logvar_tar), dim=0)
            _, _, loss_kld = kld_loss(mu.view(mu.size(0), -1), logvar.view(logvar.size(0), -1))
            total_loss = loss_seg + self.lambda_rec * loss_rec + self.beta * loss_kld
            loss_seg_cum += loss_seg.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            loss_rec_cum += loss_rec.data.cpu().numpy()
            loss_kld_cum += loss_kld.data.cpu().numpy()
            total_loss_cum += total_loss.data.cpu().numpy()

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / len(loader_src.dataset)
        loss_seg_tar_avg = loss_seg_tar_cum / len(loader_src.dataset)
        loss_rec_avg = loss_rec_cum / len(loader_src.dataset)
        loss_kld_avg = loss_kld_cum / len(loader_src.dataset)
        total_loss_avg = total_loss_cum / len(loader_src.dataset)
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss rec: %.6f - Loss kld: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, loss_rec_avg, loss_kld_avg, total_loss_avg))

        # scalars
        writer.add_scalar('test/loss-seg', loss_seg_avg, epoch)
        writer.add_scalar('test/loss-seg-tar', loss_seg_tar_avg, epoch)
        writer.add_scalar('test/loss-rec', loss_rec_avg, epoch)
        writer.add_scalar('test/loss-kld', loss_kld_avg, epoch)
        writer.add_scalar('test/loss', total_loss_avg, epoch)

        # log everything
        if writer is not None and write_images:

            # images
            x = torch.cat((x_src, x_tar), dim=0)
            y = torch.cat((y_src, y_tar), dim=0)
            x_pred = torch.cat((x_src_pred, x_tar_pred), dim=0)
            y_pred = torch.cat((y_src_pred, y_tar_pred), dim=0)
            x = vutils.make_grid(x, normalize=True, scale_each=True)
            y = vutils.make_grid(y, normalize=y.max() - y.min() > 0, scale_each=True)
            x_pred = vutils.make_grid(x_pred.data, normalize=True, scale_each=True)
            y_pred = vutils.make_grid(F.softmax(y_pred, dim=1)[:, 1:2, :, :].data,
                                      normalize=y_pred.max() - y_pred.min() > 0, scale_each=True)
            writer.add_image('test/x', x, epoch)
            writer.add_image('test/y', y, epoch)
            writer.add_image('test/x-pred', x_pred, epoch)
            writer.add_image('test/y-pred', y_pred, epoch)

        return total_loss_avg

    # trains the network
    def train_net(self, train_loader_source, train_loader_target, test_loader_source, test_loader_target,
                  optimizer, loss_seg_fn, loss_rec_fn, scheduler=None, epochs=100, test_freq=1, print_stats=1,
                  log_dir=None, write_images_freq=1):

        # log everything if necessary
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        test_loss_min = np.inf
        for epoch in range(epochs):

            print('[%s] Epoch %5d/%5d' % (datetime.datetime.now(), epoch, epochs))

            # train the model for one epoch
            self.train_epoch(loader_src=train_loader_source, loader_tar=train_loader_target,
                             optimizer=optimizer, loss_seg_fn=loss_seg_fn, loss_rec_fn=loss_rec_fn, epoch=epoch,
                             print_stats=print_stats, writer=writer, write_images=epoch % write_images_freq == 0)

            # adjust learning rate if necessary
            if scheduler is not None:
                scheduler.step(epoch=epoch)

                # and keep track of the learning rate
                writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch)

            # test the model for one epoch is necessary
            if epoch % test_freq == 0:
                test_loss = self.test_epoch(loader_src=test_loader_source, loader_tar=test_loader_target,
                                            loss_seg_fn=loss_seg_fn, loss_rec_fn=loss_rec_fn, epoch=epoch,
                                            writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()