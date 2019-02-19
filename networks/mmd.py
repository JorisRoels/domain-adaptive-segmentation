
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from networks.unet import UNetEncoder2D, UNetDecoder2D, unet_from_encoder_decoder


# MMD U-Net model
class UNet_MMD(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, feature_maps=64, levels=4, group_norm=False):
        super(UNet_MMD, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels

        # encoder
        self.encoder = UNetEncoder2D(in_channels=in_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

        # segmentation decoder
        self.segmentation_decoder = UNetDecoder2D(out_channels=out_channels, feature_maps=feature_maps, levels=levels, group_norm=group_norm)

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
        loss_seg_cum = 0.0
        losses_mmd_cum = np.zeros(len(lambdas))
        total_loss_mmd_cum = 0.0
        loss_cum = 0.0
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
            loss_seg = loss_fn(y_src_pred, y_src)
            total_loss_mmd = 0
            for j, lambda_mmd in enumerate(lambdas):
                if lambda_mmd > 0:
                    f_j_src = src_features[j]
                    f_j_tar = tar_features[j]
                    sz = f_j_src.size()
                    for s in range(sz[1]):  # compute MMD per output activation, otherwise too memory consuming
                        loss_mmd = mmd(f_j_src[:, s, :, :].view(sz[0], sz[2] * sz[3]),
                                       f_j_tar[:, s, :, :].view(sz[0], sz[2] * sz[3]))
                        losses_mmd_cum[j] = losses_mmd_cum[j] + loss_mmd.data.cpu().numpy()
                        total_loss_mmd = total_loss_mmd + lambda_mmd * loss_mmd
            loss = loss_seg + total_loss_mmd
            loss_seg_cum += loss_seg.data.cpu().numpy()
            total_loss_mmd_cum += total_loss_mmd.data.cpu().numpy()
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

            # backward prop
            loss.backward()

            # apply one step in the optimization
            optimizer.step()

            # print statistics of necessary
            if i % print_stats == 0:
                print('[%s] Epoch %5d - Iteration %5d/%5d - Loss seg: %.6f - Loss MMD: %.6f - Loss: %.6f'
                      % (datetime.datetime.now(), epoch, i, len(loader_src.dataset)/loader_src.batch_size, loss, total_loss_mmd, loss))

        # don't forget to compute the average and print it
        loss_seg_avg = loss_seg_cum / cnt
        losses_mmd_avg = losses_mmd_cum / cnt
        total_loss_mmd_avg = total_loss_mmd_cum / cnt
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Loss seg: %.6f - Loss MMD: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_avg, total_loss_mmd_avg, loss_avg))

        # scalars
        writer.add_scalar('train-src/loss-seg', loss_seg_avg, epoch)
        for i in range(len(lambdas)):
            if lambdas[i]>0:
                writer.add_scalar('train/loss-mmd-level-' + str(i), losses_mmd_avg[i], epoch)
        writer.add_scalar('train/loss-mmd', total_loss_mmd_avg, epoch)
        writer.add_scalar('train/loss', loss_avg, epoch)

        # log everything
        if writer is not None:

            if write_images:
                # write images
                x_src = vutils.make_grid(x_src, normalize=True, scale_each=True)
                x_tar = vutils.make_grid(x_tar, normalize=True, scale_each=True)
                y_src = vutils.make_grid(y_src, normalize=y_src.max() - y_src.min() > 0, scale_each=True)
                y_src_pred = vutils.make_grid(F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data,
                                              normalize=y_src_pred.max() - y_src_pred.min() > 0, scale_each=True)
                y_tar_pred = vutils.make_grid(F.softmax(y_tar_pred, dim=1)[:, 1:2, :, :].data,
                                              normalize=y_tar_pred.max() - y_tar_pred.min() > 0, scale_each=True)
                writer.add_image('train-src/x', x_src, epoch)
                writer.add_image('train-tar/x', x_tar, epoch)
                writer.add_image('train-src/y', y_src, epoch)
                writer.add_image('train-src/y-pred', y_src_pred, epoch)
                writer.add_image('train-tar/y-pred', y_tar_pred, epoch)

        return loss_avg

    # tests the network over one epoch
    def test_epoch(self, loader_src, loader_tar, loss_fn, lambdas, epoch, writer=None, write_images=False):

        # make sure network is on the gpu and in training mode
        self.cuda()
        self.eval()

        # keep track of the average loss during the epoch
        loss_seg_src_cum = 0.0
        loss_seg_tar_cum = 0.0
        losses_mmd_cum = np.zeros(len(lambdas))
        total_loss_mmd_cum = 0.0
        loss_cum = 0.0
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
            loss_seg_src = loss_fn(y_src_pred, y_src)
            loss_seg_tar = loss_fn(y_tar_pred, y_tar)
            total_loss_mmd = 0
            for j, lambda_mmd in enumerate(lambdas):
                if lambda_mmd > 0:
                    f_j_src = src_features[j]
                    f_j_tar = tar_features[j]
                    sz = f_j_src.size()
                    for s in range(sz[1]):  # compute MMD per output activation, otherwise too memory consuming
                        loss_mmd = mmd(f_j_src[:, s, :, :].view(sz[0], sz[2] * sz[3]),
                                       f_j_tar[:, s, :, :].view(sz[0], sz[2] * sz[3]))
                        losses_mmd_cum[j] = losses_mmd_cum[j] + loss_mmd.data.cpu().numpy()
                        total_loss_mmd = total_loss_mmd + lambda_mmd * loss_mmd
            loss = loss_seg_src + total_loss_mmd
            loss_seg_src_cum += loss_seg_src.data.cpu().numpy()
            loss_seg_tar_cum += loss_seg_tar.data.cpu().numpy()
            total_loss_mmd_cum += total_loss_mmd.data.cpu().numpy()
            loss_cum += loss.data.cpu().numpy()
            cnt += 1

        # don't forget to compute the average and print it
        loss_seg_src_avg = loss_seg_src_cum / cnt
        loss_seg_tar_avg = loss_seg_tar_cum / cnt
        losses_mmd_avg = losses_mmd_cum / cnt
        total_loss_mmd_avg = total_loss_mmd_cum / cnt
        loss_avg = loss_cum / cnt
        print('[%s] Epoch %5d - Loss seg src: %.6f - Loss seg tar: %.6f - Loss MMD: %.6f - Loss: %.6f'
              % (datetime.datetime.now(), epoch, loss_seg_src_avg, loss_seg_tar_avg, total_loss_mmd_avg, loss_avg))

        # scalars
        writer.add_scalar('test-src/loss-seg', loss_seg_src_avg, epoch)
        writer.add_scalar('test-tar/loss-seg', loss_seg_tar_avg, epoch)
        for i in range(len(lambdas)):
            if lambdas[i]>0:
                writer.add_scalar('test/loss-mmd-level-' + str(i), losses_mmd_avg[i], epoch)
        writer.add_scalar('test/loss-mmd', total_loss_mmd_avg, epoch)
        writer.add_scalar('test/loss', loss_avg, epoch)

        # log everything
        if writer is not None and write_images:

            # images
            xs = vutils.make_grid(x_src, normalize=True, scale_each=True)
            xt = vutils.make_grid(x_tar, normalize=True, scale_each=True)
            ys = vutils.make_grid(y_src, normalize=y_src.max() - y_src.min() > 0, scale_each=True)
            yt = vutils.make_grid(y_tar, normalize=y_tar.max() - y_tar.min() > 0, scale_each=True)
            ys_pred = vutils.make_grid(F.softmax(y_src_pred, dim=1)[:, 1:2, :, :].data,
                                       normalize=y_src_pred.max() - y_src_pred.min() > 0, scale_each=True)
            yt_pred = vutils.make_grid(F.softmax(y_tar_pred, dim=1)[:, 1:2, :, :].data,
                                       normalize=y_tar_pred.max() - y_tar_pred.min() > 0, scale_each=True)
            writer.add_image('test-src/x', xs, epoch)
            writer.add_image('test-tar/x', xt, epoch)
            writer.add_image('test-src/y', ys, epoch)
            writer.add_image('test-tar/y', yt, epoch)
            writer.add_image('test-src/y-pred', ys_pred, epoch)
            writer.add_image('test-tar/y-pred', yt_pred, epoch)

        return loss_avg

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
                test_loss = self.test_epoch(loader_src=test_loader_source, loader_tar=test_loader_target,
                                            loss_fn=loss_fn, lambdas=lambdas, epoch=epoch,
                                            writer=writer, write_images=True)

                # and save model if lower test loss is found
                if test_loss < test_loss_min:
                    test_loss_min = test_loss
                    torch.save(self, os.path.join(log_dir, 'best_checkpoint.pytorch'))

            # save model every epoch
            torch.save(self, os.path.join(log_dir, 'checkpoint.pytorch'))

        writer.close()

min_var_est = 1e-8
GAMMA = 10 ^ 3

def mmd(source, target):

    K_XX, K_XY, K_YY, d = mix_rbf_kernel(source, target, [GAMMA])

    return mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=True)

def mix_rbf_kernel(X, Y, sigma_list):
    assert (X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

def mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    return mmd2
