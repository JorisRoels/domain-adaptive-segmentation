import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from neuralnets.networks.unet import UNet2D, UNetDecoder2D, UNetEncoder2D
from neuralnets.networks.cnn import CNN2D
from neuralnets.util.losses import CrossEntropyLoss, L2Loss
from neuralnets.data.datasets import LabeledVolumeDataset

from networks.base import UNetDA2D, UNetDA2DClassifier, data_from_range, feature_regularization_loss, ReverseLayerF


class UNetMMD2D(UNetDA2D):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn='ce', lr=1e-3, lambda_mmd=0):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr, return_features=True)

        self.lambda_mmd = lambda_mmd

    def training_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred, f_src = self(x_src)
        y_tar_pred, f_tar = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_mmd = feature_regularization_loss(f_src, f_tar, method='mmd')
        loss = loss_src + loss_tar + self.lambda_mmd * loss_mmd

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('train/mIoU_src', mIoU_src, prog_bar=True)
        self.log('train/loss_src', loss_src)
        self.log('train/loss_mmd', loss_mmd, prog_bar=True)
        self.log('train/loss', loss)
        u = y_tar.unique()
        if u.numel() != 1 or int(u) != 255:
            self.log('train/mIoU_tar', mIoU_tar, prog_bar=True)
            self.log('train/loss_tar', loss_tar)

        # log images
        if batch_idx == self.train_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='train_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='train_tar')

        return loss

    def validation_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred, f_src = self(x_src)
        y_tar_pred, f_tar = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_mmd = feature_regularization_loss(f_src, f_tar, method='mmd')
        loss = loss_src + loss_tar + self.lambda_mmd * loss_mmd

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('val/mIoU_src', mIoU_src)
        self.log('val/mIoU_tar', mIoU_tar, prog_bar=True)
        self.log('val/loss_src', loss_src)
        self.log('val/loss_tar', loss_tar)
        self.log('val/loss_mmd', loss_mmd, prog_bar=True)
        self.log('val/loss', loss)

        # log images
        if batch_idx == self.val_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='val_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='val_tar')

        return loss


class UNetMMD2DClassifier(UNetDA2DClassifier):

    def __init__(self, dataset, epochs=10, gpus=(0,), accelerator='dp', log_dir='logs', log_freq=50,
                 log_refresh_rate=None, train_batch_size=1, test_batch_size=1, num_workers=1, device=0,
                 orientations=(0,), normalization='unit', transform=None, input_shape=(1, 256, 256), in_channels=1,
                 coi=(0, 1), feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', activation='relu', dropout=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3,
                 partial_labels=1, len_epoch=1000, lambda_mmd=1):
        super().__init__(dataset, epochs=epochs, gpus=gpus, accelerator=accelerator, log_dir=log_dir, log_freq=log_freq,
                         log_refresh_rate=log_refresh_rate, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size, num_workers=num_workers, device=device,
                         orientations=orientations, normalization=normalization, transform=transform,
                         input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout=dropout, loss_fn=loss_fn, lr=lr,
                         partial_labels=partial_labels, len_epoch=len_epoch)

        # parameters
        self.lambda_mmd = lambda_mmd

    def fit(self, X, y):

        X, y = data_from_range(X, self.dataset)

        # initialize model and trainer
        self.model = UNetMMD2D(in_channels=self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                               dropout_enc=self.dropout, dropout_dec=self.dropout, norm=self.norm,
                               activation=self.activation, coi=self.coi, loss_fn=self.loss_fn,
                               lambda_mmd=self.lambda_mmd)
        self.trainer = pl.Trainer(max_epochs=int(self.epochs), gpus=self.gpus, accelerator=self.accelerator,
                                  default_root_dir=self.log_dir, flush_logs_every_n_steps=self.log_freq,
                                  log_every_n_steps=self.log_freq, callbacks=self.callbacks,
                                  progress_bar_refresh_rate=self.log_refresh_rate)

        # construct dataloader
        train = LabeledVolumeDataset(X, y, input_shape=(1, *self.input_shape), in_channels=self.in_channels,
                                     batch_size=self.train_batch_size, transform=self.transform,
                                     partial_labels=self.partial_labels, len_epoch=self.len_epoch)
        loader = DataLoader(train, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True)

        # train the network
        self.trainer.fit(self.model, loader)

        return self


class UNetDAT2D(UNetDA2D):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn='ce', lr=1e-3, lambda_dat=0, conv_channels=(16, 16, 16, 16, 16),
                 fc_channels=(128, 32)):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr, return_features=True)

        self.lambda_dat = lambda_dat
        self.conv_channels = conv_channels
        self.fc_channels = fc_channels

        self.loss_ce = CrossEntropyLoss()

        # domain classifier
        self.domain_classifier = CNN2D(conv_channels, fc_channels, (feature_maps, *self.input_shape))

    def forward(self, x):

        # contractive path
        y_pred, f = super().forward(x)

        # gradient reversal on the final feature layer
        f_rev = ReverseLayerF.apply(f)
        dom_pred = self.domain_classifier(f_rev)

        return y_pred, dom_pred

    def training_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0
        dom_labels = torch.zeros((x_src.size(0) + x_tar.size(0))).long().to(x_src.device)
        dom_labels[x_src.size(0):] = 1

        # forward prop
        y_src_pred, dom_src_pred = self(x_src)
        y_tar_pred, dom_tar_pred = self(x_tar)
        dom_pred = torch.cat((dom_src_pred, dom_tar_pred), dim=0)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_dat = self.loss_ce(dom_pred, dom_labels)
        loss = loss_src + loss_tar + self.lambda_dat * loss_dat

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('train/mIoU_src', mIoU_src, prog_bar=True)
        self.log('train/loss_src', loss_src)
        self.log('train/loss_dat', loss_dat, prog_bar=True)
        self.log('train/loss', loss)
        u = y_tar.unique()
        if u.numel() != 1 or int(u) != 255:
            self.log('train/mIoU_tar', mIoU_tar, prog_bar=True)
            self.log('train/loss_tar', loss_tar)

        # log images
        if batch_idx == self.train_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='train_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='train_tar')

        return loss

    def validation_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0
        dom_labels = torch.zeros((x_src.size(0) + x_tar.size(0))).long().to(x_src.device)
        dom_labels[x_src.size(0):] = 1

        # forward prop
        y_src_pred, dom_src_pred = self(x_src)
        y_tar_pred, dom_tar_pred = self(x_tar)
        dom_pred = torch.cat((dom_src_pred, dom_tar_pred), dim=0)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_dat = self.loss_ce(dom_pred, dom_labels)
        loss = loss_src + loss_tar + self.lambda_dat * loss_dat

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('val/mIoU_src', mIoU_src)
        self.log('val/mIoU_tar', mIoU_tar, prog_bar=True)
        self.log('val/loss_src', loss_src)
        self.log('val/loss_tar', loss_tar)
        self.log('val/loss_dat', loss_dat, prog_bar=True)
        self.log('val/loss', loss)

        # log images
        if batch_idx == self.val_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='val_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='val_tar')

        return loss


class UNetDAT2DClassifier(UNetDA2DClassifier):

    def __init__(self, dataset, epochs=10, gpus=(0,), accelerator='dp', log_dir='logs', log_freq=50,
                 log_refresh_rate=None, train_batch_size=1, test_batch_size=1, num_workers=1, device=0,
                 orientations=(0,), normalization='unit', transform=None, input_shape=(1, 256, 256), in_channels=1,
                 coi=(0, 1), feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', activation='relu', dropout=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3,
                 partial_labels=1, len_epoch=1000, lambda_dat=1):
        super().__init__(dataset, epochs=epochs, gpus=gpus, accelerator=accelerator, log_dir=log_dir, log_freq=log_freq,
                         log_refresh_rate=log_refresh_rate, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size, num_workers=num_workers, device=device,
                         orientations=orientations, normalization=normalization, transform=transform,
                         input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout=dropout, loss_fn=loss_fn, lr=lr,
                         partial_labels=partial_labels, len_epoch=len_epoch)

        # parameters
        self.lambda_dat = lambda_dat

    def fit(self, X, y):

        X, y = data_from_range(X, self.dataset)

        # initialize model and trainer
        self.model = UNetDAT2D(in_channels=self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                               dropout_enc=self.dropout, dropout_dec=self.dropout, norm=self.norm,
                               activation=self.activation, coi=self.coi, loss_fn=self.loss_fn,
                               lambda_dat=self.lambda_dat, input_shape=self.input_shape)
        self.trainer = pl.Trainer(max_epochs=int(self.epochs), gpus=self.gpus, accelerator=self.accelerator,
                                  default_root_dir=self.log_dir, flush_logs_every_n_steps=self.log_freq,
                                  log_every_n_steps=self.log_freq, callbacks=self.callbacks,
                                  progress_bar_refresh_rate=self.log_refresh_rate)

        # construct dataloader
        train = LabeledVolumeDataset(X, y, input_shape=(1, *self.input_shape), in_channels=self.in_channels,
                                     batch_size=self.train_batch_size, transform=self.transform,
                                     partial_labels=self.partial_labels, len_epoch=self.len_epoch)
        loader = DataLoader(train, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True)

        # train the network
        self.trainer.fit(self.model, loader)

        return self


class YNet2D(UNetDA2D):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn='ce', lr=1e-3, lambda_rec=0):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr)

        self.lambda_rec = lambda_rec

        self.loss_rec = nn.MSELoss()

        # reconstruction decoder
        self.decoder_rec = UNetDecoder2D(in_channels, feature_maps=self.feature_maps, levels=self.levels,
                                         skip_connections=False, norm=self.norm, dropout=self.dropout_dec,
                                         activation=self.activation)

    def forward(self, x):

        # contractive path
        encoder_outputs, encoded = self.encoder(x)

        # expansive segmentation path
        _, y_pred = self.decoder(encoded, encoder_outputs)

        # expansive reconstruction path
        _, x_rec = self.decoder_rec(encoded, None)

        return y_pred, x_rec

    def training_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred, x_src_rec = self(x_src)
        y_tar_pred, x_tar_l_rec = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_rec = self.loss_rec(x_src_rec, x_src) + self.loss_rec(x_tar_l_rec, x_tar)
        loss = loss_src + loss_tar + self.lambda_rec * loss_rec

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('train/mIoU_src', mIoU_src, prog_bar=True)
        self.log('train/loss_src', loss_src)
        self.log('train/loss_rec', loss_rec, prog_bar=True)
        self.log('train/loss', loss)
        u = y_tar.unique()
        if u.numel() != 1 or int(u) != 255:
            self.log('train/mIoU_tar', mIoU_tar, prog_bar=True)
            self.log('train/loss_tar', loss_tar)

        # log images
        if batch_idx == self.train_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='train_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='train_tar')

        return loss

    def validation_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred, x_src_rec = self(x_src)
        y_tar_pred, x_tar_rec = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_rec = self.loss_rec(x_src_rec, x_src) + self.loss_rec(x_tar_rec, x_tar)
        loss = loss_src + loss_tar + self.lambda_rec * loss_rec

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('val/mIoU_src', mIoU_src)
        self.log('val/mIoU_tar', mIoU_tar, prog_bar=True)
        self.log('val/loss_src', loss_src)
        self.log('val/loss_tar', loss_tar)
        self.log('val/loss_rec', loss_rec, prog_bar=True)
        self.log('val/loss', loss)

        # log images
        if batch_idx == self.val_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='val_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='val_tar')

        return loss


class YNet2DClassifier(UNetDA2DClassifier):

    def __init__(self, dataset, epochs=10, gpus=(0,), accelerator='dp', log_dir='logs', log_freq=50,
                 log_refresh_rate=None, train_batch_size=1, test_batch_size=1, num_workers=1, device=0,
                 orientations=(0,), normalization='unit', transform=None, input_shape=(1, 256, 256), in_channels=1,
                 coi=(0, 1), feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', activation='relu', dropout=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3,
                 partial_labels=1, len_epoch=1000, lambda_rec=0):
        super().__init__(dataset, epochs=epochs, gpus=gpus, accelerator=accelerator, log_dir=log_dir, log_freq=log_freq,
                         log_refresh_rate=log_refresh_rate, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size, num_workers=num_workers, device=device,
                         orientations=orientations, normalization=normalization, transform=transform,
                         input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout=dropout, loss_fn=loss_fn, lr=lr,
                         partial_labels=partial_labels, len_epoch=len_epoch)

        # parameters
        self.lambda_rec = lambda_rec

    def fit(self, X, y):

        X, y = data_from_range(X, self.dataset)

        # initialize model and trainer
        self.model = YNet2D(in_channels=self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                            dropout_enc=self.dropout, dropout_dec=self.dropout, norm=self.norm,
                            activation=self.activation, coi=self.coi, loss_fn=self.loss_fn, lambda_rec=self.lambda_rec)
        self.trainer = pl.Trainer(max_epochs=int(self.epochs), gpus=self.gpus, accelerator=self.accelerator,
                                  default_root_dir=self.log_dir, flush_logs_every_n_steps=self.log_freq,
                                  log_every_n_steps=self.log_freq, callbacks=self.callbacks,
                                  progress_bar_refresh_rate=self.log_refresh_rate)

        # construct dataloader
        train = LabeledVolumeDataset(X, y, input_shape=(1, *self.input_shape), in_channels=self.in_channels,
                                     batch_size=self.train_batch_size, transform=self.transform,
                                     partial_labels=self.partial_labels, len_epoch=self.len_epoch)
        loader = DataLoader(train, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True)

        # train the network
        self.trainer.fit(self.model, loader)

        return self


class WNet2D(UNetDA2D):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn='ce', lr=1e-3, lambda_rec=0, lambda_dat=0, conv_channels=(16, 16, 16, 16, 16),
                 fc_channels=(128, 32)):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr, return_features=True)

        self.lambda_rec = lambda_rec
        self.lambda_dat = lambda_dat
        self.conv_channels = conv_channels
        self.fc_channels = fc_channels

        self.loss_ce = CrossEntropyLoss()
        self.loss_rec = L2Loss()

        # domain classifier
        self.domain_classifier = CNN2D(conv_channels, fc_channels, (feature_maps, *self.input_shape))

        # reconstruction network
        self.net_rec = UNet2D(input_shape=input_shape, in_channels=in_channels, coi=(1,), feature_maps=feature_maps,
                              levels=levels, skip_connections=False, norm=norm, activation=activation,
                              dropout_enc=dropout_enc, dropout_dec=dropout_dec, loss_fn='l2', lr=lr,
                              return_features=True)

    def forward(self, x):

        # reconstruction
        x_rec, f = self.net_rec(x)

        # gradient reversal on the final feature layer
        f_rev = ReverseLayerF.apply(f)
        dom_pred = self.domain_classifier(f_rev)

        # segmentation
        y_pred, _ = super().forward(x_rec)

        return y_pred, x_rec, dom_pred

    def training_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0
        dom_labels = torch.zeros((x_src.size(0) + x_tar.size(0))).long().to(x_src.device)
        dom_labels[x_src.size(0):] = 1

        # forward prop
        y_src_pred, x_src_rec, dom_src_pred = self(x_src)
        y_tar_pred, x_tar_rec, dom_tar_pred = self(x_tar)
        dom_pred = torch.cat((dom_src_pred, dom_tar_pred), dim=0)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_rec = self.loss_rec(x_src_rec, x_src) + self.loss_rec(x_tar_rec, x_tar)
        loss_dat = self.loss_ce(dom_pred, dom_labels)
        loss = loss_src + loss_tar + self.lambda_rec * loss_rec + self.lambda_dat * loss_dat

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('train/mIoU_src', mIoU_src, prog_bar=True)
        self.log('train/loss_src', loss_src)
        self.log('train/loss_rec', loss_rec, prog_bar=True)
        self.log('train/loss_dat', loss_dat, prog_bar=True)
        self.log('train/loss', loss)
        u = y_tar.unique()
        if u.numel() != 1 or int(u) != 255:
            self.log('train/mIoU_tar', mIoU_tar, prog_bar=True)
            self.log('train/loss_tar', loss_tar)

        # log images
        if batch_idx == self.train_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='train_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='train_tar')

        return loss

    def validation_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0
        dom_labels = torch.zeros((x_src.size(0) + x_tar.size(0))).long().to(x_src.device)
        dom_labels[x_src.size(0):] = 1

        # forward prop
        y_src_pred, x_src_rec, dom_src_pred = self(x_src)
        y_tar_pred, x_tar_rec, dom_tar_pred = self(x_tar)
        dom_pred = torch.cat((dom_src_pred, dom_tar_pred), dim=0)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_rec = self.loss_rec(x_src_rec, x_src) + self.loss_rec(x_tar_rec, x_tar)
        loss_dat = self.loss_ce(dom_pred, dom_labels)
        loss = loss_src + loss_tar + self.lambda_rec * loss_rec + self.lambda_dat * loss_dat

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('val/mIoU_src', mIoU_src)
        self.log('val/loss_src', loss_src)
        self.log('val/mIoU_tar', mIoU_tar, prog_bar=True)
        self.log('val/loss_tar', loss_tar)
        self.log('val/loss_rec', loss_rec, prog_bar=True)
        self.log('val/loss_dat', loss_dat, prog_bar=True)
        self.log('val/loss', loss)

        # log images
        if batch_idx == self.val_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='val_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='val_tar')

        return loss


class WNet2DClassifier(UNetDA2DClassifier):

    def __init__(self, dataset, epochs=10, gpus=(0,), accelerator='dp', log_dir='logs', log_freq=50,
                 log_refresh_rate=None, train_batch_size=1, test_batch_size=1, num_workers=1, device=0,
                 orientations=(0,), normalization='unit', transform=None, input_shape=(1, 256, 256), in_channels=1,
                 coi=(0, 1), feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', activation='relu', dropout=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3,
                 partial_labels=1, len_epoch=1000, lambda_rec=0, lambda_dat=0):
        super().__init__(dataset, epochs=epochs, gpus=gpus, accelerator=accelerator, log_dir=log_dir, log_freq=log_freq,
                         log_refresh_rate=log_refresh_rate, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size, num_workers=num_workers, device=device,
                         orientations=orientations, normalization=normalization, transform=transform,
                         input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout=dropout, loss_fn=loss_fn, lr=lr,
                         partial_labels=partial_labels, len_epoch=len_epoch)

        # parameters
        self.lambda_rec = lambda_rec
        self.lambda_dat = lambda_dat

    def fit(self, X, y):

        X, y = data_from_range(X, self.dataset)

        # initialize model and trainer
        self.model = WNet2D(in_channels=self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                            dropout_enc=self.dropout, dropout_dec=self.dropout, norm=self.norm,
                            activation=self.activation, coi=self.coi, loss_fn=self.loss_fn, lambda_rec=self.lambda_rec,
                            lambda_dat=self.lambda_dat, input_shape=self.input_shape)
        self.trainer = pl.Trainer(max_epochs=int(self.epochs), gpus=self.gpus, accelerator=self.accelerator,
                                  default_root_dir=self.log_dir, flush_logs_every_n_steps=self.log_freq,
                                  log_every_n_steps=self.log_freq, callbacks=self.callbacks,
                                  progress_bar_refresh_rate=self.log_refresh_rate)

        # construct dataloader
        train = LabeledVolumeDataset(X, y, input_shape=(1, *self.input_shape), in_channels=self.in_channels,
                                     batch_size=self.train_batch_size, transform=self.transform,
                                     partial_labels=self.partial_labels, len_epoch=self.len_epoch)
        loader = DataLoader(train, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True)

        # train the network
        self.trainer.fit(self.model, loader)

        return self


class UNetTS2D(UNetDA2D):

    def __init__(self, input_shape=(1, 256, 256), in_channels=1, coi=(0, 1), feature_maps=64, levels=4,
                 skip_connections=True, residual_connections=False, norm='instance', activation='relu', dropout_enc=0.0,
                 dropout_dec=0.0, loss_fn='ce', lr=1e-3, lambda_w=0, lambda_o=0, n_samples_coral=4096):
        super().__init__(input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout_enc=dropout_enc, dropout_dec=dropout_dec,
                         loss_fn=loss_fn, lr=lr)

        self.lambda_w = lambda_w
        self.lambda_o = lambda_o

        self.n_samples_coral = n_samples_coral

        # reconstruction decoder
        self.encoder_src = UNetEncoder2D(self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                                         norm=self.norm, dropout=self.dropout_enc, activation=self.activation)

        # parameter transfer parameters
        for i, weight in enumerate(self.encoder_src.parameters()):
            a = nn.Parameter(torch.ones(weight.shape))
            b = nn.Parameter(torch.zeros(weight.shape))
            self.register_parameter('a' + str(i), a)
            self.register_parameter('b' + str(i), b)

    def forward(self, x, target=True):

        # contractive path
        if target:
            encoder_outputs, encoded = self.encoder(x)
        else:
            encoder_outputs, encoded = self.encoder_src(x)

        # expansive path
        decoder_outputs, y_pred = self.decoder(encoded, encoder_outputs)
        f = decoder_outputs[-1]

        return y_pred, f

    def training_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred, f_src = self(x_src, target=False)
        y_tar_pred, f_tar = self(x_tar, target=True)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_o = feature_regularization_loss(f_src, f_tar, method='coral', n_samples=self.n_samples_coral)
        loss_w = self._param_regularization_loss()
        loss = loss_src + loss_tar + self.lambda_o * loss_o + self.lambda_w * loss_w

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('train/mIoU_src', mIoU_src, prog_bar=True)
        self.log('train/loss_src', loss_src)
        self.log('train/loss_o', loss_o, prog_bar=True)
        self.log('train/loss_w', loss_w, prog_bar=True)
        self.log('train/loss', loss)
        u = y_tar.unique()
        if u.numel() != 1 or int(u) != 255:
            self.log('train/mIoU_tar', mIoU_tar, prog_bar=True)
            self.log('train/loss_tar', loss_tar)

        # log images
        if batch_idx == self.train_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='train_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='train_tar')

        return loss

    def validation_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred, f_src = self(x_src, target=False)
        y_tar_pred, f_tar = self(x_tar, target=True)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss_o = feature_regularization_loss(f_src, f_tar, method='coral', n_samples=self.n_samples_coral)
        loss_w = self._param_regularization_loss()
        loss = loss_src + loss_tar + self.lambda_o * loss_o + self.lambda_w * loss_w

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('val/mIoU_src', mIoU_src)
        self.log('val/loss_src', loss_src)
        self.log('val/mIoU_tar', mIoU_tar, prog_bar=True)
        self.log('val/loss_tar', loss_tar)
        self.log('val/loss_o', loss_o, prog_bar=True)
        self.log('val/loss_w', loss_w, prog_bar=True)
        self.log('val/loss', loss)

        # log images
        if batch_idx == self.train_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='val_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='val_tar')

        return loss

    def _param_regularization_loss(self):
        """
        Computes the regularization loss on the parameters of the two streams
        :return: parameter regularization loss
        """

        src_params = self.encoder_src.parameters()
        tar_params = self.encoder.parameters()

        cum_sum = 0
        w_loss = 0
        for i, (src_weight, tar_weight) in enumerate(zip(src_params, tar_params)):
            a = getattr(self, 'a' + str(i))
            b = getattr(self, 'b' + str(i))
            d = a.mul(src_weight) + b - tar_weight
            w_loss = w_loss + torch.pow(d, 2).sum()
            cum_sum += np.prod(np.array(d.shape))
        w_loss = w_loss / cum_sum

        return w_loss


class UNetTS2DClassifier(UNetDA2DClassifier):

    def __init__(self, dataset, epochs=10, gpus=(0,), accelerator='dp', log_dir='logs', log_freq=50,
                 log_refresh_rate=None, train_batch_size=1, test_batch_size=1, num_workers=1, device=0,
                 orientations=(0,), normalization='unit', transform=None, input_shape=(1, 256, 256), in_channels=1,
                 coi=(0, 1), feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', activation='relu', dropout=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3,
                 partial_labels=1, len_epoch=1000, lambda_w=0, lambda_o=0, n_samples_coral=4096):
        super().__init__(dataset, epochs=epochs, gpus=gpus, accelerator=accelerator, log_dir=log_dir, log_freq=log_freq,
                         log_refresh_rate=log_refresh_rate, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size, num_workers=num_workers, device=device,
                         orientations=orientations, normalization=normalization, transform=transform,
                         input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout=dropout, loss_fn=loss_fn, lr=lr,
                         partial_labels=partial_labels, len_epoch=len_epoch)

        # parameters
        self.lambda_w = lambda_w
        self.lambda_o = lambda_o
        self.n_samples_coral = n_samples_coral

    def fit(self, X, y):

        X, y = data_from_range(X, self.dataset)

        # initialize model and trainer
        self.model = UNetTS2D(in_channels=self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                              dropout_enc=self.dropout, dropout_dec=self.dropout, norm=self.norm,
                              activation=self.activation, coi=self.coi, loss_fn=self.loss_fn, lambda_w=self.lambda_w,
                              lambda_o=self.lambda_o, n_samples_coral=self.n_samples_coral)
        self.trainer = pl.Trainer(max_epochs=int(self.epochs), gpus=self.gpus, accelerator=self.accelerator,
                                  default_root_dir=self.log_dir, flush_logs_every_n_steps=self.log_freq,
                                  log_every_n_steps=self.log_freq, callbacks=self.callbacks,
                                  progress_bar_refresh_rate=self.log_refresh_rate)

        # construct dataloader
        train = LabeledVolumeDataset(X, y, input_shape=(1, *self.input_shape), in_channels=self.in_channels,
                                     batch_size=self.train_batch_size, transform=self.transform,
                                     partial_labels=self.partial_labels, len_epoch=self.len_epoch)
        loader = DataLoader(train, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True)

        # train the network
        self.trainer.fit(self.model, loader)

        return self