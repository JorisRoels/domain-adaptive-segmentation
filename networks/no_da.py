import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from networks.base import UNetDA2D, UNetDA2DClassifier, data_from_range

from neuralnets.data.datasets import LabeledVolumeDataset


class UNetNoDA2D(UNetDA2D):

    def training_step(self, batch, batch_idx):

        # get data
        x, y = batch
        x_src, x_tar = x
        y_src, y_tar = y
        tar_labels_available = y_tar.size(1) > 0

        # forward prop
        y_src_pred = self(x_src)
        y_tar_pred = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss = loss_src + loss_tar

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('train/mIoU_src', mIoU_src, prog_bar=True)
        self.log('train/loss_src', loss_src)
        u = y_tar.unique()
        if u.numel() != 1 or int(u) != 255:
            self.log('train/mIoU_tar', mIoU_tar, prog_bar=True)
            self.log('train/loss_tar', loss_tar)
            self.log('train/loss', loss)

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
        y_src_pred = self(x_src)
        y_tar_pred = self(x_tar)

        # compute loss
        loss_src = self.loss_fn(y_src_pred, y_src[:, 0, ...])
        loss_tar = self.loss_fn(y_tar_pred, y_tar[:, 0, ...]) if tar_labels_available else 0
        loss = loss_src + loss_tar

        # compute iou
        y_src_pred = torch.softmax(y_src_pred, dim=1)
        y_tar_pred = torch.softmax(y_tar_pred, dim=1)
        mIoU_src = self._mIoU(y_src_pred, y_src)
        mIoU_tar = self._mIoU(y_tar_pred, y_tar) if tar_labels_available else -1
        self.log('val/mIoU_src', mIoU_src, prog_bar=True)
        self.log('val/mIoU_tar', mIoU_tar, prog_bar=True)
        self.log('val/loss_src', loss_src)
        self.log('val/loss_tar', loss_tar)
        self.log('val/loss', loss)

        # log images
        if batch_idx == self.val_batch_id:
            self._log_predictions(x_src, y_src, y_src_pred, prefix='val_src')
            self._log_predictions(x_tar, y_tar if tar_labels_available else None, y_tar_pred, prefix='val_tar')

        return loss


class UNetNoDA2DClassifier(UNetDA2DClassifier):

    def fit(self, X, y):

        X, y = data_from_range(X, self.dataset)

        # initialize model and trainer
        self.model = UNetNoDA2D(in_channels=self.in_channels, feature_maps=self.feature_maps, levels=self.levels,
                                dropout_enc=self.dropout, dropout_dec=self.dropout, norm=self.norm,
                                activation=self.activation, coi=self.coi, loss_fn=self.loss_fn)
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