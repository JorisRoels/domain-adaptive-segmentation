
import os
import numpy as np
import torch

from torch.autograd import Function

from neuralnets.cross_validation.base import UNet2DClassifier
from neuralnets.networks.unet import UNet2D
from neuralnets.util.losses import CrossEntropyLoss

LEN_EPOCH = 1000


def data_from_range(rng, dataset):

    X = []
    y = []
    for i, data in enumerate(dataset.data):
        inds = np.unique((rng * len(data)).astype(int))
        X.append(data[inds])
        y.append(dataset.labels[i][inds])

    return X, y


def _crop2match(f, g):
    """
    Center crops the largest tensor of two so that its size matches the other tensor
    :param f: first tensor
    :param g: second tensor
    :return: the cropped tensors
    """

    for d in range(f.ndim):
        if f.size(d) > g.size(d):  # crop f
            diff = f.size(d) - g.size(d)
            rest = f.size(d) - g.size(d) - (diff // 2)
            f = torch.split(f, [diff // 2, g.size(d), rest], dim=d)[1]
        elif g.size(d) > f.size(d):  # crop g
            diff = g.size(d) - f.size(d)
            rest = g.size(d) - f.size(d) - (diff // 2)
            g = torch.split(g, [diff // 2, f.size(d), rest], dim=d)[1]

    return f.contiguous(), g.contiguous()


def _compute_covariance(x):

    n = x.size(0)  # batch_size

    sum_column = torch.sum(x, dim=0, keepdim=True)
    term_mul_2 = torch.mm(sum_column.t(), sum_column) / n
    d_t_d = torch.mm(x.t(), x)

    return (d_t_d - term_mul_2) / (n - 1)


def _mix_rbf_kernel(X, Y, sigma_list):
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


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
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


def feature_regularization_loss(f_src, f_tar, method='coral', n_samples=None):
    """
    Compute the regularization loss between the feature representations (shape [B, C, Y, X]) of the two streams
    In case of high dimensionality, there is an option to subsample
    :param f_src: features of the source stream
    :param f_tar: features of the target stream
    :param method: regularization method ('coral' or 'mmd')
    :param optional n_samples: number of samples to be selected
    :return: regularization loss
    """

    # if the samples are not equally sized, center crop the largest one
    f_src, f_tar = _crop2match(f_src, f_tar)

    # view features to [N, D] shape
    src = f_src.view(f_src.size(0), -1)
    tar = f_tar.view(f_tar.size(0), -1)

    if n_samples is None:
        fs = src
        ft = tar
    else:
        inds = torch.randperm(src.size(1))[:n_samples]
        fs = src[:, inds.to(src.device)]
        ft = tar[:, inds.to(tar.device)]

    if method == 'coral':
        return coral(fs, ft)
    else:
        return mmd(fs, ft)


def coral(source, target):
    """
    Compute CORAL loss between two feature vectors (https://arxiv.org/abs/1607.01719)
    :param source: source vector [N_S, D]
    :param target: target vector [N_T, D]
    :return: CORAL loss
    """
    d = source.size(1)

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4*d*d)

    return loss


def mmd(source, target, gamma=10**3):
    """
    Compute MMD loss between two feature vectors (https://arxiv.org/abs/1605.06636)
    :param source: source vector [N_S, D]
    :param target: target vector [N_T, D]
    :return: MMD loss
    """
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(source, target, [gamma])

    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=True)


def param_regularization_loss(src_params, tar_params, a, b):
    """
    Computes the regularization loss on the parameters of the two streams
    :param src_params: parameters in the source encoder
    :param tar_params: parameters in the target encoder
    :param a: list of multiplication parameters a
    :param b: list of bias parameters b
    :return: parameter regularization loss
    """

    cum_sum = 0
    w_loss = 0
    for i, (src_weight, tar_weight) in enumerate(zip(src_params, tar_params)):
        d = a[i].mul(src_weight) + b[i] - tar_weight
        w_loss = w_loss + torch.norm(d, 2)
        cum_sum += np.prod(np.array(d.shape))
    w_loss = w_loss / cum_sum

    return w_loss


class ReverseLayerF(Function):
    """
    Gradient reversal layer (https://arxiv.org/abs/1505.07818)
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()

        return output, None


class UNetDA2D(UNet2D):

    def test_step(self, batch, batch_idx):

        # get data
        x, y = batch

        # forward prop
        y_pred = self(x)

        # compute loss
        loss = self.loss_fn(y_pred, y[:, 0, ...])

        # compute iou
        y_pred = torch.softmax(y_pred, dim=1)
        mIoU = self._mIoU(y_pred, y)
        self.log('test/mIoU_tar', mIoU)
        self.log('test/loss_tar', loss)

        return loss

    def get_unet(self):
        """
        Get the segmentation network branch
        :return: a U-Net module
        """
        net = UNet2D(in_channels=self.encoder.in_channels, coi=self.coi, feature_maps=self.encoder.feature_maps,
                     levels=self.encoder.levels, skip_connections=self.decoder.skip_connections,
                     norm=self.encoder.norm, activation=self.encoder.activation, dropout_enc=self.encoder.dropout)

        net.encoder.load_state_dict(self.encoder.state_dict())
        net.decoder.load_state_dict(self.decoder.state_dict())

        return net


class UNetDA2DClassifier(UNet2DClassifier):

    def __init__(self, dataset, epochs=10, gpus=(0,), accelerator='dp', log_dir='logs', log_freq=50,
                 log_refresh_rate=None, train_batch_size=1, test_batch_size=1, num_workers=1, device=0,
                 orientations=(0,), normalization='unit', transform=None, input_shape=(1, 256, 256), in_channels=1,
                 coi=(0, 1), feature_maps=64, levels=4, skip_connections=True, residual_connections=False,
                 norm='instance', activation='relu', dropout=0.0, loss_fn=CrossEntropyLoss(), lr=1e-3,
                 partial_labels=1):
        super().__init__(epochs=epochs, gpus=gpus, accelerator=accelerator, log_dir=log_dir, log_freq=log_freq,
                         log_refresh_rate=log_refresh_rate, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size, num_workers=num_workers, device=device,
                         orientations=orientations, normalization=normalization, transform=transform,
                         input_shape=input_shape, in_channels=in_channels, coi=coi, feature_maps=feature_maps,
                         levels=levels, skip_connections=skip_connections, residual_connections=residual_connections,
                         norm=norm, activation=activation, dropout=dropout, loss_fn=loss_fn, lr=lr)

        # parameters
        self.dataset = dataset
        self.partial_labels = partial_labels

    def fit(self, X, y):
        pass

    def predict(self, X, y):

        X, y = data_from_range(X, self.dataset)
        segmentation = super().predict(X, y)

        return segmentation

    def score(self, X, y, sample_weight=None):

        X, y = data_from_range(X, self.dataset)

        # validate each model state and save the metrics
        checkpoints = os.listdir(self.trainer.checkpoint_callback.dirpath)
        checkpoints.sort()
        metrics = np.zeros(len(checkpoints))
        for i, ckpt in enumerate(checkpoints):
            ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, ckpt)
            self.model.load_state_dict(torch.load(ckpt_path))
            model = self.model.get_unet()
            metrics[i] = model.validate(X[1], y[1], model.input_shape, in_channels=model.in_channels,
                                        classes_of_interest=model.coi, batch_size=self.test_batch_size,
                                        device=self.device, orientations=self.orientations,
                                        normalization=self.normalization, report=False)

        # find the best model state
        j = np.argmax(metrics)
        metric = metrics[j]

        # remove the remaining checkpoints
        for i, ckpt in enumerate(checkpoints):
            if i != j:
                ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, ckpt)
                os.remove(ckpt_path)

        return metric