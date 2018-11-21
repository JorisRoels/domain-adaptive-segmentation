
import random
import os
import numpy as np
import torch.utils.data as data
from scipy import misc
from util.preprocessing import normalize
from util.io import read_tif
from util.tools import sample_labeled_input, sample_unlabeled_input

class DrosophilaDataset(data.Dataset):

    def __init__(self, input_shape, train=True, split=0.67, frac=1.0,
                 len_epoch=1000, transform=None, target_transform=None):

        self.train = train  # training set or test set
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.labels = self.read_data()
        # in the training case, we split the image in 4 strips which are concatenated along the z-axis
        # this allows to select a fraction of the data along that axis (otherwise only 13 slices, now 42)
        if self.train:
            K = 2
            sz = self.data.shape
            fK = int(sz[1]/K)
            data = np.zeros((K*sz[0], fK, sz[2]))
            labels = np.zeros((K*sz[0], fK, sz[2]))
            for k in range(K):
                data[k*sz[0]:(k+1)*sz[0], :, :] = self.data[:, k*fK:(k+1)*fK, :]
                labels[k*sz[0]:(k+1)*sz[0], :, :] = self.labels[:, k*fK:(k+1)*fK, :]
            self.data = data
            self.labels = labels

            K = 2
            sz = self.data.shape
            fK = int(sz[2]/K)
            data = np.zeros((K*sz[0], sz[1], fK))
            labels = np.zeros((K*sz[0], sz[1], fK))
            for k in range(K):
                data[k*sz[0]:(k+1)*sz[0], :, :] = self.data[:, :, k*fK:(k+1)*fK]
                labels[k*sz[0]:(k+1)*sz[0], :, :] = self.labels[:, :, k*fK:(k+1)*fK]
            self.data = data
            self.labels = labels

        s = int(split * self.data.shape[0])
        if self.train:
            self.data = self.data[:s, :, :]
            self.labels = self.labels[:s, :, :]
        else:
            self.data = self.data[s:, :, :]
            self.labels = self.labels[s:, :, :]

        # normalize data
        mu, std = self.get_stats()
        self.mu = mu
        self.std = std
        self.data = normalize(self.data, mu, std)
        self.labels = normalize(self.labels, 0, 255)

        # optionally: use only a fraction of the data
        s = int(frac * self.data.shape[0])
        sel = random.sample(range(self.data.shape[0]), s)
        self.data = self.data[sel, :, :]
        self.labels = self.labels[sel, :, :]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)

        return input, target

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

    def read_data(self):

        data = np.zeros((20, 1024, 1024), dtype='uint8')
        labels = np.zeros((20, 1024, 1024), dtype='int')

        for i in range(20):
            i_str = str(i) if i>9 else '0'+str(i)
            data[i] = read_tif(os.path.join('data', 'drosophila', 'groundtruth-drosophila-vnc', 'stack1', 'raw', i_str+'.tif'), dtype='uint8')
            labels[i] = misc.imread(os.path.join('data', 'drosophila', 'groundtruth-drosophila-vnc', 'stack1', 'mitochondria', i_str+'.png'))

        return data, labels

class DrosophilaUnlabeledDataset(data.Dataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None):

        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform

        self.data = self.read_data()

        # normalize data
        mu, std = self.get_stats()
        self.mu = mu
        self.std = std
        self.data = normalize(self.data, mu, std)

    def __getitem__(self, i):

        # get random sample
        input = sample_unlabeled_input(self.data, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        return input

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

    def read_data(self):

        data = np.zeros((40, 1024, 1024), dtype='uint8')

        for i in range(20):
            i_str = str(i) if i>9 else '0'+str(i)
            data[i] = read_tif(os.path.join('data', 'drosophila', 'groundtruth-drosophila-vnc', 'stack1', 'raw', i_str+'.tif'), dtype='uint8')
            data[20+i] = read_tif(os.path.join('data', 'drosophila', 'groundtruth-drosophila-vnc', 'stack2', 'raw', i_str+'.tif'), dtype='uint8')

        return data