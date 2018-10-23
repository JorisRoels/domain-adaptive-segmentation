
import os
import random
import numpy as np
import torch.utils.data as data
from util.preprocessing import normalize
from util.io import read_tif
from util.tools import sample_labeled_input, sample_unlabeled_input

class HeLaDataset(data.Dataset):

    def __init__(self, input_shape, train=True, split=0.67, frac=1.0,
                 len_epoch=1000, transform=None, target_transform=None):

        self.train = train  # training set or test set
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform

        self.data = read_tif(os.path.join('data', 'hela', 'data.tif'), dtype='uint8')
        self.labels = read_tif(os.path.join('data', 'hela', 'mito_labels.tif'), dtype='int')
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
        if s > 0:
            self.data = self.data[sel, :, :]
            self.labels = self.labels[sel, :, :]
        else:
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

class HeLaUnlabeledDataset(data.Dataset):

    def __init__(self, input_shape, len_epoch=1000, transform=None):

        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform

        self.data = read_tif(os.path.join('data', 'hela', 'data_larger.tif'), dtype='uint8')

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