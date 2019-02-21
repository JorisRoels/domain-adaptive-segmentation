
import numpy as np
import torch.utils.data as data
from util.preprocessing import normalize
from util.io import read_tif
from util.tools import sample_unlabeled_input, sample_labeled_input

class StronglyLabeledVolumeDataset(data.Dataset):

    def __init__(self, data_path: object, label_path: object, input_shape: object, split: object = None, train: object = None, len_epoch: object = 1000,
                 preprocess: object = 'z',
                 transform: object = None,
                 target_transform: object = None,
                 dtypes: object = ('uint8', 'uint8')) -> object:

        self.data_path = data_path
        self.label_path = label_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform
        self.target_transform = target_transform

        self.data = read_tif(data_path, dtype=dtypes[0])
        self.labels = read_tif(label_path, dtype=dtypes[1])

        mu, std = self.get_stats()
        self.mu = mu
        self.std = std
        self.preprocess = preprocess
        if preprocess == 'z':
            self.data = normalize(self.data, mu, std)
        elif preprocess == 'unit':
            self.data = normalize(self.data, 0, 255)
        self.labels = normalize(self.labels, 0, 255)

        if split is not None:
            if train:
                s = int(split * self.data.shape[2])
                self.data = self.data[:, :, :s]
                self.labels = self.labels[:, :, :s]
            else:
                s = int(split * self.data.shape[2])
                self.data = self.data[:, :, s:]
                self.labels = self.labels[:, :, s:]

    def __getitem__(self, i):

        # get random sample
        input, target = sample_labeled_input(self.data, self.labels, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None and len(target)>0:
            target = self.target_transform(target)
        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...], target[np.newaxis, ...]
        else:
            return input, target

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std

class UnlabeledVolumeDataset(data.Dataset):

    def __init__(self, data_path, input_shape, len_epoch=1000, preprocess='unit', transform=None, dtype='uint8'):

        self.data_path = data_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.transform = transform

        self.data = read_tif(data_path, dtype=dtype)

        mu, std = self.get_stats()
        self.mu = mu
        self.std = std
        self.preprocess = preprocess
        if preprocess == 'z':
            self.data = normalize(self.data, mu, std)
        elif preprocess == 'unit':
            self.data = normalize(self.data, 0, 255)

    def __getitem__(self, i):

        # get random sample
        input = sample_unlabeled_input(self.data, self.input_shape)

        # perform augmentation if necessary
        if self.transform is not None:
            input = self.transform(input)

        if self.input_shape[0] > 1: # 3D data
            return input[np.newaxis, ...]
        else:
            return input

    def __len__(self):

        return self.len_epoch

    def get_stats(self):

        mu = np.mean(self.data)
        std = np.std(self.data)

        return mu, std