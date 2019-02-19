
import numpy as np
import numpy.random as rnd
import torch
import torchvision.transforms as transforms
from imgaug import augmenters as iaa

class ToFloatTensor(object):

    def __call__(self, x):

        return torch.FloatTensor(x.copy())

class ToLongTensor(object):

    def __call__(self, x):
        if isinstance(x, float):
            return torch.LongTensor([x.copy()])[0]
        else:
            return torch.LongTensor(x.copy())

class AddChannelAxis(object):

    def __call__(self, x):

        return x[np.newaxis, ...]

class AddNoise(object):

    def __init__(self, prob=0.5, sigma_min=0.0, sigma_max=1.0):

        self.prob = prob
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):

        if rnd.rand() < self.prob:
            sigma = rnd.uniform(self.sigma_min, self.sigma_max)
            return x + rnd.normal(0, sigma, x.shape)
        else:
            return x

class Normalize(object):

    def __init__(self, mu=0, std=1):

        self.mu = mu
        self.std = std

    def __call__(self, x):

        return (x-self.mu)/self.std

class RandomDeformations(object):

    def __init__(self, prob=0.5, scale=(0.01, 0.05), seed=None, thr=False):

        self.prob = prob
        self.scale = scale
        self.augmenter = iaa.PiecewiseAffine(scale=self.scale)
        if seed is not None:
            self.seed = seed
        else:
            self.seed = rnd.randint(0,2**32)
        self.thr = thr

    def __call__(self, x):

        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand() < self.prob:
            s = rnd.randint(0,2**32)
            x_aug = np.zeros_like(x)
            for i in range(x.shape[0]):
                self.augmenter.reseed(s)
                x_aug[i:i+1,...] = self.augmenter.augment_images(x[i:i+1,...].copy())
            if self.thr:
                x_aug = np.asarray(x_aug>0.5, float) # avoid weird noise artifacts
            return x_aug
        else:
            return x

class Rotate90(object):

    def __init__(self, prob=0.5, axes=(1,2), seed=None):

        self.prob = prob
        self.axes = axes
        if seed is not None:
            self.seed = seed
        else:
            self.seed = rnd.randint(0,2**32)

    def __call__(self, x):

        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand()<self.prob:
            return np.rot90(x.copy(), k=rnd.randint(1,4), axes=self.axes)
        else:
            return x

class FlipX(object):

    def __init__(self, prob=0.5, seed=None):

        self.prob = prob
        if seed is not None:
            self.seed = seed

    def __call__(self, x):

        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand()<self.prob:
            return x[:,::-1,...]
        else:
            return x

class FlipY(object):

    def __init__(self, prob=0.5, seed=None):

        self.prob = prob
        if seed is not None:
            self.seed = seed

    def __call__(self, x):

        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand()<self.prob:
            return x[:,:,::-1,...]
        else:
            return x

class FlipZ(object):

    def __init__(self, prob=0.5, seed=None):

        self.prob = prob
        if seed is not None:
            self.seed = seed

    def __call__(self, x):

        rnd.seed(self.seed)
        self.seed += 1
        if self.seed == 2**32:
            self.seed = 0
        if rnd.rand()<self.prob:
            return x[::-1,:,:]
        else:
            return x

class Flatten(object):

    def __call__(self, x):

        return x.view(-1)

def get_augmenters_2d(augment_noise=True):
    # standard augmenters: rotation, flips, deformations

    # generate seeds for synchronized augmentation
    s1 = np.random.randint(0, 2 ** 32)
    s2 = np.random.randint(0, 2 ** 32)
    s3 = np.random.randint(0, 2 ** 32)
    s4 = np.random.randint(0, 2 ** 32)

    # define transforms
    if augment_noise:
        train_xtransform = transforms.Compose([Rotate90(seed=s1),
                                               FlipX(seed=s2),
                                               FlipY(seed=s3),
                                               RandomDeformations(seed=s4),
                                               AddNoise(sigma_max=0.2),
                                               ToFloatTensor()])
    else:
        train_xtransform = transforms.Compose([Rotate90(seed=s1),
                                               FlipX(seed=s2),
                                               FlipY(seed=s3),
                                               RandomDeformations(seed=s4),
                                               ToFloatTensor()])
    train_ytransform = transforms.Compose([Rotate90(seed=s1),
                                           FlipX(seed=s2),
                                           FlipY(seed=s3),
                                           RandomDeformations(seed=s4, thr=True),
                                           ToLongTensor()])
    test_xtransform = transforms.Compose([ToFloatTensor()])
    test_ytransform = transforms.Compose([ToLongTensor()])

    return train_xtransform, train_ytransform, test_xtransform, test_ytransform

def normalize(x, mu, std):
    return (x-mu)/std
