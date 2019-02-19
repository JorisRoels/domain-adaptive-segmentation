
import numpy as np
import torch
from skimage.color import label2rgb

# generate an input and target sample of certain shape from a labeled dataset
def sample_labeled_input(data, labels, input_shape):

    # randomize seed
    np.random.seed()

    # generate random position
    x = np.random.randint(0, data.shape[0]-input_shape[0]+1)
    y = np.random.randint(0, data.shape[1]-input_shape[1]+1)
    z = np.random.randint(0, data.shape[2]-input_shape[2]+1)

    # extract input and target patch
    input = data[x:x+input_shape[0], y:y+input_shape[1], z:z+input_shape[2]]
    if len(labels)>0:
        target = labels[x:x+input_shape[0], y:y+input_shape[1], z:z+input_shape[2]]
    else:
        target = []

    return input, target

# generate an input and target sample of certain shape from a labeled dataset
def sample_unlabeled_input(data, input_shape):

    # randomize seed
    np.random.seed()

    # generate random position
    x = np.random.randint(0, data.shape[0]-input_shape[0]+1)
    y = np.random.randint(0, data.shape[1]-input_shape[1]+1)
    z = np.random.randint(0, data.shape[2]-input_shape[2]+1)

    # extract input and target patch
    input = data[x:x+input_shape[0], y:y+input_shape[1], z:z+input_shape[2]]

    return input

# returns a 3D Gaussian window that can be used for window weighting and merging
def gaussian_window(size, sigma=1):

    # half window sizes
    hwz = size[0]//2
    hwy = size[1]//2
    hwx = size[2]//2

    # construct mesh grid
    if size[0] % 2 == 0:
        axz = np.arange(-hwz, hwz)
    else:
        axz = np.arange(-hwz, hwz + 1)
    if size[1] % 2 == 0:
        axy = np.arange(-hwy, hwy)
    else:
        axy = np.arange(-hwy, hwy + 1)
    if size[2] % 2 == 0:
        axx = np.arange(-hwx, hwx)
    else:
        axx = np.arange(-hwx, hwx + 1)
    xx, zz, yy = np.meshgrid(axx, axz, axy)

    # normal distribution
    gw = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2. * sigma ** 2))

    # normalize so that the mask integrates to 1
    gw = gw / np.sum(gw)

    return gw

# load a network
def load_net(model_file):
    return torch.load(model_file)

# returns an image overlayed with a labeled mask
# x is assumed to be a grayscale numpy array
# y is assumed to be a numpy array of integers for the different labels, all zeros will be opaque
def overlay(x, y, alpha=0.3, bg_label=0):
    return label2rgb(y, image=x, alpha=alpha, bg_label=bg_label, colors=[[0,1,0]])