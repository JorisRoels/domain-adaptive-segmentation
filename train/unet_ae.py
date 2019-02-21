
"""
    This is a script that trains a U-Net autoencoder in a domain adaptive way (i.e. both source and target are inferred)
"""

"""
    Necessary libraries
"""
import os
import argparse
import datetime
import torch
from torch.utils.data import DataLoader

from data.datasets import UnlabeledVolumeDataset
from networks.unet_ae import UNetAE
from util.io import imwrite3D
from util.losses import MSELoss
from util.preprocessing import get_augmenters_2d
from util.validation import transform

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs")
parser.add_argument("--write_dir", help="Writing directory", type=str, default=None)
parser.add_argument("--src_data_train", help="Source data for training", type=str, default="../../data/EM/EPFL/training.tif")
parser.add_argument("--src_data_test", help="Source data for testing", type=str, default="../../data/EM/EPFL/testing.tif")
parser.add_argument("--tar_data_train", help="Target data for training", type=str, default="../../data/EM/VNC/data_larger.tif")
parser.add_argument("--tar_data_test", help="Target data for testing", type=str, default="../../data/EM/VNC/data_test.tif")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses", type=int, default=50)

# network parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=str, default="128,128")
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=16)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)", type=int, default=4)
parser.add_argument("--group_norm", help="Use group normalization instead of batch normalization", type=int, default=0)
parser.add_argument("--augment_noise", help="Use noise augmentation", type=int, default=0)
parser.add_argument("--lambda_dom", help="Confusion loss regularization", type=float, default=1e-3)

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay", type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=50)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=4)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=4)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]
loss_fn_rec = MSELoss()

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if args.write_dir is not None:
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    os.mkdir(os.path.join(args.write_dir, 'src_transform'))
    os.mkdir(os.path.join(args.write_dir, 'tar_transform'))

"""
    Load the data
"""
input_shape = (1, args.input_size[0], args.input_size[1])
print('[%s] Loading data' % (datetime.datetime.now()))
# augmenters
src_train_xtransform, _, src_test_xtransform, _ = get_augmenters_2d(augment_noise=(args.augment_noise == 1))
tar_train_xtransform, _, tar_test_xtransform, _ = get_augmenters_2d(augment_noise=(args.augment_noise == 1))
# load data
src_train = UnlabeledVolumeDataset(args.src_data_train, input_shape, transform=src_train_xtransform)
src_test = UnlabeledVolumeDataset(args.src_data_test, input_shape, transform=src_test_xtransform)
tar_train = UnlabeledVolumeDataset(args.tar_data_train, input_shape=input_shape, transform=tar_train_xtransform)
tar_test = UnlabeledVolumeDataset(args.tar_data_test, input_shape=input_shape, transform=tar_test_xtransform)
src_train_loader = DataLoader(src_train, batch_size=args.train_batch_size//2)
src_test_loader = DataLoader(src_test, batch_size=args.test_batch_size//2)
tar_train_loader = DataLoader(tar_train, batch_size=args.train_batch_size//2)
tar_test_loader = DataLoader(tar_test, batch_size=args.test_batch_size//2)

"""
    Setup optimization for unsupervised training
"""
print('[%s] Setting up optimization for unsupervised training' % (datetime.datetime.now()))
net = UNetAE(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm == 1), lambda_dom=args.lambda_dom, s=args.input_size[0])

"""
    Train the network unsupervised
"""
print('[%s] Training network unsupervised' % (datetime.datetime.now()))
net.train_net(src_train_loader=src_train_loader, src_test_loader=src_test_loader,
              tar_train_loader=tar_train_loader, tar_test_loader=tar_test_loader,
              loss_fn=loss_fn_rec, lr=args.lr, step_size=args.step_size, gamma=args.gamma,
              epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
              log_dir=args.log_dir)

"""
    Write out the results
"""
if args.write_dir is not None:
    print('[%s] Running the trained network on the source dataset' % (datetime.datetime.now()))
    src_data = src_test.data
    transform_last_checkpoint = transform(src_data, net, args.input_size, batch_size=args.test_batch_size)
    print('[%s] Writing the output' % (datetime.datetime.now()))
    imwrite3D(transform_last_checkpoint, os.path.join(args.write_dir, 'src_transform'), rescale=True)
    tar_data = tar_test.data
    print('[%s] Running the trained network on the target dataset' % (datetime.datetime.now()))
    transform_last_checkpoint = transform(tar_data, net, args.input_size, batch_size=args.test_batch_size)
    print('[%s] Writing the output' % (datetime.datetime.now()))
    imwrite3D(transform_last_checkpoint, os.path.join(args.write_dir, 'tar_transform'), rescale=True)

print('[%s] Finished!' % (datetime.datetime.now()))