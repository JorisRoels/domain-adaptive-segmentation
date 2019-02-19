
"""
    This is a script that pretrains U-Nets with the following approaches:
        - CORAL: correlation alignment regularization
"""

"""
    Necessary libraries
"""
import argparse
import datetime
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.datasets import UnlabeledVolumeDataset, StronglyLabeledVolumeDataset
from networks.coral import UNet_CORAL
from util.losses import CrossEntropyLoss
from util.preprocessing import get_augmenters_2d
from util.validation import segment
from util.metrics import jaccard, dice
from util.io import imwrite3D

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()

# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs")
parser.add_argument("--write_dir", help="Writing directory", type=str, default=None)
parser.add_argument("--src_data_train", help="Source data for training", type=str, default="../../data/EM/EPFL/training.tif")
parser.add_argument("--src_labels_train", help="Source labels for training", type=str, default="../../data/EM/EPFL/training_groundtruth.tif")
parser.add_argument("--src_data_test", help="Source data for testing", type=str, default="../../data/EM/EPFL/testing.tif")
parser.add_argument("--src_labels_test", help="Source labels for testing", type=str, default="../../data/EM/EPFL/testing_groundtruth.tif")
parser.add_argument("--tar_data_train", help="Target data for training", type=str, default="../../data/EM/VNC/data_larger.tif")
parser.add_argument("--tar_data_test", help="Target data for testing", type=str, default="../../data/EM/VNC/data_test.tif")
parser.add_argument("--tar_labels_test", help="Target labels for testing", type=str, default="../../data/EM/VNC/mito_labels_test.tif")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses", type=int, default=50)

# network parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=str, default="128,128")
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=16)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)", type=int, default=4)
parser.add_argument("--group_norm", help="Use group normalization instead of batch normalization", type=int, default=0)
parser.add_argument("--augment_noise", help="Use noise augmentation", type=int, default=0)

# regularization parameters
parser.add_argument('--lambdas', help='Regularization parameters for MMD/CORAL/DANN', type=str, default="0,0,0,0,1e-3,0,0,0,0")

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay", type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=10)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=4)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=4)

args = parser.parse_args()
args.input_size = [int(item) for item in args.input_size.split(',')]
args.lambdas = [float(item) for item in args.lambdas.split(',')]
loss_fn_seg = CrossEntropyLoss()

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if args.write_dir is not None:
    if not os.path.exists(args.write_dir):
        os.mkdir(args.write_dir)
    os.mkdir(os.path.join(args.write_dir, 'tar_segmentation_last'))
    os.mkdir(os.path.join(args.write_dir, 'tar_segmentation_best'))

"""
    Load the data
"""
input_shape = (1, args.input_size[0], args.input_size[1])
# load source
print('[%s] Loading data' % (datetime.datetime.now()))
# augmenters
src_train_xtransform, src_train_ytransform, src_test_xtransform, src_test_ytransform = get_augmenters_2d(augment_noise=(args.augment_noise == 1))
tar_train_xtransform, _, tar_test_xtransform, tar_test_ytransform = get_augmenters_2d(augment_noise=(args.augment_noise == 1))
# load data
src_train = StronglyLabeledVolumeDataset(args.src_data_train, args.src_labels_train, input_shape, transform=src_train_xtransform, target_transform=src_train_ytransform)
src_test = StronglyLabeledVolumeDataset(args.src_data_test, args.src_labels_test, input_shape, transform=src_test_xtransform, target_transform=src_test_ytransform)
tar_train = UnlabeledVolumeDataset(args.tar_data_train, input_shape=input_shape, transform=tar_train_xtransform)
tar_test = StronglyLabeledVolumeDataset(args.tar_data_test, args.tar_labels_test, input_shape=input_shape, transform=tar_test_xtransform, target_transform=tar_test_ytransform)
src_train_loader = DataLoader(src_train, batch_size=args.train_batch_size//2)
src_test_loader = DataLoader(src_test, batch_size=args.test_batch_size//2)
tar_train_loader = DataLoader(tar_train, batch_size=args.train_batch_size//2)
tar_test_loader = DataLoader(tar_test, batch_size=args.test_batch_size//2)

"""
    Build the network
"""
print('[%s] Building the network' % (datetime.datetime.now()))
net = UNet_CORAL(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))

"""
    Setup optimization for training
"""
print('[%s] Setting up optimization for training' % (datetime.datetime.now()))
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the network
"""
print('[%s] Starting training' % (datetime.datetime.now()))
net.train_net(train_loader_source=src_train_loader, train_loader_target=tar_train_loader,
              test_loader_source=src_test_loader, test_loader_target=tar_test_loader,
              loss_fn=loss_fn_seg, lambdas=args.lambdas, optimizer=optimizer, scheduler=scheduler,
              epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
              log_dir=args.log_dir)

"""
    Validate the trained network
"""
test_data = tar_test.data
test_labels = tar_test.labels
print('[%s] Validating the trained network' % (datetime.datetime.now()))
seg_net = net.get_segmentation_net()
segmentation_last_checkpoint = segment(test_data, seg_net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_last_checkpoint, test_labels)
d = dice(segmentation_last_checkpoint, test_labels)
print('[%s] Network performance (last checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
if args.write_dir is not None:
    print('[%s] Writing the output' % (datetime.datetime.now()))
    imwrite3D(segmentation_last_checkpoint, os.path.join(args.write_dir, 'tar_segmentation_last'), rescale=True)
net = torch.load(os.path.join(args.log_dir, 'best_checkpoint.pytorch'))
seg_net = net.get_segmentation_net()
segmentation_best_checkpoint = segment(test_data, seg_net, args.input_size, batch_size=args.test_batch_size)
j = jaccard(segmentation_best_checkpoint, test_labels)
d = dice(segmentation_best_checkpoint, test_labels)
print('[%s] Network performance (best checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
if args.write_dir is not None:
    print('[%s] Writing the output' % (datetime.datetime.now()))
    imwrite3D(segmentation_best_checkpoint, os.path.join(args.write_dir, 'tar_segmentation_best'), rescale=True)

print('[%s] Finished!' % (datetime.datetime.now()))