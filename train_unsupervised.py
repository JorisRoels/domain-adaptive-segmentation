
"""
    This is a script that pretrains U-Nets with the following approaches:
        - Finetuning (FT): straightforward pretraining on the source
        - MMD: maximum mean discrepancy regularization
        - CORAL: correlation alignment regularization
        - DANN: domain adversarial neural networks
        - Y-NET: proposed reconstruction-based domain adaptation
    Usage:
        python train_unsupervised.py --method ynet --target drosophila
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

from data.epfl.epfl import EPFLDataset
from data.drosophila.drosophila import DrosophilaDataset, DrosophilaUnlabeledDataset
from data.hela.hela import HeLaDataset, HeLaUnlabeledDataset
from networks.unet import UNet
from networks.ynet import YNet
from networks.mmd import UNet_MMD
from networks.coral import UNet_CORAL
from networks.dann import UNet_DANN
from util.losses import JaccardLoss, MSELoss
from util.preprocessing import get_augmenters
from util.validation import segment
from util.metrics import jaccard, dice
from util.io import imwrite3D

"""
    Parse all the arguments
"""
print('[%s] Parsing arguments' % (datetime.datetime.now()))
parser = argparse.ArgumentParser()
# general parameters
parser.add_argument("--method", help="Domain adaptation method", type=str, default="ynet")
parser.add_argument("--target", help="Target dataset", type=str, default="drosophila")

# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs_domain_adaptation")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses", type=int, default=100)

# network parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=int, default=512)
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=32)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)", type=int, default=4)
parser.add_argument("--group_norm", help="Use group normalization instead of batch normalization", type=int, default=0)

# regularization parameters
parser.add_argument('--lambdas', help='Regularization parameters for MMD/CORAL/DANN', type=str, default="0,0,0,0,1e-3,0,0,0,0")
parser.add_argument('--lambda_rec', help='Regularization parameters for Y-Net reconstruction', type=float, default=0.001)

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-3)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay", type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=100)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=1)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=1)
loss_fn_seg = JaccardLoss()
loss_fn_rec = MSELoss()

args = parser.parse_args()
args.lambdas = [float(item) for item in args.lambdas.split(',')]

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
train_log_dir = os.path.join(args.log_dir, args.target, str(0.0), args.method, 'logs')
directories = [args.log_dir,
               os.path.join(args.log_dir, args.target),
               os.path.join(args.log_dir, args.target, str(0.0)),
               os.path.join(args.log_dir, args.target, str(0.0), args.method),
               train_log_dir,
               os.path.join(train_log_dir, 'pretraining'),
               train_log_dir]
for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

"""
    Load the data
"""
input_shape = (1, args.input_size, args.input_size)
# load source
print('[%s] Loading source data (EPFL)' % (datetime.datetime.now()))
train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters()
src_train = EPFLDataset(input_shape=input_shape, train=True,
                        transform=train_xtransform, target_transform=train_ytransform)
src_test = EPFLDataset(input_shape=input_shape, train=False,
                       transform=test_xtransform, target_transform=test_ytransform)
src_train_loader = DataLoader(src_train, batch_size=args.train_batch_size)
src_test_loader = DataLoader(src_test, batch_size=args.test_batch_size)
# load target
print('[%s] Loading target data (%s)' % (datetime.datetime.now(), args.target))
train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters()
if args.target == "drosophila":
    tar_train_unlabeled = DrosophilaUnlabeledDataset(input_shape=input_shape, transform=train_xtransform)
    tar_test = DrosophilaDataset(input_shape=input_shape, train=False,
                                 transform=test_xtransform, target_transform=test_ytransform)
elif args.target == "hela":
    tar_train_unlabeled = HeLaUnlabeledDataset(input_shape=input_shape, transform=train_xtransform)
    tar_test = HeLaDataset(input_shape=input_shape, train=False,
                           transform=test_xtransform, target_transform=test_ytransform)
else:
    raise ValueError('Unknown target dataset: the options are drosophila or hela. ')
tar_train_unlabeled_loader = DataLoader(tar_train_unlabeled, batch_size=args.train_batch_size)
tar_test_loader = DataLoader(tar_test, batch_size=args.test_batch_size)

"""
    Build the network
"""
print('[%s] Building the network (method: %s)' % (datetime.datetime.now(), args.method))
if args.method == 'finetuning':
    net = UNet(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))
elif args.method == 'mmd':
    net = UNet_MMD(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))
elif args.method == 'coral':
    net = UNet_CORAL(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))
elif args.method == 'dann':
    net = UNet_DANN(n=args.input_size, lambdas=args.lambdas, feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))
elif args.method == 'ynet':
    net = YNet(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))
else:
    raise ValueError('Unknown method: the options are finetuning, mmd, coral, dann or ynet. ')

"""
    Setup optimization for pretraining
"""
print('[%s] Setting up optimization for pretraining' % (datetime.datetime.now()))
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Train the network unsupervised
"""
print('[%s] Starting unsupervised training' % (datetime.datetime.now()))
if args.method == 'finetuning':
    net.train_net(train_loader_src=src_train_loader, test_loader_src=src_test_loader, test_loader_tar=tar_test_loader,
                  loss_fn=loss_fn_seg, optimizer=optimizer, scheduler=scheduler,
                  epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
                  log_dir=os.path.join(train_log_dir, 'pretraining'))
elif args.method == 'mmd' or args.method == 'coral' or args.method == 'dann':

    net.train_net(train_loader_source=src_train_loader, test_loader_source=src_test_loader,
                  train_loader_target=tar_train_unlabeled_loader, test_loader_target=tar_test_loader,
                  loss_fn=loss_fn_seg, lambdas=args.lambdas, optimizer=optimizer, scheduler=scheduler,
                  epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
                  log_dir=os.path.join(train_log_dir, 'pretraining'))
elif args.method == 'ynet':
    net.train_net(train_loader_source=src_train_loader, train_loader_target=tar_train_unlabeled_loader,
                  test_loader_source=src_test_loader, test_loader_target=tar_test_loader,
                  lambda_reg=args.lambda_rec, optimizer=optimizer, loss_seg_fn=loss_fn_seg, loss_rec_fn=loss_fn_rec,
                  scheduler=scheduler, epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
                  log_dir=os.path.join(train_log_dir, 'pretraining'))

"""
    Validate the trained network
"""
# load best checkpoint
test_data = tar_test.data
test_labels = tar_test.labels
print('[%s] Validating the trained network' % (datetime.datetime.now()))
seg_net = net.get_segmentation_net()
segmentation_last_checkpoint = segment(test_data, seg_net, [args.input_size, args.input_size], batch_size=args.test_batch_size)
j = jaccard(segmentation_last_checkpoint, test_labels)
d = dice(segmentation_last_checkpoint, test_labels)
print('[%s] Network performance (last checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
net = torch.load(os.path.join(train_log_dir, 'pretraining', 'best_checkpoint.pytorch'))
seg_net = net.get_segmentation_net()
segmentation_best_checkpoint = segment(test_data, seg_net, [args.input_size, args.input_size], batch_size=args.test_batch_size)
j = jaccard(segmentation_best_checkpoint, test_labels)
d = dice(segmentation_best_checkpoint, test_labels)
print('[%s] Network performance (best checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))

print('[%s] Finished!' % (datetime.datetime.now()))