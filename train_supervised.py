
"""
    This is a script that finetunes U-Nets originating from DA networks:
        - Finetuning (FT): straightforward pretraining on the source
        - MMD: maximum mean discrepancy regularization
        - CORAL: correlation alignment regularization
        - DANN: domain adversarial neural networks
        - Y-NET: proposed reconstruction-based domain adaptation
    Usage:
        python train_supervised.py --method ynet --target drosophila
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

from data.drosophila.drosophila import DrosophilaDataset
from data.hela.hela import HeLaDataset
from networks.unet import UNet
from util.losses import JaccardLoss
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
parser.add_argument("--frac_target_labels", help="Fraction of the target labels that are used", type=float, default=0.0)

# logging parameters
parser.add_argument("--log_dir", help="Logging directory", type=str, default="logs_domain_adaptation")
parser.add_argument("--print_stats", help="Number of iterations between each time to log training losses", type=int, default=100)

# network parameters
parser.add_argument("--input_size", help="Size of the blocks that propagate through the network", type=int, default=512)
parser.add_argument("--fm", help="Number of initial feature maps in the segmentation U-Net", type=int, default=32)
parser.add_argument("--levels", help="Number of levels in the segmentation U-Net (i.e. number of pooling stages)", type=int, default=4)
parser.add_argument("--group_norm", help="Use group normalization instead of batch normalization", type=int, default=0)

# optimization parameters
parser.add_argument("--lr", help="Learning rate of the optimization", type=float, default=1e-4)
parser.add_argument("--step_size", help="Number of epochs after which the learning rate should decay", type=int, default=10)
parser.add_argument("--gamma", help="Learning rate decay factor", type=float, default=0.9)
parser.add_argument("--epochs", help="Total number of epochs to train", type=int, default=100)
parser.add_argument("--test_freq", help="Number of epochs between each test stage", type=int, default=1)
parser.add_argument("--train_batch_size", help="Batch size in the training stage", type=int, default=1)
parser.add_argument("--test_batch_size", help="Batch size in the testing stage", type=int, default=1)
loss_fn_seg = JaccardLoss()

args = parser.parse_args()

"""
    Setup logging directory
"""
print('[%s] Setting up log directories' % (datetime.datetime.now()))
train_log_dir = os.path.join(args.log_dir, args.target, str(args.frac_target_labels * 100), args.method, 'logs')
directories = [args.log_dir,
               os.path.join(args.log_dir, args.target),
               os.path.join(args.log_dir, args.target, str(args.frac_target_labels * 100)),
               os.path.join(args.log_dir, args.target, str(args.frac_target_labels * 100), args.method),
               train_log_dir,
               os.path.join(train_log_dir, 'finetuning'),
               train_log_dir]
for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

"""
    Load the data
"""
input_shape = (1, args.input_size, args.input_size)
# load target
print('[%s] Loading target data (%s)' % (datetime.datetime.now(), args.target))
train_xtransform, train_ytransform, test_xtransform, test_ytransform = get_augmenters()
if args.target == "drosophila":
    tar_train = DrosophilaDataset(input_shape=input_shape, train=True, frac=args.frac_target_labels,
                                  transform=train_xtransform, target_transform=train_ytransform)
    tar_test = DrosophilaDataset(input_shape=input_shape, train=False,
                                 transform=test_xtransform, target_transform=test_ytransform)
elif args.target == "hela":
    tar_train = HeLaDataset(input_shape=input_shape, train=True, frac=args.frac_target_labels,
                            transform=train_xtransform, target_transform=train_ytransform)
    tar_test = HeLaDataset(input_shape=input_shape, train=False,
                           transform=test_xtransform, target_transform=test_ytransform)
else:
    raise ValueError('Unknown target dataset: the options are drosophila or hela. ')
tar_train_loader = DataLoader(tar_train, batch_size=args.train_batch_size)
tar_test_loader = DataLoader(tar_test, batch_size=args.test_batch_size)


"""
    Setup optimization for finetuning
"""
print('[%s] Setting up optimization for finetuning' % (datetime.datetime.now()))
# load best checkpoint
# net = torch.load(os.path.join(args.log_dir, args.target, str(0.0), args.method, 'logs', 'pretraining', 'best_checkpoint.pytorch'))
net = UNet(feature_maps=args.fm, levels=args.levels, group_norm=(args.group_norm==1))
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

"""
    Finetune on target if necessary
"""
print('[%s] Finetuning with %d percent of the target labels' % (datetime.datetime.now(), args.frac_target_labels*100))
seg_net = net.get_segmentation_net()
if args.frac_target_labels > 0:
    seg_net.train_net(train_loader_src=tar_train_loader, test_loader_src=tar_test_loader, test_loader_tar=None,
                      loss_fn=loss_fn_seg, optimizer=optimizer, scheduler=scheduler,
                      epochs=args.epochs, test_freq=args.test_freq, print_stats=args.print_stats,
                      log_dir=os.path.join(train_log_dir, 'finetuning'))

"""
    Validate the trained network
"""
print('[%s] Validating the trained network' % (datetime.datetime.now()))
test_data = tar_test.data
test_labels = tar_test.labels
if args.frac_target_labels == 0:
    net = torch.load(os.path.join(train_log_dir, 'pretraining', 'checkpoint.pytorch'))
    seg_net = net.get_segmentation_net()
segmentation_last_checkpoint = segment(test_data, seg_net, [args.input_size, args.input_size], batch_size=args.test_batch_size)
j = jaccard(segmentation_last_checkpoint, test_labels)
d = dice(segmentation_last_checkpoint, test_labels)
print('[%s] Network performance (last checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))
if args.frac_target_labels == 0:
    net = torch.load(os.path.join(train_log_dir, 'pretraining', 'best_checkpoint.pytorch'))
    seg_net = net.get_segmentation_net()
else:
    net = torch.load(os.path.join(train_log_dir, 'finetuning', 'best_checkpoint.pytorch'))
    seg_net = net.get_segmentation_net()
segmentation_best_checkpoint = segment(test_data, seg_net, [args.input_size, args.input_size], batch_size=args.test_batch_size)
j = jaccard(segmentation_best_checkpoint, test_labels)
d = dice(segmentation_best_checkpoint, test_labels)
print('[%s] Network performance (best checkpoint): Jaccard=%f - Dice=%f' % (datetime.datetime.now(), j, d))

print('[%s] Finished!' % (datetime.datetime.now()))