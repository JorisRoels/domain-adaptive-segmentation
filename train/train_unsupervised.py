"""
    This is a script that illustrates unsupervised domain adaptive training, i.e. without using any target labels
"""

"""
    Necessary libraries
"""
import argparse
import yaml
import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from neuralnets.data.datasets import LabeledVolumeDataset, UnlabeledSlidingWindowDataset
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm, mkdir
from neuralnets.util.tools import set_seed

from util.tools import parse_params
from networks.factory import generate_model
from train.base import train, validate

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    """
        Parse all the arguments
    """
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the configuration file", type=str,
                        default='clem1.yaml')
    parser.add_argument("--clean-up", help="Boolean flag that specifies cleaning of the checkpoints",
                        action='store_true', default=False)
    args = parser.parse_args()
    with open(args.config) as file:
        params = parse_params(yaml.load(file, Loader=yaml.FullLoader))

    """
    Fix seed (for reproducibility)
    """
    set_seed(params['seed'])

    """
        Load the data
    """
    print_frm('Loading data')
    input_shape = (1, *(params['input_size']))
    split_src = params['src']['train_val_split']
    split_tar = params['tar']['train_val_split']
    transform = Compose([Rotate90(), Flip(prob=0.5, dim=0), Flip(prob=0.5, dim=1), ContrastAdjust(adj=0.1),
                         AddNoise(sigma_max=0.05)])
    print_frm('Train data...')
    train_data = LabeledVolumeDataset((params['src']['data'], params['tar']['data']), (params['src']['labels'], None),
                                      len_epoch=params['len_epoch'], input_shape=input_shape,
                                      in_channels=params['in_channels'], type=params['type'],
                                      batch_size=params['train_batch_size'], transform=transform,
                                      range_split=((0, split_src[0]), (0, split_tar[0])),
                                      range_dir=(params['src']['split_orientation'], params['tar']['split_orientation']))
    print_frm('Validation data...')
    val_data = LabeledVolumeDataset((params['src']['data'], params['tar']['data']), (params['src']['labels'], None),
                                    len_epoch=params['len_epoch'], input_shape=input_shape,
                                    in_channels=params['in_channels'], type=params['type'],
                                    batch_size=params['test_batch_size'], transform=transform,
                                    range_split=((split_src[0], 1), (split_tar[0], 1)),
                                    range_dir=(params['src']['split_orientation'], params['tar']['split_orientation']))
    print_frm('Test data...')
    test_data = UnlabeledSlidingWindowDataset(params['tar']['data'], input_shape=input_shape,
                                              in_channels=params['in_channels'], type=params['type'],
                                              batch_size=params['test_batch_size'])
    train_loader = DataLoader(train_data, batch_size=params['train_batch_size'], num_workers=params['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                            pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                             pin_memory=True)

    """
        Build the network
    """
    print_frm('Building the network')
    net = generate_model(params['method'], params)

    """
        Train the network
    """
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    print_frm('Starting joint pretraining')
    print_frm('Training with loss: %s' % params['loss'])
    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/mIoU', mode='max')
    trainer = train(net, train_loader, val_loader, [lr_monitor, checkpoint_callback], params)
    unet = net.get_unet()

    """
        Testing the network
    """
    print_frm('Testing network final performance')
    validate(unet, trainer, test_loader, params)

    """
        Save the final model
    """
    print_frm('Saving final model')
    shutil.copyfile(trainer.checkpoint_callback.best_model_path, os.path.join(trainer.log_dir, 'best_model.ckpt'))

    """
        Clean up
    """
    if args.clean_up:
        print_frm('Cleaning up')
        os.system('rm -r ' + os.path.join(trainer.log_dir, 'checkpoints'))
        mkdir(os.path.join(trainer.log_dir, 'pretraining'))
        os.system('mv ' + trainer.log_dir + '/events.out.tfevents.* ' + os.path.join(trainer.log_dir, 'pretraining'))
