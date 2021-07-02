"""
    This is a script that illustrates semi-supervised domain adaptive training
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

from neuralnets.data.datasets import LabeledVolumeDataset, LabeledSlidingWindowDataset
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed

from util.tools import parse_params
from networks.factory import generate_model

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    """
        Parse all the arguments
    """
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the configuration file", type=str,
                        default='train_semi_supervised.yaml')
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
    split_src = params['src']['train_val_test_split']
    split_tar = params['tar']['train_val_test_split']
    transform = Compose([Rotate90(), Flip(prob=0.5, dim=0), Flip(prob=0.5, dim=1), ContrastAdjust(adj=0.1),
                         AddNoise(sigma_max=0.05)])
    print_frm('Train data...')
    train = LabeledVolumeDataset((params['src']['data'], params['tar']['data']),
                                 (params['src']['labels'], params['tar']['labels']), len_epoch=2000,
                                 input_shape=input_shape, in_channels=params['in_channels'],
                                 type=params['type'], batch_size=params['train_batch_size'], transform=transform,
                                 range_split=((0, split_src[0]), (0, split_tar[0])),
                                 range_dir=(params['src']['split_orientation'], params['tar']['split_orientation']),
                                 partial_labels=(1, params['tar_labels_available']))
    print_frm('Validation data...')
    val = LabeledVolumeDataset((params['src']['data'], params['tar']['data']),
                               (params['src']['labels'], params['tar']['labels']), len_epoch=2000,
                               input_shape=input_shape, in_channels=params['in_channels'], type=params['type'],
                               batch_size=params['train_batch_size'], transform=transform,
                               range_split=((split_src[0], split_src[1]), (split_tar[0], split_tar[1])),
                               range_dir=(params['src']['split_orientation'], params['tar']['split_orientation']))
    print_frm('Test data...')
    test = LabeledSlidingWindowDataset(params['tar']['data'], params['tar']['labels'], input_shape=input_shape,
                                       in_channels=params['in_channels'], type=params['type'],
                                       batch_size=params['train_batch_size'], transform=transform,
                                       range_split=(split_tar[1], 1), range_dir=params['tar']['split_orientation'])
    train_loader = DataLoader(train, batch_size=params['train_batch_size'], num_workers=params['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val, batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                            pin_memory=True)
    test_loader = DataLoader(test, batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                             pin_memory=True)

    """
        Build the network
    """
    print_frm('Building the network')
    net = generate_model(params['method'], params)

    """
        Train the network
    """
    print_frm('Starting training')
    print_frm('Training with loss: %s' % params['loss'])
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/mIoU', mode='max')
    trainer = pl.Trainer(max_epochs=params['epochs'], gpus=params['gpus'], accelerator=params['accelerator'],
                         default_root_dir=params['log_dir'], flush_logs_every_n_steps=params['log_freq'],
                         log_every_n_steps=params['log_freq'], callbacks=[lr_monitor, checkpoint_callback],
                         progress_bar_refresh_rate=params['log_refresh_rate'])
    trainer.fit(net, train_loader, val_loader)

    """
        Testing the network
    """
    print_frm('Testing network')
    trainer.test(net.get_unet(), test_loader)

    """
        Save the final model
    """
    print_frm('Saving final model')
    shutil.copyfile(trainer.checkpoint_callback.best_model_path, os.path.join(trainer.log_dir, 'best_model.ckpt'))

    """
        Clean up
    """
    print_frm('Cleaning up')
    if args.clean_up:
        os.system('rm -r ' + os.path.join(trainer.log_dir, 'checkpoints'))
