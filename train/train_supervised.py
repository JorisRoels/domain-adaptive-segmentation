"""
    This is a script that illustrates supervised training
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

from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed

from util.tools import parse_params, get_dataloaders
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
                        default='train_supervised.yaml')
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
    train_loader, val_loader, test_loader = get_dataloaders(params, domain='tar', domain_labels_available=1.0,
                                                            supervised=True)

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
    trainer = train(net, train_loader, val_loader, [lr_monitor, checkpoint_callback], params)

    """
        Testing the network
    """
    print_frm('Testing network')
    validate(net, trainer, test_loader, params)

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
