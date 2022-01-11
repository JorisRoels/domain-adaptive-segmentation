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

from neuralnets.util.io import print_frm, mkdir
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
    train_loader, val_loader, test_loader = get_dataloaders(params)
    if params['tar_labels_available'] > 0:
        train_loader_tar, val_loader_tar, test_loader_tar = get_dataloaders(params, domain='tar',
                                                                domain_labels_available=params['tar_labels_available'])

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
    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/mIoU_tar', mode='max')
    trainer_pre = train(net, train_loader, val_loader, [lr_monitor, checkpoint_callback], params)

    # if target labels are available, finetune on the target data
    unet = net.get_unet()
    if params['tar_labels_available'] > 0:

        # reduce learning rate
        unet.lr = params['lr'] / 10

        # validate performance before finetuning
        print_frm('Testing network intermediate performance')
        validate(unet, trainer_pre, test_loader_tar, params)

        print_frm('Starting finetuning on target')
        # reduce learning rate
        checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/mIoU', mode='max')
        trainer = train(unet, train_loader_tar, val_loader_tar, [checkpoint_callback], params, reduce_epochs=True)

    else:

        trainer = trainer_pre

    """
        Testing the network
    """
    print_frm('Testing network final performance')
    validate(unet, trainer, test_loader_tar if params['tar_labels_available'] > 0 else test_loader, params)

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
        shutil.rmtree(os.path.join(trainer.log_dir, 'checkpoints'), ignore_errors=True)
        mkdir(os.path.join(trainer.log_dir, 'pretraining'))
        for file_name in os.listdir(trainer.log_dir):
            shutil.move(os.path.join(trainer.log_dir, file_name), os.path.join(trainer.log_dir, 'pretraining'))
        if params['tar_labels_available'] > 0:
            mkdir(os.path.join(trainer.log_dir, 'finetuning'))
            for file_name in os.listdir(trainer_pre.log_dir):
                shutil.move(os.path.join(trainer_pre.log_dir, file_name), os.path.join(trainer.log_dir, 'finetuning'))
            shutil.rmtree(trainer_pre.log_dir, ignore_errors=True)
            shutil.move(trainer.log_dir, trainer_pre.log_dir)
