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
import time
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from neuralnets.util.io import print_frm, mkdir
from neuralnets.util.tools import set_seed
from neuralnets.util.validation import validate

from util.tools import parse_params, process_seconds, get_dataloaders
from networks.factory import generate_model

from multiprocessing import freeze_support


def _validate(net, trainer, loader, params):
    # validates a network that was trained using a specific trainer on a dataset
    t_start = time.perf_counter()
    test_data, test_labels = loader.dataset.data[0], loader.dataset.labels[0]
    validate(net, test_data, test_labels, params['input_size'], in_channels=params['in_channels'],
             classes_of_interest=params['coi'], batch_size=params['test_batch_size'],
             write_dir=os.path.join(trainer.log_dir, 'best_predictions'),
             val_file=os.path.join(trainer.log_dir, 'metrics.npy'), device=params['gpus'][0])
    t_stop = time.perf_counter()
    print_frm('Elapsed testing time: %d hours, %d minutes, %.2f seconds' % process_seconds(t_stop - t_start))


def _train(net, train_loader, val_loader, callbacks, params, reduce_epochs=False):

    # trains a model with a specific training and validation loader, and manually specified callbacks
    epochs = params['epochs'] // 2 if reduce_epochs else params['epochs']
    trainer = pl.Trainer(max_epochs=epochs, gpus=params['gpus'], accelerator=params['accelerator'],
                         default_root_dir=params['log_dir'], flush_logs_every_n_steps=params['log_freq'],
                         log_every_n_steps=params['log_freq'], callbacks=callbacks,
                         progress_bar_refresh_rate=params['log_refresh_rate'])
    trainer.fit(net, train_loader, val_loader)

    # load the best checkpoint
    net.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])

    return trainer


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
                        action='store_true', default=True)
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
    if params['method'] == 'no-da':
        # data for pretraining
        train_loader_src, val_loader_src, test_loader_src = get_dataloaders(params, domain='src',
                                                                            domain_labels_available=1.0)
    else:
        # data for joint pretraining
        train_loader, val_loader, test_loader = get_dataloaders(params)

    # data for finetuning and final testing
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
    print_frm('Starting training')
    t_start = time.perf_counter()

    print_frm('Training with loss: %s' % params['loss'])
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    if params['method'] == 'no-da':

        # pretrain on the source data
        print_frm('Starting pretraining')
        checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/mIoU', mode='max')
        trainer_pre = _train(net, train_loader_src, val_loader_src, [lr_monitor, checkpoint_callback], params)

    else:

        # train domain adaptive on the source and target data
        print_frm('Starting joint pretraining')
        checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/mIoU_tar', mode='max')
        trainer_pre = _train(net, train_loader, val_loader, [lr_monitor, checkpoint_callback], params)

    # if target labels are available, finetune on the target data
    unet = net.get_unet()
    if params['tar_labels_available'] > 0:

        # reduce learning rate
        unet.lr = params['lr'] / 10

        # validate performance before finetuning
        print_frm('Testing network intermediate performance')
        _validate(unet, trainer_pre, test_loader_tar, params)

        print_frm('Starting finetuning on target')
        # reduce learning rate
        checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/mIoU', mode='max')
        trainer = _train(unet, train_loader_tar, val_loader_tar, [checkpoint_callback], params, reduce_epochs=True)

    else:

        trainer = trainer_pre


    t_stop = time.perf_counter()
    print_frm('Elapsed training time: %d hours, %d minutes, %.2f seconds' % process_seconds(t_stop - t_start))
    print_frm('Average time / epoch: %.2f hours, %d minutes, %.2f seconds' %
              process_seconds((t_stop - t_start) / params['epochs']))

    """
        Testing the network
    """
    print_frm('Testing network final performance')
    _validate(unet, trainer, test_loader_tar, params)

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
        mkdir(os.path.join(trainer.log_dir, 'pretraining'))
        os.system('mv ' + trainer.log_dir + '/events.out.tfevents.* ' + os.path.join(trainer.log_dir, 'pretraining'))
        if params['tar_labels_available'] > 0:
            mkdir(os.path.join(trainer.log_dir, 'finetuning'))
            os.system('mv ' + trainer_pre.log_dir + '/events.out.tfevents.* ' + os.path.join(trainer.log_dir, 'finetuning'))
            os.system('rm -r ' + trainer_pre.log_dir)
            os.system('mv ' + trainer.log_dir + ' ' + trainer_pre.log_dir)
