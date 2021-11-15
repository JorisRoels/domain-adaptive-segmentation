"""
    Necessary libraries
"""
import os
import time
import torch

import pytorch_lightning as pl

from neuralnets.util.io import print_frm
from neuralnets.util.validation import validate as validate_base

from util.tools import process_seconds


def train(net, train_loader, val_loader, callbacks, params, reduce_epochs=False):

    # trains a model with a specific training and validation loader, and manually specified callbacks
    t_start = time.perf_counter()
    epochs = params['epochs'] // 2 if reduce_epochs else params['epochs']
    trainer = pl.Trainer(max_epochs=epochs, gpus=params['gpus'], accelerator=params['accelerator'],
                         default_root_dir=params['log_dir'], flush_logs_every_n_steps=params['log_freq'],
                         log_every_n_steps=params['log_freq'], callbacks=callbacks,
                         progress_bar_refresh_rate=params['log_refresh_rate'])
    trainer.fit(net, train_loader, val_loader)
    t_stop = time.perf_counter()
    print_frm('Elapsed training time: %d hours, %d minutes, %.2f seconds' % process_seconds(t_stop - t_start))

    # load the best checkpoint
    net.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])

    return trainer


def validate(net, trainer, loader, params):
    # validates a network that was trained using a specific trainer on a dataset
    t_start = time.perf_counter()
    test_data, test_labels = loader.dataset.data[0], loader.dataset.labels[0]
    validate_base(net, test_data, test_labels, params['input_size'], in_channels=params['in_channels'],
                  classes_of_interest=params['coi'], batch_size=params['test_batch_size'],
                  write_dir=os.path.join(trainer.log_dir, 'best_predictions'),
                  val_file=os.path.join(trainer.log_dir, 'metrics.npy'), device=params['gpus'][0])
    t_stop = time.perf_counter()
    print_frm('Elapsed testing time: %d hours, %d minutes, %.2f seconds' % process_seconds(t_stop - t_start))