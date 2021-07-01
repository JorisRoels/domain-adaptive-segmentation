
from networks.no_da import UNetNoDA2D, UNetNoDA2DClassifier
from networks.da import *


def generate_model(name, params):

    if name == 'u-net':
        return UNetDA2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                        dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                        activation=params['activation'], coi=params['coi'], loss_fn=params['loss'])
    elif name == 'no-da':
        return UNetNoDA2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                          dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                          activation=params['activation'], coi=params['coi'], loss_fn=params['loss'])
    elif name == 'mmd':
        return UNetMMD2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                         dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                         activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                         lambda_mmd=params['lambda_mmd'])
    elif name == 'dat':
        return UNetDAT2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                         dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                         activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                         lambda_dat=params['lambda_dat'], input_shape=params['input_size'])
    elif name == 'ynet':
        return YNet2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                      dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                      activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                      lambda_rec=params['lambda_rec'])
    elif name == 'wnet':
        return WNet2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                      dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                      activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                      lambda_rec=params['lambda_rec'], lambda_dat=params['lambda_dat'],
                      input_shape=params['input_size'])
    elif name == 'unet-ts':
        return UNetTS2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                        dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                        activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                        lambda_w=params['lambda_w'], lambda_o=params['lambda_o'])
    else:
        return UNetNoDA2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                          dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                          activation=params['activation'], coi=params['coi'], loss_fn=params['loss'])


def generate_classifier(name, params, dataset, transform):

    if name == 'no-da':
        return UNetNoDA2DClassifier(dataset, epochs=params['epochs'], gpus=params['gpus'],
                                    accelerator=params['accelerator'], log_dir=params['log_dir'],
                                    log_freq=params['log_freq'], log_refresh_rate=params['log_refresh_rate'],
                                    train_batch_size=params['train_batch_size'],
                                    test_batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                                    device=params['gpus'][0], transform=transform, feature_maps=params['fm'],
                                    levels=params['levels'], dropout=params['dropout'], norm=params['norm'],
                                    activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                                    partial_labels=(1, params['tar_labels_available']),
                                    input_shape=params['input_size'])
    elif name == 'mmd':
        return UNetMMD2DClassifier(dataset, epochs=params['epochs'], gpus=params['gpus'],
                                   accelerator=params['accelerator'], log_dir=params['log_dir'],
                                   log_freq=params['log_freq'], log_refresh_rate=params['log_refresh_rate'],
                                   train_batch_size=params['train_batch_size'],
                                   test_batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                                   device=params['gpus'][0], transform=transform, feature_maps=params['fm'],
                                   levels=params['levels'], dropout=params['dropout'], norm=params['norm'],
                                   activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                                   partial_labels=(1, params['tar_labels_available']), lambda_mmd=params['lambda_mmd'],
                                   input_shape=params['input_size'])
    elif name == 'dat':
        return UNetDAT2DClassifier(dataset, epochs=params['epochs'], gpus=params['gpus'],
                                   accelerator=params['accelerator'], log_dir=params['log_dir'],
                                   log_freq=params['log_freq'], log_refresh_rate=params['log_refresh_rate'],
                                   train_batch_size=params['train_batch_size'],
                                   test_batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                                   device=params['gpus'][0], transform=transform, feature_maps=params['fm'],
                                   levels=params['levels'], dropout=params['dropout'], norm=params['norm'],
                                   activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                                   partial_labels=(1, params['tar_labels_available']), lambda_dat=params['lambda_dat'],
                                   input_shape=params['input_size'])
    elif name == 'ynet':
        return YNet2DClassifier(dataset, epochs=params['epochs'], gpus=params['gpus'],
                                accelerator=params['accelerator'], log_dir=params['log_dir'],
                                log_freq=params['log_freq'], log_refresh_rate=params['log_refresh_rate'],
                                train_batch_size=params['train_batch_size'], test_batch_size=params['test_batch_size'],
                                num_workers=params['num_workers'], device=params['gpus'][0], transform=transform,
                                feature_maps=params['fm'], levels=params['levels'], dropout=params['dropout'],
                                norm=params['norm'], activation=params['activation'], coi=params['coi'],
                                loss_fn=params['loss'], partial_labels=(1, params['tar_labels_available']),
                                lambda_rec=params['lambda_rec'], input_shape=params['input_size'])
    elif name == 'wnet':
        return WNet2DClassifier(dataset, epochs=params['epochs'], gpus=params['gpus'],
                                accelerator=params['accelerator'], log_dir=params['log_dir'],
                                log_freq=params['log_freq'], log_refresh_rate=params['log_refresh_rate'],
                                train_batch_size=params['train_batch_size'], test_batch_size=params['test_batch_size'],
                                num_workers=params['num_workers'], device=params['gpus'][0], transform=transform,
                                feature_maps=params['fm'], levels=params['levels'], dropout=params['dropout'],
                                norm=params['norm'], activation=params['activation'], coi=params['coi'],
                                loss_fn=params['loss'], partial_labels=(1, params['tar_labels_available']),
                                lambda_rec=params['lambda_rec'], lambda_dat=params['lambda_dat'],
                                input_shape=params['input_size'])
    elif name == 'unet-ts':
        return UNetTS2DClassifier(dataset, epochs=params['epochs'], gpus=params['gpus'],
                                  accelerator=params['accelerator'], log_dir=params['log_dir'],
                                  log_freq=params['log_freq'], log_refresh_rate=params['log_refresh_rate'],
                                  train_batch_size=params['train_batch_size'],
                                  test_batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                                  device=params['gpus'][0], transform=transform, feature_maps=params['fm'],
                                  levels=params['levels'], dropout=params['dropout'], norm=params['norm'],
                                  activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                                  partial_labels=(1, params['tar_labels_available']), lambda_w=params['lambda_w'],
                                  lambda_o=params['lambda_o'], input_shape=params['input_size'])
    else:
        return UNetNoDA2DClassifier(dataset, epochs=params['epochs'], gpus=params['gpus'],
                                    accelerator=params['accelerator'], log_dir=params['log_dir'],
                                    log_freq=params['log_freq'], log_refresh_rate=params['log_refresh_rate'],
                                    train_batch_size=params['train_batch_size'],
                                    test_batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                                    device=params['gpus'][0], transform=transform, feature_maps=params['fm'],
                                    levels=params['levels'], dropout=params['dropout'], norm=params['norm'],
                                    activation=params['activation'], coi=params['coi'], loss_fn=params['loss'],
                                    partial_labels=(1, params['tar_labels_available']),
                                    input_shape=params['input_size'])