
from networks.no_da import UNetNoDA2DClassifier
from networks.da import *

from neuralnets.util.io import print_frm


def generate_model(name, params):

    if name == 'u-net' or name == 'no-da':
        net = UNetDA2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                       dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                       activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'])
    elif name == 'mmd':
        net = UNetMMD2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                        dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                        activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                        lambda_mmd=params['lambda_mmd'])
    elif name == 'dat':
        net = UNetDAT2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                        dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                        activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                        lambda_dat=params['lambda_dat'], input_shape=params['input_size'])
    elif name == 'ynet':
        net = YNet2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                     dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                     activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                     lambda_rec=params['lambda_rec'])
    elif name == 'wnet':
        net = WNet2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                     dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                     activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                     lambda_rec=params['lambda_rec'], lambda_dat=params['lambda_dat'],
                     input_shape=params['input_size'])
    elif name == 'unet-ts':
        net = UNetTS2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                       dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                       activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'],
                       lambda_w=params['lambda_w'], lambda_o=params['lambda_o'])
    else:
        net = UNetDA2D(in_channels=params['in_channels'], feature_maps=params['fm'], levels=params['levels'],
                       dropout_enc=params['dropout'], dropout_dec=params['dropout'], norm=params['norm'],
                       activation=params['activation'], coi=params['coi'], loss_fn=params['loss'], lr=params['lr'])

    print_frm('Employed network: %s' % str(net.__class__.__name__))
    print_frm('    - Input channels: %d' % params['in_channels'])
    print_frm('    - Initial feature maps: %d' % params['fm'])
    print_frm('    - Levels: %d' % params['levels'])
    print_frm('    - Dropout: %.2f' % params['dropout'])
    print_frm('    - Normalization: %s' % params['norm'])
    print_frm('    - Activation: %s' % params['activation'])
    print_frm('    - Classes of interest: %s' % str(params['coi']))
    print_frm('    - Initial learning rate: %f' % params['lr'])

    return net


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
                                    input_shape=params['input_size'], len_epoch=params['len_epoch'])
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
                                   input_shape=params['input_size'], len_epoch=params['len_epoch'])
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
                                   input_shape=params['input_size'], len_epoch=params['len_epoch'])
    elif name == 'ynet':
        return YNet2DClassifier(dataset, epochs=params['epochs'], gpus=params['gpus'],
                                accelerator=params['accelerator'], log_dir=params['log_dir'],
                                log_freq=params['log_freq'], log_refresh_rate=params['log_refresh_rate'],
                                train_batch_size=params['train_batch_size'], test_batch_size=params['test_batch_size'],
                                num_workers=params['num_workers'], device=params['gpus'][0], transform=transform,
                                feature_maps=params['fm'], levels=params['levels'], dropout=params['dropout'],
                                norm=params['norm'], activation=params['activation'], coi=params['coi'],
                                loss_fn=params['loss'], partial_labels=(1, params['tar_labels_available']),
                                lambda_rec=params['lambda_rec'], input_shape=params['input_size'],
                                len_epoch=params['len_epoch'])
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
                                input_shape=params['input_size'], len_epoch=params['len_epoch'])
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
                                  lambda_o=params['lambda_o'], input_shape=params['input_size'],
                                  len_epoch=params['len_epoch'])
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
                                    input_shape=params['input_size'], len_epoch=params['len_epoch'])