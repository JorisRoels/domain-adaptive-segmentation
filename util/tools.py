import shutil
from torch.utils.data import DataLoader

from neuralnets.data.datasets import LabeledVolumeDataset, LabeledSlidingWindowDataset
from neuralnets.util.tools import parse_params as parse_params_base
from neuralnets.util.io import print_frm
from neuralnets.util.augmentation import *


def get_transforms(tfs, coi=None):
    """
    Builds a transform object based on a list of desired augmentations

    :param tfs: list of augmentations, options: rot90, flipx, flipy, contrast, deformation, noise
    :param coi: classes of interest (only required if deformations are included)
    :return: transform object that implements the desired augmentations
    """

    # dictionary that maps augmentation strings to transform objects
    mapper = {'rot90': Rotate90(),
              'flipx': Flip(prob=0.5, dim=0),
              'flipy': Flip(prob=0.5, dim=1),
              'contrast': ContrastAdjust(adj=0.1),
              'deformation': RandomDeformation(),
              'noise': AddNoise(sigma_max=0.05)}

    # build the transforms
    tf_list = []
    for key in mapper:
        if key in tfs:
            tf_list.append(mapper[key])

    # required post-processing
    if 'deformation' in tfs:
        tf_list.append(CleanDeformedLabels(coi))

    return Compose(tf_list)


def get_dataloaders(params, domain=None, domain_labels_available=1.0, supervised=False):

    input_shape = (1, *(params['input_size']))
    transform = get_transforms(params['augmentation'], coi=params['coi'])
    print_frm('Applying data augmentation! Specifically %s' % str(params['augmentation']))

    if domain is None:

        split_src = params['src']['train_val_test_split']
        split_tar = params['tar']['train_val_test_split']
        print_frm('Train data... ')
        train = LabeledVolumeDataset((params['src']['data'], params['tar']['data']),
                                     (params['src']['labels'], params['tar']['labels']), len_epoch=params['len_epoch'],
                                     input_shape=input_shape, in_channels=params['in_channels'],
                                     type=params['type'], batch_size=params['train_batch_size'], transform=transform,
                                     range_split=((0, split_src[0]), (0, split_tar[0])), coi=params['coi'],
                                     range_dir=(params['src']['split_orientation'], params['tar']['split_orientation']),
                                     partial_labels=(1, params['tar_labels_available']), seed=params['seed'])
        print_frm('Validation data...')
        val = LabeledVolumeDataset((params['src']['data'], params['tar']['data']),
                                   (params['src']['labels'], params['tar']['labels']), len_epoch=params['len_epoch'],
                                   input_shape=input_shape, in_channels=params['in_channels'], type=params['type'],
                                   batch_size=params['test_batch_size'], coi=params['coi'],
                                   range_split=((split_src[0], split_src[1]), (split_tar[0], split_tar[1])),
                                   range_dir=(params['src']['split_orientation'], params['tar']['split_orientation']),
                                   partial_labels=(1, params['tar_labels_available']), seed=params['seed'])
        print_frm('Test data...')
        test = LabeledSlidingWindowDataset(params['tar']['data'], params['tar']['labels'],
                                           in_channels=params['in_channels'], type=params['type'],
                                           batch_size=params['test_batch_size'], range_split=(split_tar[1], 1),
                                           range_dir=params['tar']['split_orientation'], coi=params['coi'])

        print_frm('Train volume shape: %s (source) - %s (target)' % (str(train.data[0].shape), str(train.data[1].shape)))
        print_frm('Available target labels for training: %.1f (i.e. %.2f MV)' % (params['tar_labels_available']*100,
                                            np.prod(train.data[1].shape)*params['tar_labels_available'] / 1000 / 1000))
        print_frm('Validation volume shape: %s (source) - %s (target)' % (str(val.data[0].shape), str(val.data[1].shape)))
        print_frm('Test volume shape: %s (target)' % str(test.data[0].shape))

    else:

        split = params['train_val_test_split'] if supervised else params[domain]['train_val_test_split']
        data = params['data'] if supervised else params[domain]['data']
        labels = params['labels'] if supervised else params[domain]['labels']
        range_dir = params['split_orientation'] if supervised else params[domain]['split_orientation']
        print_frm('Train data...')
        train = LabeledVolumeDataset(data, labels, len_epoch=params['len_epoch'], input_shape=input_shape,
                                     in_channels=params['in_channels'], type=params['type'],
                                     batch_size=params['train_batch_size'], transform=transform,
                                     range_split=(0, split[0]), range_dir=range_dir,
                                     partial_labels=domain_labels_available, seed=params['seed'], coi=params['coi'])
        print_frm('Validation data...')
        val = LabeledVolumeDataset(data, labels, len_epoch=params['len_epoch'], input_shape=input_shape,
                                   in_channels=params['in_channels'], type=params['type'],
                                   batch_size=params['test_batch_size'], transform=transform,
                                   range_split=(split[0], split[1]), range_dir=range_dir, coi=params['coi'],
                                   partial_labels=domain_labels_available, seed=params['seed'])
        print_frm('Test data...')
        test = LabeledSlidingWindowDataset(data, labels, in_channels=params['in_channels'], type=params['type'],
                                           batch_size=params['test_batch_size'], transform=transform,
                                           range_split=(split[1], 1), range_dir=range_dir, coi=params['coi'])

        print_frm('Train volume shape: %s' % str(train.data[0].shape))
        print_frm('Available %s labels for training: %d%% (i.e. %.2f MV)' % (domain, domain_labels_available*100,
                                                    np.prod(train.data[0].shape)*domain_labels_available / 1000 / 1000))
        print_frm('Validation volume shape: %s' % str(val.data[0].shape))
        print_frm('Test volume shape: %s' % str(test.data[0].shape))

    train_loader = DataLoader(train, batch_size=params['train_batch_size'], num_workers=params['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val, batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                            pin_memory=True)
    test_loader = DataLoader(test, batch_size=params['test_batch_size'], num_workers=params['num_workers'],
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def parse_params(params):
    """
    Parse a YAML parameter dictionary

    :param params: dictionary containing the parameters
    :return: parsed dictionary
    """

    params = parse_params_base(params)

    keys = params.keys()

    if 'tar_labels_available' in keys:
        params['tar_labels_available'] = float(params['tar_labels_available'])

    if 'len_epoch' in keys:
        params['len_epoch'] = int(params['len_epoch'])
    else:
        params['len_epoch'] = 2000

    if 'augmentation' in keys:
        params['augmentation'] = params['augmentation'].split(',')
    else:
        params['augmentation'] = 'rot90,flipx,flipy,contrast,noise'.split(',')

    for dom in ['src', 'tar']:
        if dom in keys:
            ks = params[dom].keys()
            for split_key in ['train_val_test_split', 'train_val_split']:
                if split_key in ks:
                    if isinstance(params[dom][split_key], float):
                        params[dom][split_key] = [params[dom][split_key]]
                    else:
                        params[dom][split_key] = [float(item) for item in params[dom][split_key].split(',')]

    return params


def _correct_type(param, values):

    vs = values.split(';')
    values = []
    for v in vs:
        if param == 'feature_maps' or param == 'levels' or param == 'epochs':
            v_ = int(v)
        elif param == 'skip_connections' or param == 'residual_connections':
            v_ = bool(int(v))
        elif param == 'dropout' or param == 'lr' or param == 'lambda_mmd' or param == 'lambda_dat' or \
                param == 'lambda_rec' or param == 'lambda_o' or param == 'lambda_w':
            v_ = float(v)
        elif param == 'input_shape':
            v_ = [int(item) for item in v.split(',')]
        else:
            v_ = v
        values.append(v_)

    return param, values


def parse_search_grid(sg_str):

    sg = sg_str.split('#')
    search_grid = {}
    for s in sg:
        param, values = s.split(':')
        param, values = _correct_type(param, values)
        search_grid[param] = values

    return search_grid


def process_seconds(s):
    """
    Processes an amount of seconds to (hours, minutes, seconds)

    :param s: an amount of seconds
    :return: a tuple (h, m, s) that corresponds with the amount of hours, minutes and seconds, respectively
    """

    h = s // 3600
    s -= h*3600
    m = s // 60
    s -= m*60

    return h, m, s


def rmdir(dir):
    print_frm('    Removing %s' % dir)
    shutil.rmtree(dir, ignore_errors=True)


def cp(source, target):
    print_frm('    Copying %s -> %s' % (source, target))
    shutil.copyfile(source, target)


def mv(source, target):
    print_frm('    Moving %s -> %s' % (source, target))
    shutil.move(source, target)