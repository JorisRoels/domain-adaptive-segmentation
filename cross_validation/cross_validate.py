"""
    This is a script that illustrates grid search cross validation to optimize parameters for domain adaptive training
"""

"""
    Necessary libraries
"""
import argparse
import yaml
import os

from neuralnets.data.datasets import LabeledVolumeDataset
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm, save
from neuralnets.util.tools import set_seed, log_hparams

from multiprocessing import freeze_support
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader

from util.tools import parse_params, parse_search_grid, get_transforms
from networks.factory import generate_classifier


if __name__ == '__main__':
    freeze_support()

    """
        Parse all the arguments
    """
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the configuration file", type=str,
                        default='cross_validate.yaml')
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
    transform = get_transforms(params['augmentation'], coi=params['coi'])
    print_frm('Training data...')
    train = LabeledVolumeDataset((params['src']['data'], params['tar']['data']),
                                 (params['src']['labels'], params['tar']['labels']), len_epoch=params['len_epoch'],
                                 input_shape=input_shape, in_channels=params['in_channels'],
                                 type=params['type'], batch_size=params['train_batch_size'], transform=transform,
                                 range_split=((0, split_src[0]), (0, split_tar[0])), coi=params['coi'],
                                 range_dir=(params['src']['split_orientation'], params['tar']['split_orientation']),
                                 partial_labels=(1, 1), seed=params['seed'])
    loader = DataLoader(train, batch_size=params['train_batch_size'], num_workers=params['num_workers'])
    X_train = np.linspace(0, 1 - 1 / len(train), num=len(train))
    y_train = np.random.randint(2, size=(len(train)))

    """
        Build the network
    """
    print_frm('Building the network')
    clf = generate_classifier(params['method'], params, train, transform)

    """
        Perform cross validation grid search
    """
    print_frm('Starting grid search cross validation')
    search_grid = parse_search_grid(params['search_grid'])
    gs = GridSearchCV(clf, search_grid, cv=params['folds'], verbose=4)
    gs.fit(X_train, y_train)

    """
        Save and report results
    """
    save(gs, params['results_file'])
    hparams_dir = os.path.join(params['log_dir'], 'hparams')
    log_hparams(gs, log_dir=hparams_dir)
    print_frm(gs.best_params_)
    print_frm('Best mIoU: %.6f' % gs.best_score_)

    print_frm('Finished!')
