
import os
import argparse
import yaml
from yaml.loader import SafeLoader

EPFL = 'EPFL'
UROCELL = 'UroCell'
PO936Q = 'po936q'
MITOEM_H = 'MitoEM-H'
MITOEM_R = 'MitoEM-R'
MIRA = 'MiRA'
KASTHURI = 'Kasthuri'
VNC = 'VNC'
EMBL_HELA = 'EMBL'
VIB_EVHELA = 'evhela'

# methods
NO_DA = 'no-da'
MMD = 'mmd'
DAT = 'dat'
YNET = 'ynet'
UNET_TS = 'unet-ts'


def _default_params():

    params = {'train_val_test_split': {}, 'split_orientation': {}, 'input_size': {}, 'coi': {}, 'method-params': {}}

    # method parameters
    params['method-params'][NO_DA] = []
    params['method-params'][MMD] = [('<LAMBDA_MMD>', 0.01)]
    params['method-params'][DAT] = [('<LAMBDA_DAT>', 0.01)]
    params['method-params'][YNET] = [('<LAMBDA_REC>', 100)]
    params['method-params'][UNET_TS] = [('<LAMBDA_O>', 10000), ('<LAMBDA_W>', 100)]
    for method in [NO_DA, MMD, DAT, UNET_TS]:
        params['dropout'][method] = '0.00'
    params['dropout'][YNET] = '0.25'

    # train/val/test split parameters
    params['train_val_test_split'][EPFL] = '0.40,0.50'
    for DOM in [UROCELL, PO936Q, MITOEM_H, MITOEM_R, VIB_EVHELA]:
        params['train_val_test_split'][DOM] = '0.48,0.60'
    params['train_val_test_split'][VNC] = '0.30,0.50'
    params['train_val_test_split'][MIRA] = '0.50,0.70'
    params['train_val_test_split'][KASTHURI] = '0.426,0.532'
    params['train_val_test_split'][EMBL_HELA] = '0.50,0.75'

    # split orientation parameters
    for DOM in [EPFL, UROCELL, PO936Q, MITOEM_H, MITOEM_R, KASTHURI, VIB_EVHELA, EMBL_HELA]:
        params['split_orientation'][DOM] = 'z'
    for DOM in [MIRA, VNC]:
        params['split_orientation'][DOM] = 'y'

    # input size parameters
    for DOM in [EPFL, MITOEM_H, MITOEM_R, MIRA, KASTHURI]:
        params['input_size'][DOM] = '512,512'
    for DOM in [PO936Q, VIB_EVHELA]:
        params['input_size'][DOM] = '448,448'
    for DOM in [UROCELL, EMBL_HELA]:
        params['input_size'][DOM] = '256,256'
    params['input_size'][VNC] = '192,192'

    return params


def _get_sz(src_sz, tar_sz):
    ssz = int(str.split(src_sz, ',')[0])
    tsz = int(str.split(tar_sz, ',')[0])
    sz = min(ssz, tsz)
    return str(sz) + ',' + str(sz)


parser = argparse.ArgumentParser()
parser.add_argument("--base_file", "-b", help="Path to the base configuration file", type=str, default='clem1.yaml')
parser.add_argument("--src-domain", "-ds", help="Source domain", type=str, default='EPFL')
parser.add_argument("--tar-domain", "-dt", help="Target domain", type=str, default='VNC')
parser.add_argument("--method", "-m", help="Method to use", type=str, default='unet-ts')
parser.add_argument("--available-labels", "-al", help="Fraction of available target labels", type=float, default=0.0)
parser.add_argument("--gpu", "-g", help="Index of the GPU computing device", type=int, default=0)
parser.add_argument("--coi", "-c", help="Class of interest (1: mito, 2: er, 3: nm)", type=int, default=1)
parser.add_argument("--params", "-p", help="Parameters that should be set (e.g. <LAMBDA_MMD>)", type=str, default="")
parser.add_argument("--values", "-v", help="Values of the parameters that should be set", type=str, default="")
parser.add_argument("--n_param", "-n", help="Index of the parameter experiment", type=int, default=0)
args = parser.parse_args()

if len(args.params) != 0 and len(args.values) != 0:
    args.params = args.params.split(',')
    args.values = [float(v) for v in args.values.split(',')]
else:
    args.params = []
    args.values = []

# get default parameters
params = _default_params()

# read and adjust template
with open(args.base_file, 'r') as f:
    data = yaml.load(f, Loader=SafeLoader)
    sz = _get_sz(params['input_size'][args.src_domain], params['input_size'][args.tar_domain])
    method_params = params['method-params'][args.method]
    if len(args.params) > 0:
        # override default params if necessary
        method_params = [(p, v) for p, v in zip(args.params, args.values)]
    logdir = args.method + '-' + str(args.n_param) + '-' + args.src_domain + '2' + args.tar_domain
    for k in data.keys():
        if type(data[k]) == str:
            data[k] = data[k].replace('<LOG_DIR>', logdir)
            data[k] = data[k].replace('<METHOD>', args.method)
            data[k] = data[k].replace('<INPUT_SIZE>', sz)
            if data[k] == '<COI>':
                data[k] = '0,' + str(args.coi)
            elif data[k] == '<AVAILABLE_LABELS>':
                data[k] = args.available_labels
            elif data[k] == '<GPU>':
                data[k] = args.gpu
            else:
                for p, v in method_params:
                    if data[k] == p:
                        data[k] = v
                data[k] = data[k].replace('<DROPOUT>', params['dropout'][args.method])
        elif type(data[k]) == dict:
            domain = args.src_domain if k == 'src' else args.tar_domain
            for l in data[k].keys():
                data[k][l] = data[k][l].replace('<DOMAIN>', domain)
                data[k][l] = data[k][l].replace('<TRAIN_VAL_TEST_SPLIT>', params['train_val_test_split'][domain])
                data[k][l] = data[k][l].replace('<SPLIT_ORIENTATION>', params['split_orientation'][domain])


# write config file
with open(os.path.join(os.path.dirname(args.base_file), logdir + '.yaml'), 'w') as f:
    documents = yaml.dump(data, f)
