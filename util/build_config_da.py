
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
    params['method-params'][MMD] = [('<LAMBDA_MMD>', 1e-2)]
    params['method-params'][DAT] = [('<LAMBDA_DAT>', 1e-2)]
    params['method-params'][YNET] = [('<LAMBDA_REC>', 1e2)]
    params['method-params'][UNET_TS] = [('<LAMBDA_O>', 1e4), ('<LAMBDA_W>', 1e2)]

    # train/val/test split parameters
    params['train_val_test_split'][EPFL] = '0.40,0.50'
    for DOM in [UROCELL, PO936Q, MITOEM_H, MITOEM_R, MIRA, VIB_EVHELA]:
        params['train_val_test_split'][DOM] = '0.48,0.60'
    params['train_val_test_split'][VNC] = '0.30,0.50'
    params['train_val_test_split'][KASTHURI] = '0.426,0.532'
    params['train_val_test_split'][EMBL_HELA] = '0.40,0.65'

    # split orientation parameters
    for DOM in [EPFL, UROCELL, PO936Q, MITOEM_H, MITOEM_R, KASTHURI, VIB_EVHELA]:
        params['split_orientation'][DOM] = 'z'
    for DOM in [MIRA, VNC, EMBL_HELA]:
        params['split_orientation'][DOM] = 'y'

    # input size parameters
    for DOM in [MITOEM_H, MITOEM_R, MIRA, KASTHURI]:
        params['input_size'][DOM] = '768,768'
    params['input_size'][EPFL] = '512,512'
    for DOM in [PO936Q, VIB_EVHELA]:
        params['input_size'][DOM] = '448,448'
    params['input_size'][UROCELL] = '256,256'
    params['input_size'][VNC] = '192,192'
    params['input_size'][EMBL_HELA] = '128,128'

    # classes of interest parameters
    for DOM in [EPFL, UROCELL, MITOEM_H, MITOEM_R, MIRA, KASTHURI, VNC, PO936Q]:
        params['coi'][DOM] = '0,1'
    params['coi'][EMBL_HELA] = '0,2'
    params['coi'][VIB_EVHELA] = '0,1,2,3'

    return params


def _get_sz(src_sz, tar_sz):
    ssz = int(str.split(src_sz, ',')[0])
    tsz = int(str.split(tar_sz, ',')[0])
    sz = min(ssz, tsz)
    return str(sz) + ',' + str(sz)


parser = argparse.ArgumentParser()
parser.add_argument("--base_file", "-b", help="Path to the base configuration file", type=str, default='base.yaml')
parser.add_argument("--src-domain", "-ds", help="Source domain", type=str, default='EPFL')
parser.add_argument("--tar-domain", "-dt", help="Target domain", type=str, default='VNC')
parser.add_argument("--method", "-m", help="Method to use", type=str, default='unet-ts')
parser.add_argument("--gpu", "-g", help="Index of the GPU computing device", type=int, default=0)
args = parser.parse_args()

# get default parameters
params = _default_params()

# read and adjust template
with open(args.base_file, 'r') as f:
    data = yaml.load(f, Loader=SafeLoader)
    sz = _get_sz(params['input_size'][args.src_domain], params['input_size'][args.tar_domain])
    method_params = params['method-params'][args.method]
    for k in data.keys():
        if type(data[k]) == str:
            data[k] = data[k].replace('<DOMAIN>', args.src_domain + '2' + args.tar_domain)
            for m_param in method_params:
                data[k] = data[k].replace(m_param[0], str(m_param[1]))
            data[k] = data[k].replace('<METHOD>', args.method)
            data[k] = data[k].replace('<INPUT_SIZE>', sz)
            data[k] = data[k].replace('<COI>', params['coi'][args.tar_domain])
            if data[k] == '<GPU>':
                data[k] = args.gpu
        elif type(data[k]) == dict:
            domain = args.src_domain if k == 'src' else args.tar_domain
            for l in data[k].keys():
                data[k][l] = data[k][l].replace('<DOMAIN>', domain)
                data[k][l] = data[k][l].replace('<TRAIN_VAL_TEST_SPLIT>', params['train_val_test_split'][domain])
                data[k][l] = data[k][l].replace('<SPLIT_ORIENTATION>', params['split_orientation'][domain])


# write config file
with open(os.path.join(os.path.dirname(args.base_file), args.src_domain + '2' + args.tar_domain + '.yaml'), 'w') as f:
    documents = yaml.dump(data, f)
