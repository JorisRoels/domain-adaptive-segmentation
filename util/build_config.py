
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

# fill in parameters
params = {'train_val_test_split': {}, 'split_orientation': {}, 'input_size': {}, 'coi': {}}
for DOM in [EPFL, EMBL_HELA]:
    params['train_val_test_split'][DOM] = '0.25,0.50'
for DOM in [UROCELL, PO936Q, MITOEM_H, MITOEM_R, MIRA, VNC, VIB_EVHELA]:
    params['train_val_test_split'][DOM] = '0.30,0.60'
params['train_val_test_split'][KASTHURI] = '0.266,0.532'
for DOM in [EPFL, UROCELL, PO936Q, MITOEM_H, MITOEM_R, KASTHURI, VIB_EVHELA]:
    params['split_orientation'][DOM] = 'z'
for DOM in [MIRA, VNC, EMBL_HELA]:
    params['split_orientation'][DOM] = 'y'
for DOM in [EPFL, MITOEM_H, MITOEM_R, MIRA, KASTHURI]:
    params['input_size'][DOM] = '512,512'
for DOM in [PO936Q, VIB_EVHELA]:
    params['input_size'][DOM] = '448,448'
params['input_size'][UROCELL] = '256,256'
for DOM in [VNC, EMBL_HELA]:
    params['input_size'][DOM] = '128,128'
for DOM in [EPFL, UROCELL, MITOEM_H, MITOEM_R, MIRA, KASTHURI, VNC]:
    params['coi'][DOM] = '0,1'
params['coi'][EMBL_HELA] = '0,2'
params['coi'][PO936Q] = '0,1,3'
params['coi'][VIB_EVHELA] = '0,1,2,3'

parser = argparse.ArgumentParser()
parser.add_argument("--base_file", "-b", help="Path to the base configuration file", type=str, default='base.yaml')
parser.add_argument("--domain", "-d", help="Target domain", type=str, default='epfl')
parser.add_argument("--gpu", "-g", help="Index of the GPU computing device", type=int, default=0)
args = parser.parse_args()

# read and adjust template
with open(args.base_file, 'r') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for k in data.keys():
        if type(data[k]) == str:
            data[k] = data[k].replace('<DOMAIN>', args.domain)
            data[k] = data[k].replace('<TRAIN_VAL_TEST_SPLIT>', params['train_val_test_split'][args.domain])
            data[k] = data[k].replace('<SPLIT_ORIENTATION>', params['split_orientation'][args.domain])
            data[k] = data[k].replace('<INPUT_SIZE>', params['input_size'][args.domain])
            data[k] = data[k].replace('<COI>', params['coi'][args.domain])
            if data[k] == '<GPU>':
                data[k] = args.gpu

# write config file
with open(os.path.join(os.path.dirname(args.base_file), args.domain + '.yaml'), 'w') as f:
    documents = yaml.dump(data, f)
