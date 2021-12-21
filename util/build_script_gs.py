# This code builds the grid search scripts that will be launched on HPC

import os
import argparse

import numpy as np

from neuralnets.util.io import mkdir

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
DOMAINS = [VIB_EVHELA, VNC, MITOEM_H, EPFL, KASTHURI]
SRC_DOMAINS = '"' + '" "'.join(DOMAINS) + '"'

# methods
NO_DA = 'no-da'
MMD = 'mmd'
DAT = 'dat'
YNET = 'ynet'
UNET_TS = 'unet-ts'
METHODS = [MMD, DAT, YNET, UNET_TS]
PARAMS = {MMD: {'lambda_mmd': (3, 9, 1)}, DAT: {'lambda_dat': (-3, 3, 1)},
          YNET: {'lambda_rec': (-1, 5, 1)}, UNET_TS: {'lambda_o': (0, 9, 2), 'lambda_w': (-2, 7, 2)}}

# available labels
al = 0.20

parser = argparse.ArgumentParser()
parser.add_argument("--base_file", "-b", help="Path to the base script", required=True, type=str)
parser.add_argument("--coi", "-c", help="Class of interest", type=int, default=1)
parser.add_argument("--target_dir", "-t", help="Path to the directory where the scripts will be saved", required=True,
                    type=str)
args = parser.parse_args()

# load the base script
mkdir(args.target_dir)
with open(args.base_file, 'r') as f:
    lines = f.readlines()
    for method in METHODS:
        params = PARAMS[method]
        values = params.values()
        params = params.keys()
        prms = np.meshgrid(*[10**(np.arange(*v).astype(float)) for v in values])
        for n in range(prms[0].size):
            param_values = [str(p.item(n)) for p in prms]
            lines_ = []
            for line in lines:
                line = line.replace('<PARAMS>', '"' + ','.join(params) + '"')
                line = line.replace('<VALUES>', '"' + ','.join(param_values) + '"')
                line = line.replace('<METHOD>', method)
                line = line.replace('<COI>', str(args.coi))
                line = line.replace('<AVAILABLE_LABELS>', str(al))
                line = line.replace('<N>', str(n))
                lines_.append(line)

                with open(os.path.join(args.target_dir, 'run_%s_%d_%d.sh' % (method, n, args.coi)), 'w') as f:
                    for line in lines_:
                        f.write(line)
