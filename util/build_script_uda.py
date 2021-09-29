# This code builds the scripts that will be launched on HPC

import os
import argparse

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
DOMAINS = [EPFL, UROCELL, PO936Q, MITOEM_H, MITOEM_R, MIRA, KASTHURI, VNC, VIB_EVHELA]
SRC_DOMAINS = '"' + '" "'.join(DOMAINS) + '"'

# methods
NO_DA = 'no-da'
MMD = 'mmd'
DAT = 'dat'
YNET = 'ynet'
UNET_TS = 'unet-ts'
METHODS = [NO_DA, MMD, DAT, YNET, UNET_TS]

# available labels
AVAILABLE_LABELS = [0.0]


parser = argparse.ArgumentParser()
parser.add_argument("--base_file", "-b", help="Path to the base script", required=True, type=str)
parser.add_argument("--target_dir", "-t", help="Path to the directory where the scripts will be saved", required=True,
                    type=str)
args = parser.parse_args()

# load the base script
mkdir(args.target_dir)
with open(args.base_file, 'r') as f:
    lines = f.readlines()
    for method in METHODS:
        for i, al in enumerate(AVAILABLE_LABELS):
            for tar_domain in DOMAINS:

                lines_ = []
                for line in lines:
                    line = line.replace('<METHOD>', method)
                    line = line.replace('<AVAILABLE_LABELS>', str(al))
                    line = line.replace('<SRC_DOMAINS>', SRC_DOMAINS)
                    line = line.replace('<TAR_DOMAINS>', '"' + tar_domain + '"')
                    lines_.append(line)

                with open(os.path.join(args.target_dir, 'run_%s_%d_%s.sh' % (method, i, tar_domain)), 'w') as f:
                    for line in lines_:
                        f.write(line)

# replace the method, available labels, source and target domains