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
DOMAINS = [VIB_EVHELA, VNC, MITOEM_H, EPFL, KASTHURI]
SRC_DOMAINS = '"' + '" "'.join(DOMAINS) + '"'

# methods
NO_DA = 'no-da'
MMD = 'mmd'
DAT = 'dat'
YNET = 'ynet'
UNET_TS = 'unet-ts'
METHODS = [NO_DA, MMD, DAT, YNET, UNET_TS]

# available labels
AVAILABLE_LABELS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]


parser = argparse.ArgumentParser()
parser.add_argument("--base_file", "-b", help="Path to the base script", required=True, type=str)
parser.add_argument("--target_dir", "-t", help="Path to the directory where the scripts will be saved", required=True,
                    type=str)
parser.add_argument("--src_domains", "-sd", help="List of source domains of interest", required=True, type=str)
parser.add_argument("--tar_domains", "-td", help="List of target domains of interest", required=True, type=str)
parser.add_argument("--methods", "-m", help="List of methods of interest", required=True, type=str)
parser.add_argument("--available_labels", "-al", help="List of available labels of interest", required=True, type=str)
parser.add_argument("--coi", "-c", help="Class of interest", type=int, default=1)
args = parser.parse_args()
args.src_domains = args.src_domains.split(',')
args.tar_domains = args.tar_domains.split(',')
args.methods = args.methods.split(',')
args.available_labels = [float(x) for x in args.available_labels.split(',')]

# load the base script
mkdir(args.target_dir)
with open(args.base_file, 'r') as f:
    lines = f.readlines()
    for method in args.methods:
        for i, al in enumerate(args.available_labels):
            for src_domain in args.src_domains:
                for tar_domain in args.tar_domains:
                    if src_domain != tar_domain:

                        lines_ = []
                        for line in lines:
                            line = line.replace('<METHOD>', method)
                            line = line.replace('<AVAILABLE_LABELS>', str(al))
                            line = line.replace('<SRC_DOMAINS>', '"' + src_domain + '"')
                            line = line.replace('<TAR_DOMAINS>', '"' + tar_domain + '"')
                            lines_.append(line)

                        with open(os.path.join(args.target_dir, 'run_%s_%d_%s2%s.sh' % (method, i, src_domain, tar_domain)), 'w') as f:
                            for line in lines_:
                                f.write(line)

# replace the method, available labels, source and target domains