#!/bin/bash -l
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=28:00:00
module purge
module load PyTorch
module unload PyTorch

# define context variables
PYTHON_EXE=python

# define environment variables
PROJECT_DIR=$HOME/research/domain-adaptive-segmentation
export PYTHONPATH=$HOME/research/neuralnets:$HOME/research/lib/python:$PROJECT_DIR
LOG_DIR=ssda

# DOMAINS=("EPFL" "UroCell" "po936q" "MitoEM-H" "MitoEM-R" "MiRA" "Kasthuri" "VNC" "EMBL" "evhela")
DOMAINS_SRC=(<SRC_DOMAINS>)
DOMAINS_TAR=(<TAR_DOMAINS>)

METHOD=<METHOD>

AVAILABLE_LABELS=<AVAILABLE_LABELS>

COI=<COI>

# number of experiments
N=${#DOMAINS_SRC[@]}
M=${#DOMAINS_TAR[@]}

# run experiments (first build the script, then run it)
for (( i=0; i<$N; i++ ))
do
    for (( j=0; j<$M; j++ ))
    do
        if [[ ${DOMAINS_SRC[$i]} != ${DOMAINS_TAR[$j]} ]]
        then
            CONFIG_FILE=${METHOD}-${AVAILABLE_LABELS}-${DOMAINS_SRC[$i]}2${DOMAINS_TAR[$j]}
	          $PYTHON_EXE $PROJECT_DIR/util/build_config_da.py -b $PROJECT_DIR/train/semi-supervised-da/config/base.yaml -ds ${DOMAINS_SRC[$i]} -dt ${DOMAINS_TAR[$j]} -m $METHOD -al $AVAILABLE_LABELS -c $COI -g 0
	          $PYTHON_EXE $PROJECT_DIR/train/train_semi_supervised.py -c $PROJECT_DIR/train/semi-supervised-da/config/${CONFIG_FILE}.yaml --clean-up &> $PROJECT_DIR/train/semi-supervised-da/logs/${CONFIG_FILE}.logs
        fi
    done
done
