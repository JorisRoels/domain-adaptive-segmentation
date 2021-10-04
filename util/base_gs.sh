#!/bin/bash -l
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=24:00:00

# define context variables
PYTHON_EXE=$PYTHON_DEFAULT

# define environment variables
PROJECT_DIR=$HOME/research/domain-adaptive-segmentation
export PYTHONPATH=$PYTHONPATH:$HOME/research/lib/python:$HOME/research/neuralnets:$PROJECT_DIR

# DOMAINS=("EPFL" "UroCell" "po936q" "MitoEM-H" "MitoEM-R" "MiRA" "Kasthuri" "VNC" "EMBL" "evhela")
DOMAIN_SRC="MiRA"
DOMAIN_TAR="Kasthuri"

METHOD=<METHOD>
AVAILABLE_LABELS=0.20
PARAMS=<PARAMS>
VALUES=<VALUES>
COI=1

# run experiments (first build the script, then run it)
CONFIG_FILE=${METHOD}-<N_PARAM>-${DOMAIN_SRC}2${DOMAIN_TAR}
python $PROJECT_DIR/util/build_config_gs.py -b $PROJECT_DIR/train/gs/config/base.yaml -ds ${DOMAIN_SRC} -dt ${DOMAIN_TAR} -m $METHOD -al $AVAILABLE_LABELS -c $COI -g 0 -p $PARAMS -v $VALUES
python $PROJECT_DIR/train/train_semi_supervised.py -c $PROJECT_DIR/train/gs/config/${CONFIG_FILE}.yaml --clean-up &> $PROJECT_DIR/train/gs/logs/${CONFIG_FILE}.logs

DOMAIN_SRC="EPFL"
DOMAIN_TAR="MitoEM-R"

# run experiments (first build the script, then run it)
CONFIG_FILE=${METHOD}-<N_PARAM>-${DOMAIN_SRC}2${DOMAIN_TAR}
python $PROJECT_DIR/util/build_config_gs.py -b $PROJECT_DIR/train/gs/config/base.yaml -ds ${DOMAIN_SRC} -dt ${DOMAIN_TAR} -m $METHOD -al $AVAILABLE_LABELS -c $COI -g 0 -p $PARAMS -v $VALUES
python $PROJECT_DIR/train/train_semi_supervised.py -c $PROJECT_DIR/train/gs/config/${CONFIG_FILE}.yaml --clean-up &> $PROJECT_DIR/train/gs/logs/${CONFIG_FILE}.logs

DOMAIN_SRC="VNC"
DOMAIN_TAR="evhela"

# run experiments (first build the script, then run it)
CONFIG_FILE=${METHOD}-<N_PARAM>-${DOMAIN_SRC}2${DOMAIN_TAR}
python $PROJECT_DIR/util/build_config_gs.py -b $PROJECT_DIR/train/gs/config/base.yaml -ds ${DOMAIN_SRC} -dt ${DOMAIN_TAR} -m $METHOD -al $AVAILABLE_LABELS -c $COI -g 0 -p $PARAMS -v $VALUES
python $PROJECT_DIR/train/train_semi_supervised.py -c $PROJECT_DIR/train/gs/config/${CONFIG_FILE}.yaml --clean-up &> $PROJECT_DIR/train/gs/logs/${CONFIG_FILE}.logs
