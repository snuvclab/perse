#!/bin/bash
source $HOME/conda/etc/profile.d/conda.sh
conda activate

#############################################
ENV_NAME=portrait_champ
DEVICE=4
WORK_DIR=$HOME/GitHub/perse_dev
DATASET_NAME=han
CONFIG_PATH=configs/inference/$DATASET_NAME/beard.yaml
#############################################

conda activate $ENV_NAME
export CUDA_VISIBLE_DEVICES=$DEVICE

cd $WORK_DIR/code/dataset_generator/portrait_champ
ntfy done python inference_ours.py --config $CONFIG_PATH