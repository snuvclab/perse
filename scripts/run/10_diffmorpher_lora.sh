#!/bin/zsh
set -e 

source $HOME/conda/etc/profile.d/conda.sh
conda activate

############################
ENV_NAME=perse_avatar
WORK_DIR=$HOME/GitHub/perse-dev
DEVICE=1
DATASET_NAME=supp_video
START_IDX=0
END_IDX=12
LORA_EPOCHS=5
############################

MODEL_NAME=stabilityai/stable-diffusion-2-1-base
DATASET_PATH=$WORK_DIR/data/datasets/$DATASET_NAME/synthetic_dataset

conda activate $ENV_NAME
export CUDA_VISIBLE_DEVICES=$DEVICE

cd $WORK_DIR'/code/submodules/diffmorpher-for-perse'
ntfy done python main.py --model_path $MODEL_NAME \
                         --dataset_path $DATASET_PATH \
                         --start_idx $START_IDX \
                         --end_idx $END_IDX \
                         --lora_epochs $LORA_EPOCHS \
                         --use_adain \
                         --save_inter