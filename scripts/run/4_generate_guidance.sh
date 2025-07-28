#!/bin/bash
source $HOME/conda/etc/profile.d/conda.sh
conda activate

#############################################
DEVICE=4
ENV_NAME=liveportrait
WORK_DIR=$HOME/GitHub/perse_dev
DATASET_NAME=han
SOURCE_DIR=$WORK_DIR/data/datasets/$DATASET_NAME/portrait.png
TARGET_DIR=driving_videos
DRIVING_DIR=$WORK_DIR/data/driving_videos/imavatar_mvi_1810_1812_limit.pkl
OUTPUT_DIR=$WORK_DIR/data/datasets/$DATASET_NAME/
START_IDX=0
END_IDX=None
#############################################

export CUDA_VISIBLE_DEVICES=$DEVICE
cd $WORK_DIR/code/submodules/LivePortrait

set -e
conda activate $ENV_NAME
ntfy done python inference_guidance.py --source_dir $SOURCE_DIR \
                                       --target_dir $TARGET_DIR \
                                       --driving_dir $DRIVING_DIR \
                                       --output_dir $OUTPUT_DIR \
                                       --start_idx $START_IDX \
                                       --end_idx $END_IDX
