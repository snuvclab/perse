#!/bin/bash
#############################################
DEVICE=6
ENV_NAME=portrait-editor
DATASET_NAME=portrait-release
PORTRAIT_IMAGE_PATH=/home/hyunsoocha/GitHub/perse-dev/assets/portrait.jpg
START_INDEX=0
END_INDEX=1000
#############################################

source ../common.sh
activate_env

cd $PORTRAIT_EDITOR_PATH
mkdir -p $DATASET_PATH/portrait-edit-image

ntfy done python run_portrait_editor.py \
    --input_image $PORTRAIT_IMAGE_PATH \
    --output_dir $DATASET_PATH/portrait-edit-image \
    --output_ext png \
    --csv_filename $DATASET_PATH/prompt/prompt.csv \
    --current_device $DEVICE \
    --start_index $START_INDEX \
    --end_index $END_INDEX \
    --seed 42