#!/bin/bash
#############################################
DEVICE=0
ENV_NAME=portrait-editor
DATASET_NAME=portrait-release
#############################################

source ../common.sh
activate_env

mkdir -p $DATASET_PATH/prompt
mkdir -p $DATASET_PATH/index_range
cd $PORTRAIT_EDITOR_PATH

ntfy done python prompt_generator.py \
    --output_prompt_csv_path $DATASET_PATH/prompt/prompt.csv \
    --output_index_range_path $DATASET_PATH/index_range/index_range.csv \
    --attribute_features $PORTRAIT_EDITOR_PATH/config/attribute_small.yaml