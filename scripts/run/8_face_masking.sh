#!/bin/bash
set -e

###################################
DEVICE=4
WORK_DIR=$HOME/GitHub/perse_dev
DATASET_NAME=han
TARGET_DIR=$WORK_DIR/data/datasets/$DATASET_NAME/synthetic_dataset
START_IDX=1
END_IDX=5
ENV_NAME=portrait_editor
EXT=png
###################################
SAPIENS_DIR=$WORK_DIR/code/submodules/sapiens/seg/scripts/demo/local/
SAPIENS_CHECKPOINT_ROOT=$WORK_DIR/code/submodules/sapiens/sapiens_host
###################################

export CUDA_VISIBLE_DEVICES=$DEVICE
source $HOME/conda/etc/profile.d/conda.sh
conda activate $ENV_NAME

TARGET_DIR_PARENTS=($(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | sort))

# 디렉토리 안의 모든 하위 디렉토리를 찾고 루프를 돌립니다.
for ((i=START_IDX; i<=END_IDX && i<${#TARGET_DIR_PARENTS[@]}; i++)); do
    TARGET_DIR="${TARGET_DIR_PARENTS[i]}"
    echo "[INFO] Processing $TARGET_DIR"

    echo "[INFO] Sapiens Masking and Segmentation"
    cd $SAPIENS_DIR
    bash seg_face_png_only.sh $TARGET_DIR/image $TARGET_DIR/segment $EXT $DEVICE $SAPIENS_CHECKPOINT_ROOT $ENV_NAME
done