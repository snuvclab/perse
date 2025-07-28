#!/bin/bash
# set -e

###################################
DEVICE=4
WORK_DIR=$HOME/GitHub/perse_dev
DATASET_NAME=han
TARGET_DIR=$WORK_DIR/data/datasets/$DATASET_NAME/synthetic_dataset
START_IDX=1
END_IDX=5
ENV_NAME=emoca
RESIZE=512
FX=1539.67462
FY=1508.93280
CX=261.442628
CY=253.231895
GLOBAL_TRANS=True
EXT=jpg
###################################
DECA_DIR=$WORK_DIR/code/submodules/DECA
FACE_TRACKING_DIR=$WORK_DIR/code/submodules/face_tracking
###################################

export CUDA_VISIBLE_DEVICES=$DEVICE
source $HOME/conda/etc/profile.d/conda.sh
conda activate $ENV_NAME

TARGET_DIR_PARENTS=($(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | sort))

# 디렉토리 안의 모든 하위 디렉토리를 찾고 루프를 돌립니다.
for ((i=START_IDX; i<=END_IDX && i<${#TARGET_DIR_PARENTS[@]}; i++)); do
    TARGET_DIR="${TARGET_DIR_PARENTS[i]}"
    echo "[INFO] Processing $TARGET_DIR"

    # flame_params.json이 이미 존재하면 스킵
    if [ -f "$TARGET_DIR/flame_params.json" ]; then
        echo "[INFO] flame_params.json exists. Skipping $TARGET_DIR"
        continue
    fi

    echo "[INFO] DECA FLAME parameter estimation" 
    cd $DECA_DIR
    python demos/demo_reconstruct.py -i $TARGET_DIR/image --savefolder $TARGET_DIR/deca --saveCode True --saveVis False --sample_step 1  --render_orig False

    echo "[INFO] face alignment landmark detector" 
    cd $FACE_TRACKING_DIR
    python keypoint_detector.py --path $TARGET_DIR

    echo "[INFO] iris segmentation with fdlite"
    cd $FACE_TRACKING_DIR
    python iris.py --path $TARGET_DIR

    echo "[INFO] DECA Optimization"
    cd $DECA_DIR
    python optimize.py --path $TARGET_DIR \
                       --cx $CX --cy $CY --fx $FX --fy $FY \
                       --size $RESIZE \
                       --global_trans $GLOBAL_TRANS \
                       --save_name 'flame_params' \
                       --ext $EXT
done