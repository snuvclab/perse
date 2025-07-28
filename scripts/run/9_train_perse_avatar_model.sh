#!/bin/zsh
set -e

source $HOME/conda/etc/profile.d/conda.sh
conda activate

############################
ENV_NAME=perse_avatar
DEVICE=1,2,3,4,5,6,7
WORK_DIR=$HOME/GitHub/perse-dev
BASE_CONF=confs/base.conf
EXP_CONF=confs/stage1_supp_video.conf
############################

conda activate $ENV_NAME
export CUDA_VISIBLE_DEVICES=$DEVICE
NPROC=$(echo "$DEVICE" | tr ',' '\n' | wc -l)
cd $WORK_DIR/code/avatar_model

ntfy done python -m torch.distributed.launch \
    --master_port 29506 \
    --nproc_per_node=$NPROC \
    scripts/exp_runner.py \
    --base_conf $BASE_CONF \
    --exp_conf $EXP_CONF