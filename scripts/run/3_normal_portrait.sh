#!/bin/bash
#############################################
DEVICE=7
ENV_NAME=portrait-editor
DATASET_NAME=portrait-release
ATTRIBUTE=hair
#############################################

source ../common.sh
activate_env

EXT=png
INPUT=$DATASET_PATH/portrait-edit-image/$ATTRIBUTE/   
cd $SUBMODULE_PATH/sapiens/seg/scripts/demo/local/

ntfy done bash seg.sh $INPUT/image \
                      $INPUT/segment \
                      $EXT \
                      $DEVICE \
                      $SAPIENS_CHECKPOINT_PATH \
                      $ENV_NAME

ntfy done bash normal.sh $INPUT/image \
                         $INPUT/segment \
                         $INPUT/normal \
                         $EXT \
                         $DEVICE \
                         $SAPIENS_CHECKPOINT_PATH \
                         $ENV_NAME
rm -rf $INPUT/segment
rm $INPUT/*.txt