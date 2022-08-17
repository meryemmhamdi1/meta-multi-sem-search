#!/usr/bin/env bash
MODE_TRANSFER=${1:-"MONO_MONO"}
GPU_NUM=${2:-"6"}
USE_SIM=${3:-0}
PRE_FINE_TUNE=${4:-0}
DO_EVALUATION=${5:-0}
USE_RANDOM=${6:-0}
USE_CROSS_VAL=${7:-0}
CROSS_VAL_SPLIT=${8:-"0"}
NEG_SAMPLING_APPROACH=${9:-"random"}

split="0"
for TRANSFER_OPTION in "MONO_MONO" "MONO_BIL" "MONO_MULTI" "BIL_MULTI" "MIXT" "TRANS"
do
    sh multi_meta_ssd/scripts/asymsearch/ft.sh $TRANSFER_OPTION $GPU_NUM 1 0 0 0 1 $split $NEG_SAMPLING_APPROACH
done