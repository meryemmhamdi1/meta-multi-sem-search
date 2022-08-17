#!/usr/bin/env bash
MODE_TRANSFER=${1:-"BIL_MULTI"}
GPU_NUM=${2:-"3"}
MODEL_TYPE=${3:-"sbert-retrieval"}
MODEL_NAME_OR_PATH=${4:-"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
CROSS_VAL_SPLIT=${5:-"0"}
NEG_SAMPLING_APPROACH=${6:-"random"}

for split in "0"
do
    sh multi_meta_ssd/scripts/asymsearch/ft.sh $MODE_TRANSFER $MODEL_TYPE $MODEL_NAME_OR_PATH $GPU_NUM 1 0 0 1 1 $split $NEG_SAMPLING_APPROACH
done

