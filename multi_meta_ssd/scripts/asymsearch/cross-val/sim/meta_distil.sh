#!/usr/bin/env bash
MODE_TRANSFER=${1:-"MONO_BIL_MULTI"}
GPU_NUM=${2:-"7"}
MODEL_TYPE=${3:-"sbert-retrieval"}
MODEL_NAME_OR_PATH=${4:-"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
META_LEARN_ALG=${5:-"maml_align"}
CROSS_VAL_SPLIT=${6:-"0"}
NEG_SAMPLING_APPROACH=${7:-"random"}

echo "MODE_TRANSFER "$MODE_TRANSFER

for split in "0" "1" "2" "3" "4"
do
    sh multi_meta_ssd/scripts/asymsearch/meta_distil.sh $MODE_TRANSFER $MODEL_TYPE $MODEL_NAME_OR_PATH $META_LEARN_ALG $GPU_NUM \
         1 0 0 0 1 $split $NEG_SAMPLING_APPROACH
done
