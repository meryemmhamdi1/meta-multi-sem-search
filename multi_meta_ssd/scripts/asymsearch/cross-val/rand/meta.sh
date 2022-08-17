#!/usr/bin/env bash
MODE_TRANSFER=${1:-"MONO_MONO"}
GPU_NUM=${2:-"0"}
MODEL_TYPE=${3:-"sbert-retrieval"}
MODEL_NAME_OR_PATH=${4:-"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
META_LEARN_ALG=${5:-"maml"}
CROSS_VAL_SPLIT=${6:-"0"}
NEG_SAMPLING_APPROACH=${7:-"random"}


# USE_SIM=${3:-0}
# PRE_FINE_TUNE=${4:-0} is not relevant needed for SBERT especially 
# DO_EVALUATION=${5:-0} is not needed
# USE_RANDOM=${6:-0}
# USE_CROSS_VAL=${7:-0}

for split in "0" "1" "2" "3" "4"
do
    sh multi_meta_ssd/scripts/asymsearch/meta.sh $MODE_TRANSFER $MODEL_TYPE $MODEL_NAME_OR_PATH $META_LEARN_ALG $GPU_NUM \
    1 0 0 1 1 $split $NEG_SAMPLING_APPROACH
done
