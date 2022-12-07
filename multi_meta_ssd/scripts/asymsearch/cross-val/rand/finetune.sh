#!/usr/bin/env bash
MODE_TRANSFER=${1:-"MONO_MONO"}
GPU_NUM=${2:-"0"}
MODEL_TYPE=${3:-"sbert-retrieval"}
MODEL_NAME_OR_PATH=${4:-"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
# MODEL_TYPE=${3:-"bert-retrieval"}
# MODEL_NAME_OR_PATH=${4:-"bert-base-multilingual-cased"}
CROSS_VAL_SPLIT=${5:-"0"}
NEG_SAMPLING_APPROACH=${6:-"random"}

for MODE_TRANSFER in "MONO_MONO" "MONO_BIL" "MONO_MULTI" "BIL_MULTI" "MIXT" "TRANS"
do
    for SEED in 109883242 209883242 309883242
    do
        for SPLIT in "0" "1" "2" "3" "4"
        do
            # sh multi_meta_ssd/scripts/asymsearch/ft.sh $MODE_TRANSFER $MODEL_TYPE $MODEL_NAME_OR_PATH $GPU_NUM 1 1 0 1 1 $split $NEG_SAMPLING_APPROACH
            sh multi_meta_ssd/scripts/asymsearch/ft.sh $MODE_TRANSFER \
                                                    $MODEL_TYPE \
                                                    $MODEL_NAME_OR_PATH \
                                                    $GPU_NUM \
                                                    1 \
                                                    0 \
                                                    0 \
                                                    1 \
                                                    1 \
                                                    $SPLIT \
                                                    $NEG_SAMPLING_APPROACH \
                                                    $SEED
        done
    done
done
