#!/usr/bin/env bash
MODE_TRANSFER=${1:-"MONO_MONO"}
GPU_NUM=${2:-"0"}
MODEL_TYPE=${3:-"sbert-retrieval"}
MODEL_NAME_OR_PATH=${4:-"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
# MODEL_TYPE=${3:-"bert-retrieval"}
# MODEL_NAME_OR_PATH=${4:-"bert-base-multilingual-cased"}
META_LEARN_ALG=${5:-"maml"}
NEG_SAMPLING_APPROACH=${6:-"random"}
SEED=${7:-109883242} # 209883242 309883242


# USE_SIM=${3:-0}
# PRE_FINE_TUNE=${4:-0} is not relevant needed for SBERT especially 
# DO_EVALUATION=${5:-0} is not needed
# USE_RANDOM=${6:-0}
# USE_CROSS_VAL=${7:-0}

for MODE_TRANSFER in "MONO_MONO" "MONO_BIL" #"MONO_MULTI" "BIL_MULTI" "MIXT" "TRANS"
do
    for split in "0" "1" "2" "3" "4"
    do
        for SEED in 109883242 209883242 309883242
        do
            # SBERT
            sh multi_meta_ssd/scripts/asymsearch/meta.sh $MODE_TRANSFER \
                                                        $MODEL_TYPE \
                                                        $MODEL_NAME_OR_PATH \
                                                        $META_LEARN_ALG \
                                                        $GPU_NUM \
                                                        1 \
                                                        0 \
                                                        0 \
                                                        1 \
                                                        1 \
                                                        $split \
                                                        $NEG_SAMPLING_APPROACH \
                                                        $SEED

        done


    #     sh multi_meta_ssd/scripts/asymsearch/meta.sh $MODE_TRANSFER \
    #                                                  $MODEL_TYPE \
    #                                                  $MODEL_NAME_OR_PATH \
    #                                                  $META_LEARN_ALG \
    #                                                  $GPU_NUM \
    #                                                  1 \
    #                                                  1 \
    #                                                  0 \
    #                                                  1 \
    #                                                  1 \
    #                                                  $split \
    #                                                  $NEG_SAMPLING_APPROACH \
    #                                                  $SEED
    done
done