#!/usr/bin/env bash
MODE_TRANSFER=${1:-"MONO_MONO"}
MODEL_TYPE=${2:-"sbert-retrieval"}
MODEL_NAME_OR_PATH=${3:-"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
META_LEARN_ALG=${4:-"maml"}
GPU_NUM=${5:-1}
USE_SIM=${6:-0}
PRE_FINE_TUNE=${7:-0}
DO_EVALUATION=${8:-0}
USE_RANDOM=${9:-0}
USE_CROSS_VAL=${10:-0}
CROSS_VAL_SPLIT=${11:-"0"}
NEG_SAMPLING_APPROACH=${12:-"random"}

OPTIONS=""

if [ "$USE_SIM" = 1 ]; then
    OPTIONS=" --use_sim_embedder "
fi

if [ "$PRE_FINE_TUNE" = 1 ]; then
    OPTIONS=$OPTIONS" --do_pre_finetune "
fi

if [ "$DO_EVALUATION" = 1 ]; then
    OPTIONS=$OPTIONS" --do_evaluate "
fi

if [ "$USE_RANDOM" = 1 ]; then
    OPTIONS=$OPTIONS" --mode_qry=random "
fi

if [ "$USE_CROSS_VAL" = 1 ]; then
    OPTIONS=$OPTIONS" --use_cross_val --cross_val_split "$CROSS_VAL_SPLIT
fi

multi_meta_ssd asymsearch  --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME_OR_PATH --use_meta_learn --meta_learn_alg $META_LEARN_ALG \
                           --mode_transfer $MODE_TRANSFER --gpu_order $GPU_NUM $OPTIONS --use_triplet_loss --neg_sampling_approach $NEG_SAMPLING_APPROACH
