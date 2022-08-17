#!/usr/bin/env bash
MODE_TRANSFER=${1:-"MONO_MONO"}
META_LEARN_ALG=${2:-"maml"}
GPU_NUM=${3:-1}
USE_SIM=${4:-0}
PRE_FINE_TUNE=${5:-0}
DO_EVALUATION=${6:-0}
USE_RANDOM=${7:-0}
USE_CROSS_VAL=${8:-0}
CROSS_VAL_SPLIT=${9:-"0"}
NEG_SAMPLING_APPROACH=${10:-"random"}

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

multi_meta_ssd asymsearch  --use_meta_learn --meta_learn_alg $META_LEARN_ALG --mode_transfer $MODE_TRANSFER --gpu_order $GPU_NUM $OPTIONS \
                           --use_triplet_loss --neg_sampling_approach $NEG_SAMPLING_APPROACH --
