#!/usr/bin/env bash
MODE_TRANSFER=${1:-"MONO_MONO"}
META_LEARN_ALG=${2:-"maml"}
GPU_NUM=${3:-1}

multi_meta_ssd asymsearch  --use_meta_learn --meta_learn_alg $META_LEARN_ALG --mode_transfer $MODE_TRANSFER --gpu_order $GPU_NUM --do_pre_finetune
