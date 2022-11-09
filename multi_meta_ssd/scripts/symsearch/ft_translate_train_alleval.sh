#!/usr/bin/env bash
MODE_TRANSFER=${1:-"MONO_MONO"}
MODEL_TYPE=${2:-"sbert-retrieval"}
MODEL_NAME_OR_PATH=${3:-"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
GPU_NUM=${4:-1}


multi_meta_ssd symsearch_train --model_type $MODEL_TYPE \
                               --model_name_or_path $MODEL_NAME_OR_PATH \
                               --mode_transfer $MODE_TRANSFER \
                               --gpu_order $GPU_NUM \
                               --translate_train \
                               --translate_train_langs "en,ar,es,tr"