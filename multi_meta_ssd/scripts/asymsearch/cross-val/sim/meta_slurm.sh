#!/usr/bin/env bash

#SBATCH --partition=isi
#SBATCH --mem=100g
#SBATCH --time=7-24:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p100:1
#SBATCH --output=R-%x.out.%j
#SBATCH --error=R-%x.err.%j
#SBATCH --export=NONE

hostname -f
echo "Hello World : args:  ${@}"


# MODE_TRANSFER=${1:-"MONO_MONO"}
# GPU_NUM=${2:-"0"}cd
# MODEL_TYPE=${3:-"sbert-retrieval"}
# MODEL_NAME_OR_PATH=${4:-"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
# META_LEARN_ALG=${5:-"maml"}
# CROSS_VAL_SPLIT=${6:-"0"}
# NEG_SAMPLING_APPROACH=${7:-"random"}

# for split in "0" "1" "2" "3" "4"
# do
#     sh multi_meta_ssd/scripts/asymsearch/meta.sh $MODE_TRANSFER $MODEL_TYPE $MODEL_NAME_OR_PATH $META_LEARN_ALG $GPU_NUM \
#          1 0 0 0 1 $split $NEG_SAMPLING_APPROACH
# done
