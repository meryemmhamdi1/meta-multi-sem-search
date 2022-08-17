#!/usr/bin/env bash
mode=${1:-"MONO_MONO"}

for mode in "MONO_MONO" "MONO_BIL"
do
    sh multi_meta_ssd/scripts/asymsearch/meta.sh $mode "maml" 1 1 0 0 1 0 "0"
done


sh multi_meta_ssd/scripts/asymsearch/meta.sh "MONO_BIL" "maml" 2 1 0 0 1 1 "0" && sh multi_meta_ssd/scripts/asymsearch/meta.sh "MONO_BIL" "maml" 2 1 0 0 1 1 "1" && sh multi_meta_ssd/scripts/asymsearch/meta.sh "MONO_BIL" "maml" 2 1 0 0 1 1 "2" && sh multi_meta_ssd/scripts/asymsearch/meta.sh "MONO_BIL" "maml" 2 1 0 0 1 1 "3" && sh multi_meta_ssd/scripts/asymsearch/meta.sh "MONO_BIL" "maml" 2 1 0 0 1 1 "4" && sh multi_meta_ssd/scripts/asymsearch/meta.sh "MONO_BIL" "maml" 2 1 0 0 1 1 "5"