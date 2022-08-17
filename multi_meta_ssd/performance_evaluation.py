import os
import pickle
import torch
import numpy as np
import json
from tqdm import tqdm
from multi_meta_ssd.processors.downstream import utils_lareqa
from transformers import BertConfig, BertTokenizer, XLMRobertaTokenizer
from multi_meta_ssd.models.downstream.dual_encoders.bert import BertForRetrieval


root_results_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/bert-retrieval/"
def read_results(mode="finetune", meta_task_mode= "BIL_MULTI"):

    for split in ["train", "valid", "test"]:
        with open(os.path.join(root_results_path, mode, meta_task_mode, "PreFineTune/TripletLoss/runs/", split+"_map_qry_all_total.pickle"), "rb") as file:
            results = pickle.load(file)

        print(mode, meta_task_mode, results)

for mode in ["finetune", "maml"]:
    for meta_task_mode in ["BIL_MULTI", "MIXT", "MONO_BIL", "MONO_MONO", "MONO_MULTI", "TRANS"]:
        read_results(mode=mode, meta_task_mode= meta_task_mode)