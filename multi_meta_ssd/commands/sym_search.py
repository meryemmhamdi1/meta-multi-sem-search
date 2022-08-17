import os
import pickle
import torch
import numpy as np
import json
import importlib
from tqdm import tqdm
from multi_meta_ssd.processors.downstream import utils_lareqa
from multi_meta_ssd.commands.get_arguments import get_train_params, get_path_options, get_adam_optim_params, get_base_model_options, get_language_options, get_meta_task_options
from multi_meta_ssd.models.downstream.dual_encoders.bert import BertForRetrieval
from multi_meta_ssd.models.downstream.dual_encoders.sent_trans import SBERTForRetrieval
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr
from transformers import (
    AdamW, BertConfig, BertTokenizer, WEIGHTS_NAME, XLMRobertaTokenizer,
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup)

import random as rnd
try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

import pickle


def sym_eval(subparser):
    parser = subparser.add_parser("sym_eval", help="Split Lareqa using cross-validation")
    get_train_params(parser)
    get_path_options(parser)
    get_adam_optim_params(parser)
    get_base_model_options(parser)
    get_language_options(parser)
    get_meta_task_options(parser)
    parser.set_defaults(func=evaluation_stsb_data)

def zero_shot_evaluation(sentences_pair, scores_gs, tokenizer, base_model,  meta_learn_split_config, args, cross_val):
    
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
   
    # sentences_pairs["sentences1"], sentences_pairs["sentences2"]

    computed_scores = []
    for i in range(len(sentences_pair["sentences1"])):
        # Get the encoding of the question using the base model
        sentences1 = sentences_pair["sentences1"]
        sentences2 = sentences_pair["sentences2"]

        base_model = base_model.to(args.device)
        features_sent1 = tokenizer.encode_plus(sentences1[i],
                                               max_length=meta_learn_split_config["max_answer_length"],
                                               pad_to_max_length=True,
                                               return_token_type_ids=True)

        q_features_sent1 = {"input_ids": torch.unsqueeze(torch.tensor(features_sent1["input_ids"], dtype=torch.long), dim=0),
                          "attention_mask": torch.unsqueeze(torch.tensor(features_sent1["attention_mask"], dtype=torch.long), dim=0),
                          "token_type_ids": torch.unsqueeze(torch.tensor(features_sent1["token_type_ids"], dtype=torch.long), dim=0)}

        q_features_sent1 = {k:v.to(args.device) for k, v in q_features_sent1.items()}

        ####

        features_sent2 = tokenizer.encode_plus(sentences2[i],
                                               max_length=meta_learn_split_config["max_answer_length"],
                                               pad_to_max_length=True,
                                               return_token_type_ids=True)

        q_features_sent2 = {"input_ids": torch.unsqueeze(torch.tensor(features_sent2["input_ids"], dtype=torch.long), dim=0),
                          "attention_mask": torch.unsqueeze(torch.tensor(features_sent2["attention_mask"], dtype=torch.long), dim=0),
                          "token_type_ids": torch.unsqueeze(torch.tensor(features_sent2["token_type_ids"], dtype=torch.long), dim=0)}

        q_features_sent2 = {k:v.to(args.device) for k, v in q_features_sent2.items()}


        with torch.no_grad():
            sentence1_encoding = base_model(q_input_ids=q_features_sent1["input_ids"],
                                            q_attention_mask=q_features_sent1["attention_mask"],
                                            q_token_type_ids=q_features_sent1["token_type_ids"],
                                            inference=True)

            sentence2_encoding = base_model(q_input_ids=q_features_sent2["input_ids"],
                                            q_attention_mask=q_features_sent2["attention_mask"],
                                            q_token_type_ids=q_features_sent2["token_type_ids"],
                                            inference=True)

        scores = sentence1_encoding.dot(sentence2_encoding.T)
    
        computed_scores.append(scores[0][0])
    # print(scores[i][0] for i in range(len(scores)))
    
    scores_normalized = (scores_gs - np.min(scores_gs)) / (np.max(scores_gs) - np.min(scores_gs))
    # print("len(scores_normalized):", len(scores_normalized), " len(computed_scores):", len(computed_scores))
    
    correlation, p_value = pearsonr(computed_scores, scores_normalized)
    # correlation = np.corrcoef(scores_normalized, computed_scores)
    # if i == 5:
    #     data = scores_gs[0:5]
    #     print("data:", data)
    #     print("np.min(data):", np.min(data), " np.max(data):", np.max(data))
    #     scores_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    #     print("data:", data, " scores_normalized:", scores_normalized)
    #     correlation, p_value = pearsonr(scores_normalized, scores)
    #     print("correlation:", correlation)
    #     exit(0)
        

    return correlation

def read_txt_file(lang_pair):
    root_path = "/sensei-fs/users/mhamdi/Datasets/STS2017/"
    if lang_pair == "ar-ar":
        track_id = "1"
    elif lang_pair == "ar-en":
        track_id = "2"
    elif lang_pair == "es-es":
        track_id = "3"
    elif lang_pair == "es-en":
        track_id = "4a"
    elif lang_pair == "en-en":
        track_id = "5"
    elif lang_pair == "tr-en":
        track_id = "6"
    
    print(lang_pair)
    with open(os.path.join(root_path, "STS2017.eval.v1.1", "STS.input.track"+track_id+"."+lang_pair+".txt"), "r") as file:
        data = file.read().splitlines()

    with open(os.path.join(root_path, "STS2017.gs", "STS.gs.track"+track_id+"."+lang_pair+".txt"), "r") as file:
        scores = file.read().splitlines()

    scores = [float(score) for score in scores]

    sentences1 = []
    sentences2 = []

    for line in data:
        sent1, sent2 = line.split("\t")
        sentences1.append(sent1)
        sentences2.append(sent2)
    
    return sentences1, sentences2, scores

def evaluation_stsb_data(args):
    base_model = "sbert-retrieval"
    upstream_model = "maml"
    use_base = False

    if upstream_model == "maml_align": #0.8551330950410478
        meta_task_modes = ["MONO_BIL_MULTI"]
    else:
        meta_task_modes = ["MONO_BIL"]#["MONO_MONO", "MONO_BIL", "BIL_MULTI", "MIXT", "MONO_MULTI", "TRANS"]

    train_valid_mode = "valid"
    cross_vals = list(range(1))
    random = True
    prefinetune = False
    if random:
        random_string = "random"
    else:
        random_string = "paraphrase-multilingual-mpnet-base-v2"

    LANG_PAIRS = ["ar-ar", "ar-en", "es-es", "es-en", "en-en"]#, "tr-en"]
    
    sentences_pairs = {}
    for lang_pair in LANG_PAIRS:
        sentences1_list, sentences2_list, scores = read_txt_file(lang_pair)
        sentences_pairs[lang_pair] = {"sentences1": sentences1_list, "sentences2": sentences2_list}

    root_results_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/"+base_model
    for cross_val in cross_vals:
        middle_path = ""
        if prefinetune:
            middle_path += "PreFineTune/"

        middle_path += "TripletLoss/"

        if cross_val != -1:
            middle_path += "CrossVal_"+str(cross_val)+"/"

        if random:
            middle_path += "random/"

        middle_path += "checkpoints/"
        os.environ["CUDA_VISIBLE_DEVICES"]="7"
        args.device = torch.device("cuda")
        NUM_EPOCHS = 1
        all_multilingual_evaluations = {meta_task_mode: {} for meta_task_mode in meta_task_modes}
        for meta_task_mode in meta_task_modes:
            test_mode_evaluation_epochs = {epoch: {lang_pair: {} for lang_pair in LANG_PAIRS} for epoch in range(NUM_EPOCHS)}
            for epoch in range(NUM_EPOCHS):
                model_load_file = os.path.join(root_results_path, upstream_model, meta_task_mode, middle_path, "pytorch_model_"+train_valid_mode+str(epoch)+".bin") 
                if not os.path.exists(model_load_file):
                    model_load_file = os.path.join(root_results_path, upstream_model, meta_task_mode, middle_path, "pytorch_model_"+str(epoch)+".bin") 

                model_name_or_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                cache_dir = ""
                config_name = None
                tokenizer_name = None

                config = AutoConfig.from_pretrained(config_name if config_name else model_name_or_path,
                                                    cache_dir=cache_dir if cache_dir else None)

                base_model = SBERTForRetrieval(config=config,
                                            trans_model_name=model_name_or_path)

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                                        #   do_lower_case=True,
                                                        cache_dir=cache_dir if cache_dir else None)

                base_model.to(args.device)

                if not use_base:
                    base_model.load_state_dict(torch.load(model_load_file, map_location=args.device), strict=False)

                meta_learn_split_config = {"train": {"n_tasks_total": args.n_train_tasks,
                                                    "n_tasks_batch": args.n_train_tasks_batch,
                                                    "n_up_steps": args.n_up_train_steps,
                                                    "alpha_lr": args.alpha_lr_train,
                                                    "beta_lr": args.beta_lr_train,
                                                    "lang_pairs": args.train_lang_pairs},

                                            "valid": {"n_tasks_total": args.n_valid_tasks,
                                                    "n_tasks_batch": args.n_valid_tasks_batch,
                                                    "n_up_steps": args.n_up_valid_steps,
                                                    "alpha_lr": args.alpha_lr_valid,
                                                    "lang_pairs": args.valid_lang_pairs},

                                            "test": {"n_tasks_total": args.n_test_tasks,
                                                    "n_tasks_batch": args.n_test_tasks_batch,
                                                    "n_up_steps": args.n_up_test_steps,
                                                    "alpha_lr": args.alpha_lr_test,
                                                    "lang_pairs": args.test_lang_pairs},
                                            "max_seq_length": args.max_seq_length,
                                            "max_query_length": args.max_query_length,
                                            "max_answer_length": args.max_answer_length,
                                            "n_neg_eg": args.n_neg_eg,
                                            "neg_sampling_approach": args.neg_sampling_approach,
                                            "mode": args.mode_qry
                                        }

                for lang_pair in LANG_PAIRS:
                    ad_hoc_eval = zero_shot_evaluation(sentences_pairs[lang_pair], scores, tokenizer, base_model, meta_learn_split_config, args, cross_val)
                    print("lang_pair:", lang_pair, " ad_hoc_eval:", ad_hoc_eval)
                    test_mode_evaluation_epochs[epoch][lang_pair] = ad_hoc_eval

            all_multilingual_evaluations[meta_task_mode] = test_mode_evaluation_epochs

        if use_base:
            name_model = "BASE"
        else:
            name_model = upstream_model
        save_path = root_results_path+"/summary_SEMEVAL-"+upstream_model+"-"+train_valid_mode+"-"+random_string+"-cross_val_"+str(cross_val)
        with open(save_path+"ad_hoc_perf.pickle", "wb") as file:
            pickle.dump(all_multilingual_evaluations,file)