import os, pickle
import numpy as np
from tqdm import tqdm
import torch
import json
from sentence_transformers import SentenceTransformer
from multi_meta_ssd.processors.downstream import utils_lareqa
from multi_meta_ssd.commands.get_arguments import get_train_params, get_path_options, get_adam_optim_params, get_base_model_options, get_language_options, get_meta_task_options

from transformers import (AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup)

from multi_meta_ssd.models.downstream.dual_encoders.sent_trans import SBERTForRetrieval
from laserembeddings import Laser
from scipy.stats import pearsonr
import pandas as pd
import csv

laser = Laser()

data_root = "/project/jonmay_231/meryem/Datasets/stsb-multi-mt/data"
LANGUAGES = ["ar-ar", "ar-en", "es-es", "es-en", "en-en", "tr-en"]
split_names = ["train", "dev", "test"]
def evaluate_stsb_mt_options(subparser):
    parser = subparser.add_parser("evaluate_stsb_mt_options", help="MT Performance Evaluation")
    get_train_params(parser)
    get_path_options(parser)
    get_adam_optim_params(parser)
    get_base_model_options(parser)
    get_language_options(parser)
    get_meta_task_options(parser)
    parser.set_defaults(func=perf_mt_stsb_res_data)

def zero_shot_evaluation(sentences_pair, lang, tokenizer, base_model,  meta_learn_split_config, args):
    computed_scores = []
    for i in range(len(sentences_pair[lang]["sentences1"])):
        # Get the encoding of the question using the base model
        sentences1 = sentences_pair[lang]["sentences1"]
        sentences2 = sentences_pair[lang]["sentences2"]
        scores_gs = sentences_pair[lang]["scores"]

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
    
    scores_normalized = (computed_scores - np.min(scores_gs)) / (np.max(scores_gs) - np.min(scores_gs))
    
    min_ = min(len(scores_gs), len(scores_normalized))
    correlation, _ = pearsonr(scores_gs[:min_], scores_normalized[:min_])
   
    return correlation

def eval_base_model(args):
    def multilingual_eval():
        """ English -> All Languages """


    def bilingual_eval():
        """ Language X -> Language Y """

    def monolingual_eval():
        """ Language X -> Language X """


def translate_test(args):
    def monolingual_eval():
        """ Translate the non-english side(s) to english and check the correlation there"""


def read_csv_file(lang, split):
    with open(os.path.join(data_root, "stsb-"+lang+"-"+split+".csv")) as tsv_file:
        reader = csv.reader(tsv_file, delimiter=",")

        sentences1 = []
        sentences2 = []
        scores = []

        for _, line in enumerate(reader):
            sent1, sent2, score = line
            sentences1.append(sent1)
            sentences2.append(sent2)
            scores.append(float(score))
    
    return sentences1, sentences2, scores

def read_txt_file(lang_pair, root_path):
    lang_track_dict = {"ar-ar": "1", "ar-en": "2", "es-es": "3", "es-en": "4a", "en-en": "5", "tr-en": "6"}

    with open(os.path.join(root_path, "STS2017.eval.v1.1", "STS.input.track"+lang_track_dict[lang_pair]+"."+lang_pair+".txt"), "r") as file:
        data = file.read().splitlines()

    sentences1 = []
    sentences2 = []

    for line in data:
        sent1, sent2 = line.split("\t")
        sentences1.append(sent1)
        sentences2.append(sent2)

    with open(os.path.join(root_path, "STS2017.gs", "STS.gs.track"+lang_track_dict[lang_pair]+"."+lang_pair+".txt"), "r") as file:
        scores = file.read().splitlines()

    scores = [float(score) for score in scores]
    
    return sentences1, sentences2, scores

def perf_mt_stsb_res_data(args):
    sentences_pairs = {}
    split = "test"
    config_name = None
    tokenizer_name = None
    model_name_or_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    cache_dir = None

    args.device = torch.device("cuda")

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

    config = AutoConfig.from_pretrained(config_name if config_name else model_name_or_path,
                                            cache_dir=cache_dir if cache_dir else None)

    base_model = SBERTForRetrieval(config=config,
                                    trans_model_name=model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                                cache_dir=cache_dir if cache_dir else None)

    root_path = "/project/jonmay_231/meryem/Datasets/STS2017/"

    for lang in LANGUAGES:
        sentences1_list, sentences2_list, scores = read_txt_file(lang, root_path)

        ### Reading sentences1_list 
        ## Reading sentences2_list
        if lang in ["ar-ar", "ar-en", "es-es", "es-en", "tr-en"]:
            with open(root_path+"Translations/GoogleTrans/"+lang+"_sentences1.txt", "r") as file:
                sentences1_list = file.read().splitlines()

        if lang in ["ar-ar", "es-es"]:
            with open(root_path+"Translations/GoogleTrans/"+lang+"_sentences2.txt", "r") as file:
                sentences2_list = file.read().splitlines()

        print("len(sentences1_list):", len(sentences1_list), " sentences1_list[0]:", sentences1_list[0], " sentences2_list[0]:", sentences2_list[0], " scores[0]:", scores[0])
        sentences_pairs[lang] = {"sentences1": sentences1_list, "sentences2": sentences2_list, "scores": scores}

        ad_hoc_eval = zero_shot_evaluation(sentences_pairs, lang, tokenizer, base_model, meta_learn_split_config, args)

        print("lang:", lang, " ad_hoc_eval:", ad_hoc_eval)

    # for split in split_names:
    #     for lang in LANGUAGES:
    #         data = pd.read_csv(os.path.join(data_root, "stsb-"+lang+"-"+split+".csv"))
    