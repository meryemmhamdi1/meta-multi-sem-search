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

LANGUAGES = ["ar", "de", "el", "hi", "ru", "th", "tr"]

def perf_eval_options(subparser):
    parser = subparser.add_parser("perf_eval_options", help="Split Lareqa using cross-validation")
    get_train_params(parser)
    get_path_options(parser)
    get_adam_optim_params(parser)
    get_base_model_options(parser)
    get_language_options(parser)
    get_meta_task_options(parser)
    parser.set_defaults(func=perf_eval_analyze_cross_val)

def perf_eval_mono_cross_val(args):
    conf_matrix = {lang: {lang: [] for lang in LANGUAGES} for lang in LANGUAGES}
    conf_matrix_mean = {lang: {lang: 0.0 for lang in LANGUAGES} for lang in LANGUAGES}
    root_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval-base-sbert-cross_val_"
    # root_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval-maml-mono_bilONLY_train_sim-cross_val_"
    for i in range(5):
        load_file_name = ""
        if i in [0, 1, 4]:
            load_file_name += "-random"
        with open(root_path + str(i) + load_file_name+"multi_mono_cross.pickle", "rb") as file:
            data = pickle.load(file)
        
        if i in [0, 1]:
            mode_transfer = "ALL"
        else:
            mode_transfer = "MONO_MONO"

        # mode_transfer = "MONO_BIL"

        for lang1 in LANGUAGES:
            for lang2 in LANGUAGES:
                conf_matrix[lang1][lang2].append(data["mono_cross"][mode_transfer][0][lang1][lang2][lang1])

    for lang1 in LANGUAGES:
        for lang2 in LANGUAGES:
            conf_matrix_mean[lang1][lang2] = round(np.mean(conf_matrix[lang1][lang2])*100, 2)
            print(round(np.mean(conf_matrix[lang1][lang2])*100, 2))#, round(np.std(conf_matrix[lang1][lang2])*100, 2))
    
    print("conf_matrix_mean:", conf_matrix_mean)

    monolingual_mean = []
    crosslingual_mean = []
    for lang1 in LANGUAGES:
        for lang2 in LANGUAGES:
            conf_matrix_mean[lang1][lang2] = round(np.mean(conf_matrix[lang1][lang2])*100, 2)
            if lang1 == lang2:
                monolingual_mean.append(conf_matrix_mean[lang1][lang2])
                # print(round(np.mean(conf_matrix[lang1][lang2])*100, 2))#, round(np.std(conf_matrix[lang1][lang2])*100, 2))
            else:
                crosslingual_mean.append(conf_matrix_mean[lang1][lang2])
    
    print("conf_matrix_mean:", conf_matrix_mean)
    print("monolingual_mean:", np.mean(monolingual_mean))
    print("crosslingual_mean:", np.mean(crosslingual_mean))

def perf_eval_analyze_cross_val(args):
    mean_cross_vals = []
    std_cross_vals = []
    mean_cross_vals_lang = {lang: [] for lang in LANGUAGES}
    root_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval-base-sbert-cross_val_"
    # root_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval-maml-mono_bilONLY_train_sim-cross_val_"
    for i in range(5):
        load_file_name = ""
        if i in [0, 1, 4]:
            load_file_name += "-random"
        with open(root_path + str(i) + load_file_name+"multi_mono_cross.pickle", "rb") as file:
            data = pickle.load(file)
        
        if i in [0, 1]:
            mode_transfer = "ALL"
        else:
            mode_transfer = "MONO_MONO"


        print(np.mean([v for k, v in data["multi"][mode_transfer][0].items()]))
        all_lang_for_split = data["multi"][mode_transfer][0]

        mean_for_split = np.mean([v for k, v in all_lang_for_split.items()])
        std_for_split = np.std([v for k, v in all_lang_for_split.items()])

        mean_cross_vals.append(mean_for_split)
        std_cross_vals.append(std_for_split)
        for lang in LANGUAGES:
            mean_cross_vals_lang[lang].append(all_lang_for_split[lang])

    print("BASE MEAN:", round(np.mean(mean_cross_vals)*100, 2), " STD: ", round(np.mean(std_cross_vals)*100,2))
    mean_all_langs = {lang: round(np.mean(mean_cross_vals_lang[lang])*100, 2) for lang in LANGUAGES}
    std_all_langs = {lang: round(np.std(mean_cross_vals_lang[lang])*100, 2) for lang in LANGUAGES}
    print(mean_all_langs, std_all_langs)
    print("MEAN LANGUAGES:", np.mean([v for k, v in mean_all_langs.items()]))

    # print("MULTIPLE MODES OF TRANSFER")
    # mean_cross_vals_modes = {mode_transfer: [] for mode_transfer in modes_transfer}
    #  mean_cross_vals_lang = {mode_transfer:  {lang: [] for lang in LANGUAGES} for mode_transfer in modes_transfer}
    # root_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval-maml-new_valid_sim-cross_val_"
    # for i in range(5):
    #     with open(root_path + str(i) + "multi_mono_cross.pickle", "rb") as file:
    #         data = pickle.load(file)
    #     for mode_transfer in modes_transfer:
    #         mean_cross_vals_modes[mode_transfer].append(np.mean([v for k, v in data["multi"][mode_transfer][0].items()]))

    # for mode_transfer in modes_transfer:
    #     print(mode_transfer, round(np.mean(mean_cross_vals_modes[mode_transfer])*100, 2))


    # print("MONO_BIL PERFORMANCE")
    # mean_cross_vals = []
    # mean_cross_vals_lang = {lang: [] for lang in LANGUAGES}
    # root_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval-maml-mono_bilONLY_train_sim-cross_val_"
    # for i in range(5):
    #     with open(root_path + str(i) + "multi_mono_cross.pickle", "rb") as file:
    #         data = pickle.load(file)

    #     all_lang_for_split = data["multi"]["MONO_BIL"][0]
    #     print("MONO_BIL MEAN:", i, np.mean([v for k, v in all_lang_for_split.items()]))
    #     mean_cross_vals.append(np.mean([v for k, v in all_lang_for_split.items()]))

    #     for lang in LANGUAGES:
    #         mean_cross_vals_lang[lang].append(all_lang_for_split[lang])

    # mean_all_langs = {lang: round(np.mean(mean_cross_vals_lang[lang])*100, 2) for lang in LANGUAGES}
    # print(mean_all_langs)

    # print(round(np.mean(mean_cross_vals)*100, 2))

    # print("SIM MULTIPLE MODES OF TRANSFER FINETUNE")
    # mean_cross_vals_modes = {mode_transfer: [] for mode_transfer in modes_transfer}
    # mean_cross_vals_lang = {mode_transfer:  {lang: [] for lang in LANGUAGES} for mode_transfer in modes_transfer}
    # root_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval-finetune-valid_sim-cross_val_"
    # for i in range(5):
    #     with open(root_path + str(i) + "multi_mono_cross.pickle", "rb") as file:
    #         data = pickle.load(file)
    #     for mode_transfer in modes_transfer:
    #         mean_cross_vals_modes[mode_transfer].append(np.mean([v for k, v in data["multi"][mode_transfer][0].items()]))
    #         for lang in LANGUAGES:
    #             mean_cross_vals_lang[mode_transfer][lang].append(data["multi"][mode_transfer][0][lang])

    # for mode_transfer in modes_transfer:
    #     mean_all_langs = {lang: round(np.mean(mean_cross_vals_lang[mode_transfer][lang])*100, 2) for lang in LANGUAGES}
    #     print(mode_transfer, round(np.mean(mean_cross_vals_modes[mode_transfer])*100, 2), mean_all_langs)

    root_meta_align_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval-maml_align-valid-paraphrase-multilingual-mpnet-base-v2-cross_val_"
    # root_meta_align_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval-maml_align-valid-random-cross_val_"
    meta_distil_cross_vals = []
    meta_distil_langs = {lang: [] for lang in LANGUAGES}
    for i in range(5):
        with open(root_meta_align_path+str(i)+"multi_mono_cross.pickle", "rb") as file:
            data = pickle.load(file)

        print("-----------------------------------------------------------")
        for epoch in range(1):
            print(np.mean([v for k, v in data["multi"]["MONO_BIL_MULTI"][epoch].items()]))

        meta_distil_cross_vals.append(np.mean([v for k, v in data["multi"]["MONO_BIL_MULTI"][0].items()]))
        for lang in LANGUAGES:
            meta_distil_langs[lang].append(data["multi"]["MONO_BIL_MULTI"][0][lang])

    print("MEAN: ", round(np.mean(meta_distil_cross_vals)*100, 2), " STD:", round(np.std(meta_distil_cross_vals)*100, 2))
    for lang in LANGUAGES:
        print(lang, round(np.mean(meta_distil_langs[lang])*100, 2), round(np.std(meta_distil_langs[lang])*100, 2))

    exit(0)

    print("*********************************************************************************************************")
    modes_transfer = ["MONO_MONO", "MONO_BIL", "MONO_MULTI", "BIL_MULTI", "MIXT", "TRANS"]
    root_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/summary_eval"
    for upstream_model in ["maml", "finetune"]: 
        for sim_mode in ["sim", "rand"]:
            results_path = root_path+"-"+upstream_model
            if upstream_model == "maml":
                if sim_mode == "rand":
                    results_path += "-valid_rand-cross_val_"
                else:
                    results_path += "-new_valid_sim-cross_val_"
            else:
                if sim_mode == "rand":
                    results_path += "-valid-random-cross_val_"
                else:
                    results_path += "-valid_sim-cross_val_"

            mean_cross_vals_modes = {mode_transfer: [] for mode_transfer in modes_transfer}
            mean_cross_vals_lang = {mode_transfer:  {lang: [] for lang in LANGUAGES} for mode_transfer in modes_transfer}

            for i in range(5):
                results_path_val_split = results_path + str(i)
                if sim_mode == "rand" and upstream_model == "maml":
                    results_path_val_split += "-random"

                with open(results_path_val_split + "multi_mono_cross.pickle", "rb") as file:
                    data = pickle.load(file)

                if sim_mode == "sim" and upstream_model == "maml":
                    results_mono_bil_path = root_path+"-"+upstream_model+"-mono_bilONLY_train_sim-cross_val_" + str(i)
                    with open(results_mono_bil_path + "multi_mono_cross.pickle", "rb") as file:
                        data_mono_bil = pickle.load(file)
                
                
                for mode_transfer in modes_transfer:
                    if sim_mode == "sim" and upstream_model == "maml":
                        if mode_transfer == "MONO_BIL":
                            data_results = data_mono_bil
                        else:
                            data_results = data
                    else:
                        data_results = data

                    mean_cross_vals_modes[mode_transfer].append(np.mean([v for k, v in data_results["multi"][mode_transfer][0].items()]))
                    for lang in LANGUAGES:
                        mean_cross_vals_lang[mode_transfer][lang].append(data_results["multi"][mode_transfer][0][lang])
    

            print("-------------------------------------------------------------------------------------------------")
            print("upstream_model:", upstream_model, " sim_mode:", sim_mode)
            for mode_transfer in modes_transfer:
                mean_all_langs = {lang: round(np.mean(mean_cross_vals_lang[mode_transfer][lang])*100, 2) for lang in LANGUAGES}
                std_all_langs = {lang: round(np.std(mean_cross_vals_lang[mode_transfer][lang])*100, 2) for lang in LANGUAGES}
                print(mode_transfer, " MEAN: ", round(np.mean(mean_cross_vals_modes[mode_transfer])*100, 2), " STD: ", round(np.std(mean_cross_vals_modes[mode_transfer])*100, 2), " MEAN Languages: ", mean_all_langs, "STD Languages: ", std_all_langs)


def read_results(meta_task_mode, base_model, upstream_model, cross_val=-1, prefinetune=False, random=False):
    root_results_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/"+base_model

    middle_path = ""
    if prefinetune:
        middle_path += "PreFineTune/"

    middle_path += "TripletLoss/"

    if cross_val != -1:
        middle_path += "CrossVal_"+str(cross_val)+"/"

    if random:
        middle_path += "random/"

    middle_path += "runs/"

    test_results = []
    train_results = []
    for split in ["test"]: #["train", "valid", "test"]:
        results_file_name = os.path.join(root_results_path, upstream_model, meta_task_mode, middle_path, split+"_map_scores_lang.pickle")
        if os.path.exists(results_file_name):
            with open(results_file_name, "rb") as file:
                results = pickle.load(file)

            results_epochs = []
            for i in range(len(results)):
                results_epochs.append(np.mean([v for k, v in results[i].items()]))

            print("argmax:", np.argmax(results_epochs), " results_epochs:", len(results_epochs))

            per_lang_perf = {k: round(v*100, 2) for k, v in results[np.argmax(results_epochs)].items()}
            # per_lang_perf = {k: round(v*100, 2) for k, v in results[0].items()}
            print(split, upstream_model, meta_task_mode, cross_val, per_lang_perf, round(np.mean([v for k, v in per_lang_perf.items()]), 2))

            # print(split, mode, meta_task_mode, round(np.max(results_epochs)*100, 2))
            # print(mode, meta_task_mode, np.mean([v for k, v in results[0].items()]))#, results[-1])
        else:
            print("File doesn't exit:", results_file_name, meta_task_mode, cross_val)

def perf_multi_eval_results(args):
    base_model = "sbert-retrieval"
    for upstream_model in ["maml"]:
    # for upstream_model in ["maml", "finetune"]:#["finetune", "maml"]:
        # for meta_task_mode in ["MONO_MONO", "MONO_BIL", "MONO_MULTI", "BIL_MULTI", "MIXT", "TRANS"] : #"MONO_BIL", "MONO_MULTI", "BIL_MULTI", "MIXT", "TRANS"]:
        for meta_task_mode in ["MONO_MONO", "MONO_BIL", "MONO_MULTI", "BIL_MULTI", "MIXT", "TRANS"]:
            for i in range(5):
                # read_results(meta_task_mode, base_model, upstream_model, cross_val=-1, prefinetune=True, random=False)
                # read_results(meta_task_mode, base_model, upstream_model, prefinetune=True) # 
                # read_results(meta_task_mode, base_model, upstream_model, prefinetune=True, random=True) # RANDOM bert-retrieval
                # read_results(meta_task_mode, base_model, upstream_model, cross_val=i, prefinetune=False, random=True)
                read_results(meta_task_mode, base_model, upstream_model, cross_val=i, prefinetune=False, random=False)

            print("-----------------------------------")

def zero_shot_evaluation(transfer_mode, random_string, query_languages, candidate_languages, tokenizer, base_model,  meta_learn_split_config, args, cross_val_split, split_name="test"):
    # Get all candidates in answer_languages and convert them to features
    question_set = {split:{} for split in ["test"]}
    candidate_set = {split:{} for split in ["test"]}

    meta_tasks_dir = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/meta_tasks/TripletLoss/"+transfer_mode+"/"+random_string+"/ar,de,el,hi,ru,th,tr/CrossVal_"+str(cross_val_split)+"/"

    with open(os.path.join(meta_tasks_dir, split_name+"_question_set.pickle"), "rb") as file:
        question_set["test"] = pickle.load(file)

    with open(os.path.join(meta_tasks_dir, split_name+"_candidate_set.pickle"), "rb") as file:
        candidate_set["test"] = pickle.load(file)

    print("candidates: ", {lang: len(cand) for lang, cand in candidate_set["test"].by_lang.items()})
    print("questions: ", {lang: len(quest) for lang, quest in question_set["test"].by_lang.items()})
    use_base_model = True
    candidates_list = []
    all_uuids = []
    for candidate_language in candidate_languages:
        print("candidate_language:", candidate_language, "candidate_set[split_name].by_lang[candidate_language]:", len(candidate_set[split_name].by_lang[candidate_language]))
        candidates_list.extend(candidate_set[split_name].by_lang[candidate_language])
        all_uuids.extend([candidate.uid for candidate in candidate_set[split_name].by_lang[candidate_language]])
    print("all_uuids:", len(all_uuids), " len(candidates_list):", len(candidates_list))

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    
    if use_base_model:
        # all_candidates = [candidate for lang, candidate in candidate_set.by_lang.item() if lang in answer_languages]
        c_features = [tokenizer.encode_plus((candidate.sentence+candidate.context).replace("\n", ""),
                                            max_length=meta_learn_split_config["max_answer_length"],
                                            pad_to_max_length=True,
                                            return_token_type_ids=True) for candidate in candidates_list]


        batch_size = 24

        all_candidate_vecs_list = []
        for i_batch in range(0, len(c_features), batch_size):
            if i_batch + batch_size > len(c_features):
                last = len(c_features)
            else:
                last = i_batch + batch_size

            c_features_batch = c_features[i_batch: last]
            input_ids_batch = [c_features_["input_ids"] for c_features_ in c_features_batch]
            attention_mask_batch = [c_features_["attention_mask"] for c_features_ in c_features_batch]
            token_type_ids_batch = [c_features_["token_type_ids"] for  c_features_ in c_features_batch]

            c_features_new = {"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                              "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
                              "token_type_ids": torch.tensor(token_type_ids_batch, dtype=torch.long)}

            c_features_new = {k:v.to(args.device) for k, v in c_features_new.items()}

            base_model = base_model.to(args.device)

            with torch.no_grad():
                candidate_encoding = base_model(q_input_ids=c_features_new["input_ids"],
                                                q_attention_mask=c_features_new["attention_mask"],
                                                q_token_type_ids=c_features_new["token_type_ids"],
                                                inference=True)

            # batch_candidate_vecs = np.expand_dims(candidate_encoding, 0)

            all_candidate_vecs_list.append(candidate_encoding)

        all_candidate_vecs = np.concatenate(all_candidate_vecs_list, axis=0)
        print("all_candidate_vecs.shape:", all_candidate_vecs.shape)
        # print("all_candidate_vecs sum:", np.sum(all_candidate_vecs))

        # print("candidate.encoding.shape:", np.squeeze(candidate_set[split_name].as_list()[0].encoding).shape)
        print(np.sum(all_candidate_vecs))
    else:
        # all_candidate_vecs = np.concatenate([np.expand_dims(candidate.encoding, 0) for candidate in candidate_set[split_name].as_list()], axis=0)
        # all_candidate_vecs = np.concatenate([candidate.encoding for candidate in candidate_set[split_name].as_list()], axis=0)
        all_candidate_vecs = np.concatenate([candidate.encoding for candidate in candidates_list], axis=0) # np.concatenate([np.expand_dims(model.encode((candidate.sentence+candidate.context).replace("\n", "")), 0) for candidate in candidates_list], axis=0)
        # print("all_candidate_vecs.shape:", all_candidate_vecs.shape)
        print(np.sum(all_candidate_vecs))

    print("all_candidate_vecs.shape:", all_candidate_vecs.shape)


    map_scores_lang = {lang: 0.0 for lang in query_languages}
    for query_lang in tqdm(query_languages):
        # For each query language, we compute map for each query in all answer languages
        # print("Computing map for language %s ..."%query_lang)
        map_scores = []
        for question in tqdm(question_set[split_name].by_lang[query_lang]):
            # Get the encoding of the question using the base model
            if use_base_model:
                print("query_lang:", query_lang, " question.question:", question.question)
                base_model = base_model.to(args.device)
                q_features = tokenizer.encode_plus(question.question,
                                                   max_length=meta_learn_split_config["max_query_length"],
                                                   pad_to_max_length=True,
                                                   return_token_type_ids=True)
                # q_features_new = tokenizer(question.question, padding=True, truncation=True, return_tensors='pt')

                q_features_new = {"input_ids": torch.unsqueeze(torch.tensor(q_features["input_ids"], dtype=torch.long), dim=0),
                                  "attention_mask": torch.unsqueeze(torch.tensor(q_features["attention_mask"], dtype=torch.long), dim=0),
                                  "token_type_ids": torch.unsqueeze(torch.tensor(q_features["token_type_ids"], dtype=torch.long), dim=0)}

                q_features_new = {k:v.to(args.device) for k, v in q_features_new.items()}

                with torch.no_grad():
                    question_encoding = base_model(q_input_ids=q_features_new["input_ids"],
                                                   q_attention_mask=q_features_new["attention_mask"],
                                                   q_token_type_ids=q_features_new["token_type_ids"],
                                                   inference=True)

            else:
                question_encoding = question.encoding # np.expand_dims(model.encode(question.question), 0) 
                # question_encoding = np.expand_dims(question_encoding, 0)


            # print("question_encoding.shape:", question_encoding.shape)

            scores = question_encoding.dot(all_candidate_vecs.T)

            # print("scores.shape:", scores.shape)
            y_true = np.zeros(scores.shape[1])
            # print("y_true.shape:", y_true.shape)
            all_correct_cands = set(candidate_set[split_name].by_xling_id[question.xling_id])

            # positions = []
            # for ans in all_correct_cands:
            #     positions.append(candidate_set[split_name].pos[ans])
            #     y_true[candidate_set[split_name].pos[ans]] = 1
            # count = 0
            all_correct_uuids = [candidate.uid for candidate in all_correct_cands]
            other_positions = []
            for c_idx, uuid in enumerate(all_uuids):
                if uuid in all_correct_uuids:
                    other_positions.append(c_idx)
                    y_true[c_idx] = 1
                    # count += 1
            # print("positions:", positions, " other_positions:", other_positions)

            map_scores.append(utils_lareqa.average_precision_at_k(np.where(y_true == 1)[0], np.squeeze(scores).argsort()[::-1]))
        map_scores_lang[query_lang] = np.mean(map_scores)
    return map_scores_lang

def plot_confusion_matrix(args):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    mono_cross_perf = {'ar': {'ar': {'ar': 0.636247977744942}, 'de': {'ar': 0.5796067181941404}, 'el': {'ar': 0.5612908799581989}, 'hi': {'ar': 0.5739294212048544}, 'ru': {'ar': 0.5774700041827752}, 'th': {'ar': 0.5560516164557034}, 'tr': {'ar': 0.6262293289121869}},
                      'de': {'ar': {'de': 0.6306786908786691}, 'de': {'de': 0.7552826685425553}, 'el': {'de': 0.6696554065579099}, 'hi': {'de': 0.6585499473608825}, 'ru': {'de': 0.705740657344261}, 'th': {'de': 0.6462330364509029}, 'tr': {'de': 0.7346137519495184}}, 'el': {'ar': {'el': 0.6341129176521616}, 'de': {'el': 0.6902607122986159}, 'el': {'el': 0.6903929620385181}, 'hi': {'el': 0.6704098366826469}, 'ru': {'el': 0.6937645683883057}, 'th': {'el': 0.6463644959930355}, 'tr': {'el': 0.6898966264455624}}, 'hi': {'ar': {'hi': 0.6026821077171824}, 'de': {'hi': 0.6568170882471864}, 'el': {'hi': 0.6193880974129584}, 'hi': {'hi': 0.70564916824524}, 'ru': {'hi': 0.6461580727427437}, 'th': {'hi': 0.6279386705283013}, 'tr': {'hi': 0.6784618890767897}}, 'ru': {'ar': {'ru': 0.6290308729686924}, 'de': {'ru': 0.6885010855191648}, 'el': {'ru': 0.6329854723599674}, 'hi': {'ru': 0.6689185503361171}, 'ru': {'ru': 0.7162866549310065}, 'th': {'ru': 0.6669406260589683}, 'tr': {'ru': 0.6867917263776102}}, 'th': {'ar': {'th': 0.6381985673856334}, 'de': {'th': 0.681677166396833}, 'el': {'th': 0.6403265476786414}, 'hi': {'th': 0.6877234846524356}, 'ru': {'th': 0.6756895444381483}, 'th': {'th': 0.6974783939403948}, 'tr': {'th': 0.7056805551921481}}, 'tr': {'ar': {'tr': 0.6191223509451599}, 'de': {'tr': 0.7185364735799606}, 'el': {'tr': 0.6565133236899513}, 'hi': {'tr': 0.6482414419021918}, 'ru': {'tr': 0.6939518816306065}, 'th': {'tr': 0.6325894507369926}, 'tr': {'tr': 0.7435567109067701}}}
    
    array = []
    for lang in ["ar", "de", "el", "hi", "ru", "th", "tr"]:
        array_sub = []
        for lang1 in ["ar", "de", "el", "hi", "ru", "th", "tr"]:
            array_sub.append(round(mono_cross_perf[lang][lang2][lang]*100, 2))
        array.append(array_sub)

    df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"],
                    columns = [i for i in "ABCDEFGHIJK"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)

def perf_multi_mono_eval_results(args):
    base_model = "sbert-retrieval"
    upstream_model = "maml_align" # "maml" "maml_align"

    if upstream_model == "maml_align":
        meta_task_modes = ["MONO_BIL_MULTI"]
    else:
        meta_task_modes = ["MONO_MONO", "MONO_BIL", "BIL_MULTI", "MIXT", "MONO_MULTI", "TRANS"]

    train_valid_mode = "valid"
    cross_vals = list(range(5))
    random = True
    prefinetune = False
    if random:
        random_string = "random"
    else:
        random_string = "paraphrase-multilingual-mpnet-base-v2"
    
    root_results_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/"+base_model
    for cross_val in [0, 1]: #cross_vals:
        middle_path = ""
        if prefinetune:
            middle_path += "PreFineTune/"

        middle_path += "TripletLoss/"

        if cross_val != -1:
            middle_path += "CrossVal_"+str(cross_val)+"/"

        if random:
            middle_path += "random/"

        middle_path += "checkpoints/"

        LANGUAGES = ["ar", "de", "el", "hi", "ru", "th", "tr"]
        os.environ["CUDA_VISIBLE_DEVICES"]="7"
        args.device = torch.device("cuda")
        NUM_EPOCHS = 1
        all_multilingual_evaluations = {meta_task_mode: {} for meta_task_mode in meta_task_modes}
        all_mono_cross_evaluations = {meta_task_mode: {} for meta_task_mode in meta_task_modes}
        for meta_task_mode in meta_task_modes:
            test_multilingual_evaluation_epochs = {epoch: {} for epoch in range(NUM_EPOCHS)}
            mono_cross_evaluation_epochs = {epoch: {query_language:{candidate_language: {} for candidate_language in LANGUAGES} for query_language in LANGUAGES} for epoch in range(NUM_EPOCHS)}
            for epoch in range(NUM_EPOCHS):
                # if cross_val != 0:
                model_load_file = os.path.join(root_results_path, upstream_model, meta_task_mode, middle_path, "pytorch_model_"+train_valid_mode+str(epoch)+".bin") 
                if not os.path.exists(model_load_file):
                    model_load_file = os.path.join(root_results_path, upstream_model, meta_task_mode, middle_path, "pytorch_model_"+str(epoch)+".bin") 


                # else:
                # model_load_file = os.path.join(root_results_path, upstream_model, meta_task_mode, middle_path, "pytorch_model_"+str(epoch)+".bin") 
                model_name_or_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                cache_dir = ""
                config_name = None
                tokenizer_name = None

                config = AutoConfig.from_pretrained(config_name if config_name else model_name_or_path,
                                                    cache_dir=cache_dir if cache_dir else None)

                base_model = SBERTForRetrieval(config=config,
                                            trans_model_name=model_name_or_path)

                # base_model = SBERTForRetrieval.from_pretrained(model_name_or_path,
                #                                             from_tf=bool(".ckpt" in model_name_or_path),
                #                                             config=config,
                #                                             cache_dir=cache_dir if cache_dir else None)

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                                        #   do_lower_case=True,
                                                        cache_dir=cache_dir if cache_dir else None)

                base_model.to(args.device)

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

                test_multilingual_evaluation = zero_shot_evaluation(meta_task_mode, random_string, LANGUAGES, LANGUAGES, tokenizer, base_model, meta_learn_split_config, args, cross_val, split_name="test")
                test_multilingual_evaluation_epochs[epoch] = test_multilingual_evaluation
                print("Multi:", "epoch:", epoch, test_multilingual_evaluation," MEAN:", np.mean([test_multilingual_evaluation[lang] for lang in test_multilingual_evaluation if lang!= "en"]))
                for query_language in LANGUAGES:
                    for candidate_language in LANGUAGES:
                        mono_cross_evaluation_epochs[epoch][query_language][candidate_language] = zero_shot_evaluation(meta_task_mode, random_string, [query_language], [candidate_language], tokenizer, base_model, meta_learn_split_config, args, cross_val, split_name="test")
                        print(query_language, candidate_language, mono_cross_evaluation_epochs[epoch][query_language][candidate_language])


            all_multilingual_evaluations[meta_task_mode] = test_multilingual_evaluation_epochs
            all_mono_cross_evaluations[meta_task_mode] = mono_cross_evaluation_epochs

        save_path = root_results_path+"/summary_eval-"+upstream_model+"-"+train_valid_mode+"-"+random_string+"-cross_val_"+str(cross_val)
        # save_path = root_results_path + "/summary_eval-base-sbert-cross_val_"+str(cross_val)
        if random:
            save_path += "-random"
        with open(save_path+"multi_mono_cross.pickle", "wb") as file:
            pickle.dump({"multi": all_multilingual_evaluations, "mono_cross": all_mono_cross_evaluations},file)

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

    return optim

def perf_meta_test(args):
    base_model = "sbert-retrieval"
    upstream_model = "maml"
    meta_task_mode = "MONO_BIL"
    cross_val = 0
    random = False
    prefinetune = False
    root_results_path = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/"+base_model

    middle_path = ""
    if prefinetune:
        middle_path += "PreFineTune/"

    middle_path += "TripletLoss/"

    if cross_val != -1:
        middle_path += "CrossVal_"+str(cross_val)+"/"

    if random:
        middle_path += "random/"

    LANGUAGES = ["ar", "de", "el", "hi", "ru", "th", "tr"]
    epoch = 0
    model_load_file = os.path.join(root_results_path, upstream_model, meta_task_mode, middle_path, "checkpoints", "pytorch_model_"+str(epoch)+".bin") 
    model_name_or_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    cache_dir = ""
    config_name = None
    tokenizer_name = None

    os.environ["CUDA_VISIBLE_DEVICES"]="7"

    args.device = torch.device("cuda")

    config = AutoConfig.from_pretrained(config_name if config_name else model_name_or_path,
                                        cache_dir=cache_dir if cache_dir else None)

    base_model = SBERTForRetrieval(config=config,
                                    trans_model_name=model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                            cache_dir=cache_dir if cache_dir else None)

    base_model.to(args.device)

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

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
        "params": [p for n, p in base_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in base_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = AdamW(optimizer_grouped_parameters,
                lr=5e-5,
                eps=1e-8)

    opt_load_file = os.path.join(root_results_path, upstream_model, meta_task_mode, middle_path, "checkpoints", "optimizer_"+str(epoch)+".pt") 
    opt.load_state_dict(torch.load(opt_load_file, map_location=args.device)) 
    opt = optimizer_to(opt, args.device)

    Model = importlib.import_module('multi_meta_ssd.models.upstream.' + 'maml')
    meta_learner = Model.MetaLearner(tokenizer,
                                     base_model,
                                     args.device,
                                     meta_learn_split_config,
                                     opt)

    meta_tasks_dir = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/meta_tasks/TripletLoss/MONO_MONO/random/ar,de,el,hi,ru,th,tr/CrossVal_"+str(cross_val)+"/"

    with open(os.path.join(meta_tasks_dir, "test_meta_dataset.pickle"), "rb") as file:
        meta_dataset_test = pickle.load(file)
    
    # qry_shapes: [torch.Size([4, 96]), torch.Size([4, 96]), torch.Size([4, 96]), torch.Size([4, 256]), torch.Size([4, 256]), torch.Size([4, 256]), torch.Size([4, 256]), torch.Size([4, 256]), torch.Size([4, 256])]
    # qry_shapes: [torch.Size([4, 96]), torch.Size([4, 96]), torch.Size([4, 96]), torch.Size([4, 256]), torch.Size([4, 256]), torch.Size([4, 256]), torch.Size([4, 256]), torch.Size([4, 256]), torch.Size([4, 256])]
    n_tasks_batch = args.n_test_tasks_batch
    n_triplets = args.triplet_batch_size

    meta_tasks_test = meta_dataset_test.meta_tasks 

    rnd.shuffle(meta_tasks_test)

    runs_dir = os.path.join(root_results_path, upstream_model, meta_task_mode, middle_path, "runs")
    writer = SummaryWriter(runs_dir)

    for ep in range(1):
        for batch_step in tqdm(range(0, len(meta_tasks_test)//(n_tasks_batch*n_triplets), n_tasks_batch*n_triplets)):
            meta_tasks_batch = meta_tasks_test[batch_step: batch_step+(n_tasks_batch*n_triplets)]
            print("len(meta_tasks_batch):", len(meta_tasks_batch))
            print("list:", list(range(batch_step, batch_step+(n_tasks_batch*n_triplets), n_triplets)))
            use_triplet_loss = True
            if use_triplet_loss:
                # Concatenate n_tasks_batch meta_tasks
                meta_task_triplets = []
                meta_tasks_batches = []
                for k in range(0, n_tasks_batch*n_triplets, n_triplets):
                    meta_trip = meta_tasks_batch[k: k+n_triplets]
                    meta_task_triplets.append(meta_trip)
                    spt_features_triplets = []
                    qry_features_triplets = []
                    spt_questions = []
                    spt_candidates = []
                    qry_questions = []
                    qry_candidates = []
                    for k1 in range(n_triplets):
                        qry_features_triplets_parts = []
                        spt_features_triplets.append(meta_trip[k1].spt_features.items())

                        len_q = meta_trip[k1].spt_features["q_input_ids"].shape[0]
                        len_a = meta_trip[k1].spt_features["a_input_ids"].shape[0]
                        for s_n in range(len_a//len_q):
                            spt_questions.append([meta_trip[k1].spt.question_cluster])
                            spt_candidates.append([meta_trip[k1].spt.all_candidates])

                        for qr_k in range(len(meta_trip[k1].qry_features)):
                            qry_features_triplets_parts.append(meta_trip[k1].qry_features[qr_k].items())
                        len_q = meta_trip[k1].qry_features[qr_k]["q_input_ids"].shape[0]
                        len_a = meta_trip[k1].qry_features[qr_k]["a_input_ids"].shape[0]
                        for q_n in range(len_a//len_q):
                            qry_questions.append([meta_trip[k1].qry[q_n].question_cluster for q_n in range(len(meta_trip[k1].qry))])
                            qry_candidates.append([meta_trip[k1].qry[q_n].all_candidates for q_n in range(len(meta_trip[k1].qry))])
                        qry_features_triplets.append(qry_features_triplets_parts)

                    concatenated_spt_features_triplets = {k: [] for k, _ in spt_features_triplets[0]}
                    for spt_features in spt_features_triplets:
                        for k, v in spt_features:
                            concatenated_spt_features_triplets[k].append(v)

                    concatenated_spt_features_triplets_all = []
                    concatenated_spt_features_triplets = {k: torch.concat(concatenated_spt_features_triplets[k]) for k in concatenated_spt_features_triplets}
                    len_q = concatenated_spt_features_triplets["q_input_ids"].shape[0]
                    len_a = concatenated_spt_features_triplets["a_input_ids"].shape[0]
                    concatenated_spt_features_triplets_extend_l = []
                    for o in range(len_a//len_q):
                        concatenated_spt_features_triplets_extend = {}
                        concatenated_spt_features_triplets_extend.update({k: concatenated_spt_features_triplets[k] for k in ['q_input_ids', 'q_attention_mask', 'q_token_type_ids',
                                                                                                                                'n_input_ids', 'n_attention_mask', 'n_token_type_ids']})

                        concatenated_spt_features_triplets_extend.update({k: concatenated_spt_features_triplets[k][o*len_q:(o+1)*len_q] for k in ['a_input_ids', 'a_attention_mask', 'a_token_type_ids']})
                        concatenated_spt_features_triplets_extend_l.append(concatenated_spt_features_triplets_extend)
                    # print("concatenated_spt_features_triplets_shape:", [v.shape for k,v in concatenated_spt_features_triplets_extend_l[0].items()])
                    concatenated_spt_features_triplets_all.extend(concatenated_spt_features_triplets_extend_l)

                    concatenated_qry_features_triplets_all = []
                    for qry_features in qry_features_triplets:
                        concatenated_qry_features_triplets = {k: [] for k, _ in qry_features[0]}
                        for qr_k in range(len(qry_features)):
                            for k, v in qry_features[qr_k]:
                                concatenated_qry_features_triplets[k].append(v)
                        concatenated_qry_features_triplets = {k: torch.concat(concatenated_qry_features_triplets[k]) for k in concatenated_qry_features_triplets}
                        # print("concatenated_qry_features_triplets.keys():", concatenated_qry_features_triplets.keys())
                        len_q = concatenated_qry_features_triplets["q_input_ids"].shape[0]
                        len_a = concatenated_qry_features_triplets["a_input_ids"].shape[0]
                        concatenated_qry_features_triplets_extend_l = []
                        print("QUERY len_a//len_q:", len_a//len_q)
                        for o in range(len_a//len_q):
                            concatenated_qry_features_triplets_extend = {}
                            concatenated_qry_features_triplets_extend.update({k: concatenated_qry_features_triplets[k] for k in ['q_input_ids', 'q_attention_mask', 'q_token_type_ids',
                                                                                                                                    'n_input_ids', 'n_attention_mask', 'n_token_type_ids']})

                            concatenated_qry_features_triplets_extend.update({k: concatenated_qry_features_triplets[k][o*len_q:(o+1)*len_q] for k in ['a_input_ids', 'a_attention_mask', 'a_token_type_ids']})
                            concatenated_qry_features_triplets_extend_l.append(concatenated_qry_features_triplets_extend)
                        # print("concatenated_qry_features_triplets_shape:", [v.shape for k,v in concatenated_qry_features_triplets_extend_l[0].items()])
                        concatenated_qry_features_triplets_all.extend(concatenated_qry_features_triplets_extend_l)

                    meta_tasks_batches.append({"spt_features": concatenated_spt_features_triplets_all,
                                                "qry_features": concatenated_qry_features_triplets_all,
                                                "spt_questions": spt_questions,
                                                "spt_candidates": spt_candidates,
                                                "qry_questions": qry_questions,
                                                "qry_candidates": qry_candidates})
            else:
                spt_questions = []
                spt_candidates = []
                qry_questions = []
                qry_candidates = []
                for k in range(len(meta_tasks_batch)):
                    spt_questions.append([meta_tasks_batch[k].spt.question_cluster])
                    spt_candidates.append([meta_tasks_batch[k].spt.all_candidates])
                    qry_questions.append([meta_tasks_batch[k].qry[q_n].question_cluster for q_n in range(len(meta_tasks_batch[k].qry))])
                    qry_candidates.append([meta_tasks_batch[k].qry[q_n].all_candidates for q_n in range(len(meta_tasks_batch[k].qry))])


                meta_tasks_batches = [{"spt_features": meta_tasks_batch[k2].spt_features.items(), "qry_features": meta_tasks_batch[k2].qry_features,
                                        "spt_questions": spt_questions[k2], "spt_candidates": spt_candidates, "qry_questions": qry_questions[k2],
                                        "qry_candidates": qry_candidates} for k2 in range(len(meta_tasks_batch))]
        
            loss_qry_avg_batch, loss_qry_all, map_qry_all = meta_learner("test", meta_tasks_batches, ep, batch_step, writer)

            print("map_qry_all:", map_qry_all)
    
    