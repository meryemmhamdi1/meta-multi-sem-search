import os, torch, pickle, importlib, gc, csv
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr

from multi_meta_ssd.log import *
from multi_meta_ssd.commands.get_arguments import get_train_params, get_path_options, get_adam_optim_params, get_base_model_options, get_language_options, get_meta_task_options, get_translate_train_params
from multi_meta_ssd.commands.train_evaluate_utils import set_device, set_seed, get_config_params, optimizer_to, save_torch_model, circle_batch, load_torch_model, get_config_tokenizer_model, set_out_dir
from multi_meta_ssd.models.downstream.dual_encoders.sym_sent_trans import SBERTForRetrieval
from multi_meta_ssd.processors.upstream.meta_task_stsb import MetaDataset

from transformers import (AdamW, AutoTokenizer, AutoConfig)

# Tensorboard
try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

MODEL_CLASSES = {
    "sbert-retrieval": (AutoConfig, SBERTForRetrieval, AutoTokenizer)
}

LANGUAGES = ["en", "ar", "de", "es", "fr", "it", "ja", "nl", "pl", "pt", "ru", "tr", "zh"]
SPLIT_NAMES = ["train", "dev", "test"]
LANG_PAIRS = ["ar-ar", "ar-en", "es-es", "es-en", "en-en", "tr-en"]
LANG_TRACK_DICT = {"ar-ar": "1", "ar-en": "2", "es-es": "3", "es-en": "4a", "en-en": "5", "tr-en": "6"}
MAX_SENT_LEN = 100
EMBEDDER = None #SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def create_sym_search_parser(subparser):
    parser = subparser.add_parser("symsearch_train", help="Train on symmetric Search")
    get_train_params(parser)
    get_path_options(parser)
    get_adam_optim_params(parser)
    get_base_model_options(parser)
    get_language_options(parser)
    get_meta_task_options(parser)
    get_translate_train_params(parser)
    parser.set_defaults(func=run_main)

def read_csv_file(root_path):
    sentences_pair = {split_name: {lang: {"sentences1": [], "sentences2": [], "scores": [], "sentences1_feat": [], "sentences2_feat": []} for lang in LANGUAGES} for split_name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        for lang in LANGUAGES:
            with open(os.path.join(root_path, "stsb-multi-mt", "data", "stsb-"+lang+"-"+split_name+".csv")) as tsv_file:
                reader = csv.reader(tsv_file, delimiter=",")

                for _, line in enumerate(reader):
                    sent1, sent2, score = line
                    sentences_pair[split_name][lang]["sentences1"].append(sent1)
                    sentences_pair[split_name][lang]["sentences2"].append(sent2)
                    sentences_pair[split_name][lang]["scores"].append(float(score)/5.0)

    return sentences_pair

def read_multi_file(root_path, cross_val_split, tokenizer):
    sentences_pair = {split_name: {lang_pair: {} for lang_pair in LANG_TRACK_DICT} for split_name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        for lang_pair in LANG_TRACK_DICT:
            # with open(os.path.join(root_path, "STS2017.eval.v1.1", "STS.input.track"+LANG_TRACK_DICT[lang_pair]+"."+lang_pair+".txt"), "r") as file:
            with open(os.path.join(root_path, "cross_val", cross_val_split, split_name, lang_pair+".txt"), "r") as file:
                data = file.read().splitlines()

            sentences1 = []
            sentences2 = []
            scores = []

            for line in data:
                sent1, sent2, score = line.split("\t")
                sentences1.append(sent1)
                sentences2.append(sent2)
                scores.append(float(score)/5.0)

            sentences_pair[split_name][lang_pair] = {"sentences1": sentences1, "sentences2": sentences2, "scores": scores, "sentences1_feat": {}, "sentences2_feat": {}}

            sentences1_feat = [tokenizer.encode_plus(sent1,
                                                     max_length=MAX_SENT_LEN,
                                                     pad_to_max_length=True,
                                                     return_token_type_ids=True) for sent1 in sentences_pair[split_name][lang_pair]["sentences1"]]

            sentences2_feat = [tokenizer.encode_plus(sent2,
                                                     max_length=MAX_SENT_LEN,
                                                     pad_to_max_length=True,
                                                     return_token_type_ids=True) for sent2 in sentences_pair[split_name][lang_pair]["sentences2"]]


            sentences_pair[split_name][lang_pair]["sentences1_feat"] = sentences1_feat
            sentences_pair[split_name][lang_pair]["sentences2_feat"] = sentences2_feat

    return sentences_pair

def read_translation_data(tokenizer):
    root_path = "/project/jonmay_231/meryem/Datasets/STS2017/Translations/GoogleTrans/"
    sentences_pair = {split_name: {lang_pair: {} for lang_pair in LANG_TRACK_DICT} for split_name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        for lang_pair in LANG_TRACK_DICT:
            # with open(os.path.join(root_path, "STS2017.eval.v1.1", "STS.input.track"+LANG_TRACK_DICT[lang_pair]+"."+lang_pair+".txt"), "r") as file:
            with open(os.path.join(root_path, lang_pair+".txt"), "r") as file:
                data = file.read().splitlines()

            sentences1 = []
            sentences2 = []
            scores = []

            for line in data:
                sent1, sent2, score = line.split("\t")
                sentences1.append(sent1)
                sentences2.append(sent2)
                scores.append(float(score)/5.0)

            sentences_pair[split_name][lang_pair] = {"sentences1": sentences1, "sentences2": sentences2, "scores": scores, "sentences1_feat": {}, "sentences2_feat": {}}

            sentences1_feat = [tokenizer.encode_plus(sent1,
                                                     max_length=MAX_SENT_LEN,
                                                     pad_to_max_length=True,
                                                     return_token_type_ids=True) for sent1 in sentences_pair[split_name][lang_pair]["sentences1"]]

            sentences2_feat = [tokenizer.encode_plus(sent2,
                                                     max_length=MAX_SENT_LEN,
                                                     pad_to_max_length=True,
                                                     return_token_type_ids=True) for sent2 in sentences_pair[split_name][lang_pair]["sentences2"]]


            sentences_pair[split_name][lang_pair]["sentences1_feat"] = sentences1_feat
            sentences_pair[split_name][lang_pair]["sentences2_feat"] = sentences2_feat

    return sentences_pair

def create_meta_dataset_stsb(args, meta_learn_split_config, meta_tasks_dir, tokenizer, embedder=None):
    meta_dataset = {split:{} for split in SPLIT_NAMES}

    sentences_pair = read_csv_file(args.data_root)
    eval_sentences_pair = read_multi_file(args.data_root, args.cross_val_split, tokenizer)

    for split_name in SPLIT_NAMES:
        for lang in LANGUAGES:

            sentences1_feat = [tokenizer.encode_plus(sent1,
                                                     max_length=MAX_SENT_LEN,
                                                     pad_to_max_length=True,
                                                     return_token_type_ids=True) for sent1 in sentences_pair[split_name][lang]["sentences1"]]

            sentences2_feat = [tokenizer.encode_plus(sent2,
                                                     max_length=MAX_SENT_LEN,
                                                     pad_to_max_length=True,
                                                     return_token_type_ids=True) for sent2 in sentences_pair[split_name][lang]["sentences2"]]


            sentences_pair[split_name][lang]["sentences1_feat"] = sentences1_feat
            sentences_pair[split_name][lang]["sentences2_feat"] = sentences2_feat

    meta_file_names = [os.path.join(meta_tasks_dir, split_name+"_meta_dataset.pickle") for split_name in SPLIT_NAMES]

    if False:#not args.update_meta_data and all([os.path.isfile(meta_file_names[i]) for i in range(len(SPLIT_NAMES))]):
        print("Metadatasets exit so will load them ONLYYY ", meta_tasks_dir)
        for split_name in SPLIT_NAMES:
            with open(os.path.join(meta_tasks_dir, split_name+"_meta_dataset.pickle"), "rb") as file:
                meta_dataset[split_name] = pickle.load(file)
    else:
        print("META DATASETS DON'T EXIST SO WILL BE CREATED FROM SCRATCH ", meta_tasks_dir)
        for split_name in SPLIT_NAMES:
            # Create meta-dataset tasks for that split
            logger.info("------>Constructing the meta-dataset for {} number of tasks {} .... ".format(split_name,
                                                                                                      meta_learn_split_config[split_name]["n_tasks_total"]))
            if args.translate_train:
                meta_sentences_pair = sentences_pair
            else:
                meta_sentences_pair = eval_sentences_pair

            meta_dataset[split_name] = MetaDataset(meta_learn_split_config,
                                                   meta_sentences_pair,
                                                   split_name,
                                                   tokenizer,
                                                   args.translate_train)

            with open(os.path.join(meta_tasks_dir, split_name+"_meta_dataset.pickle"), "wb") as file:
                pickle.dump(meta_dataset[split_name], file)

    meta_tasks = {split_name: meta_dataset[split_name].meta_tasks for split_name in SPLIT_NAMES}

    logger.info({split_name: len(meta_tasks[split_name]) for split_name in SPLIT_NAMES})

    return meta_dataset, meta_tasks, sentences_pair, eval_sentences_pair

def zero_shot_evaluation(sentences_pair, lang, tokenizer, base_model,  meta_learn_split_config, args, split_name):
    computed_scores = []
    sentences1 = sentences_pair[split_name][lang]["sentences1"]
    sentences2 = sentences_pair[split_name][lang]["sentences2"]
    scores_gs = sentences_pair[split_name][lang]["scores"]
    for i in range(len(sentences1)):
        # Get the encoding of the question using the base model

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
            sentence1_encoding = base_model(sent1_input_ids=q_features_sent1["input_ids"],
                                            sent1_attention_mask=q_features_sent1["attention_mask"],
                                            sent1_token_type_ids=q_features_sent1["token_type_ids"],
                                            inference=True)

            sentence2_encoding = base_model(sent1_input_ids=q_features_sent2["input_ids"],
                                            sent1_attention_mask=q_features_sent2["attention_mask"],
                                            sent1_token_type_ids=q_features_sent2["token_type_ids"],
                                            inference=True)

        scores = sentence1_encoding.dot(sentence2_encoding.T)
    
        computed_scores.append(scores[0][0])
    
    scores_normalized = (computed_scores - np.min(scores_gs)) / (np.max(scores_gs) - np.min(scores_gs))

    min_ = min(len(scores_gs), len(scores_normalized))
    correlation, _ = pearsonr(scores_gs[:min_], scores_normalized[:min_])
   
    return correlation

def train_validate(args, meta_learn_split_config, meta_tasks, meta_tasks_dir, sentences_pair, eval_sentences_pair, tokenizer, base_model, loss_qry_avg_batch_total, loss_qry_all_total, map_qry_all_total, checkpoints_dir, writer):
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

    opt = optimizer_to(opt, args.device)
    scores_lang_dict = {split_name: [] for split_name in SPLIT_NAMES+["eval_multi"]}
    meta_learn_alg_name = args.meta_learn_alg
    if args.meta_learn_alg == "maml":
        meta_learn_alg_name += "_stsb"

    Model = importlib.import_module('multi_meta_ssd.models.upstream.' + meta_learn_alg_name)
    meta_learner = Model.MetaLearner(base_model,
                                     args.device,
                                     meta_learn_split_config,
                                     opt)
    if args.use_meta_learn:
        for ep in tqdm(range(args.num_train_epochs)):
            for split_name in SPLIT_NAMES:
                n_tasks_batch = meta_learner.meta_learn_config[split_name]["n_tasks_batch"]

                print("split_name:", split_name, " n_tasks_batch:", n_tasks_batch, " len(meta_tasks[split_name]):", len(meta_tasks[split_name]),
                    " len(meta_tasks[split_name])//n_tasks_batch:", len(meta_tasks[split_name])//n_tasks_batch)

                meta_task_batch_gen = circle_batch(meta_tasks[split_name], n_tasks_batch)
                for batch_step in tqdm(range(0, len(meta_tasks[split_name])//n_tasks_batch)):
                    meta_tasks_batch = next(meta_task_batch_gen)

                    loss_qry_avg_batch, loss_qry_all, map_qry_all = meta_learner(split_name, meta_tasks_batch, ep, batch_step, writer)
                    # loss_qry_avg_batch, loss_qry_all, map_qry_all = meta_learner(split_name, meta_tasks_batches, ep, batch_step, writer)
                    torch.cuda.empty_cache()
                    gc.collect()

                    loss_qry_avg_batch_total[split_name][ep].append(loss_qry_avg_batch)
                    loss_qry_all_total[split_name][ep].append(loss_qry_all)
                    map_qry_all_total[split_name][ep].append(map_qry_all)

                    for lang_pair in LANG_PAIRS:
                        test_evaluation = zero_shot_evaluation(eval_sentences_pair, lang_pair, tokenizer, base_model,  meta_learn_split_config, args, "test")
                        print(lang_pair, test_evaluation)

                    save_torch_model(args, meta_learner.base_model, meta_learner.opt, checkpoints_dir, split_name+str(ep)+str(batch_step))

                    save_torch_model(args, meta_learner.maml, meta_learner.opt, checkpoints_dir, "maml_"+split_name+str(ep)+str(batch_step))
                    save_torch_model(args, meta_learner.learner, meta_learner.opt, checkpoints_dir, "learner_"+split_name+str(ep)+str(batch_step))

    else:       
        if args.translate_train:
            translate_train_languages = args.translate_train_langs.split(",")
            X = []
            Y = []
            S = []
            for translate_train_lang in translate_train_languages:
                X.extend(sentences_pair["train"][translate_train_lang]["sentences1_feat"])
                Y.extend(sentences_pair["train"][translate_train_lang]["sentences2_feat"])
                S.extend(sentences_pair["train"][translate_train_lang]["scores"])

            pbar = tqdm(range(args.num_train_epochs))
            pbar.set_description("Training Epoch Progress")
            for epoch in pbar:
                # X is a torch Variable
                permutation = torch.randperm(len(X))

                pbar_b = tqdm(range(0, len(X), args.batch_size))
                pbar_b.set_description(" --- Batch Progress")

                for i in pbar_b:
                    opt.zero_grad()

                    indices = permutation[i:i+args.batch_size]

                    sent1_inputs_ids = [X[j]["input_ids"] for j in indices]
                    sent1_attention_mask = [X[j]["attention_mask"] for j in indices]
                    sent1_token_type_ids = [X[j]["token_type_ids"] for j in indices]

                    sent2_inputs_ids = [Y[j]["input_ids"] for j in indices]
                    sent2_attention_mask = [Y[j]["attention_mask"] for j in indices]
                    sent2_token_type_ids = [Y[j]["token_type_ids"] for j in indices]

                    scores_gs = [S[j] for j in indices]


                    inputs = {"sent1_input_ids": torch.tensor(sent1_inputs_ids, dtype=torch.long),
                              "sent1_attention_mask": torch.tensor(sent1_attention_mask, dtype=torch.long),
                              "sent1_token_type_ids": torch.tensor(sent1_token_type_ids, dtype=torch.long),
                              "sent2_input_ids": torch.tensor(sent2_inputs_ids, dtype=torch.long),
                              "sent2_attention_mask": torch.tensor(sent2_attention_mask, dtype=torch.long),
                              "sent2_token_type_ids": torch.tensor(sent2_token_type_ids, dtype=torch.long),
                              "scores_gs": torch.tensor(scores_gs, dtype=torch.float)} # 

                    inputs = {k:v.to(args.device) for k, v in inputs.items()}
                    base_model = base_model.to(args.device)

                    outputs = base_model(**inputs)

                    loss, q_encodings, a_encodings = outputs

                    loss.backward()
                    opt.step()

                    base_model.zero_grad()

                logger.info("Test Zero-shot Evaluation from that model on stsb-multi-mt data at the End of the epoch")
                for lang_pair in LANG_PAIRS:
                    scores_lang = zero_shot_evaluation(eval_sentences_pair, lang_pair, tokenizer, base_model,  meta_learn_split_config, args, "test")
                    print("lang_pair:", lang_pair, " scores_lang:", scores_lang)

                    scores_lang_dict["eval_multi"].append({lang_pair: scores_lang})

                ## Save FINAL MODEL
                save_torch_model(args, base_model, opt, checkpoints_dir, str(epoch))

            ## Save FINAL MODEL
            save_torch_model(args, base_model, opt, checkpoints_dir, "final")

    return loss_qry_avg_batch_total, loss_qry_all_total, map_qry_all_total, scores_lang_dict

import sys
def run_main(args):
    print("Running main Symmetric Semantic Search")

    # Setting up the configuration parameters by reading them from their paths
    args, meta_learn_split_config = get_config_params(args, 'sym')
    logger.info(args)

    # Setting the seed and the device
    set_seed(args)
    set_device(args)

    # Setting the output directory
    meta_tasks_dir, checkpoints_dir, runs_dir, logs_dir, writer = set_out_dir(args, "sym")

    stdoutOrigin = sys.stdout
    sys.stdout = open(os.path.join(logs_dir, args.logs_file), "w")

    # Setting logstats and writing config to json file
    logger.info("Saving to logs_dir: {}".format(logs_dir))
    logstats_init(os.path.join(logs_dir, args.stats_file))

    config_path = os.path.join(logs_dir, 'config.json')
    logstats_add_args('config', args)
    args_var = {k: v for k, v in vars(args).items() if k not in ["func", "device"]}
    logstats_write_json(args_var, config_path)

    # Loading config, tokenizer, and downstream model
    tokenizer, base_model = get_config_tokenizer_model(MODEL_CLASSES, args)

    ## Create/load meta-dataset
    meta_dataset, meta_tasks, sentences_pair, eval_sentences_pair = create_meta_dataset_stsb(args,
                                                                                             meta_learn_split_config,
                                                                                             meta_tasks_dir,
                                                                                             tokenizer,
                                                                                             base_model)
    
    for lang_pair in LANG_PAIRS:
        test_evaluation = zero_shot_evaluation(eval_sentences_pair, lang_pair, tokenizer, base_model,  meta_learn_split_config, args, "test")
        print("INITIAL map_scores_lang: ", lang_pair, test_evaluation)

    # for lang in LANGUAGES:
    #     test_evaluation = zero_shot_evaluation(sentences_pair, lang, tokenizer, base_model,  meta_learn_split_config, args, "test")
    #     print("INITIAL TRANSLATE-TEST map_scores_lang: ", lang, test_evaluation)

    if args.do_evaluate:
        ## Train and validation
        opt = torch.optim.Adam(base_model.parameters(),
                               lr=meta_learn_split_config["train"]["beta_lr"])

        # model_load_file = args.load_pre_finetune_path + "pytorch_model.bin"
        # optim_load_file = args.load_pre_finetune_path + "training_args.bin"

        optim_load_file = os.path.join(checkpoints_dir, "optimizer.pt")
        model_load_file =os.path.join(checkpoints_dir, "pytorch_model.bin")
        opt, base_model = load_torch_model(args, opt, base_model, optim_load_file, model_load_file)

        logger.info("Multilingual Evaluation for that model")
        for lang_pair in LANG_PAIRS:
            test_evaluation = zero_shot_evaluation(sentences_pair, lang_pair, tokenizer, base_model,  meta_learn_split_config, args, "test")
            print("INITIAL map_scores_lang: ", lang_pair, test_evaluation)
    else:
        def split_ep_dict(args):
            return {split_name: {ep:[] for ep in range(args.num_train_epochs)} for split_name in SPLIT_NAMES}

        loss_qry_all_total, map_qry_all_total, loss_qry_avg_batch_total = split_ep_dict(args) , \
                                                                          split_ep_dict(args), \
                                                                          split_ep_dict(args)

        loss_qry_avg_batch_total, loss_qry_all_total, map_qry_all_total, scores_lang_dict = train_validate(args,
                                                                                                           meta_learn_split_config,
                                                                                                           meta_tasks,
                                                                                                           meta_tasks_dir,
                                                                                                           sentences_pair,
                                                                                                           eval_sentences_pair, 
                                                                                                           tokenizer,
                                                                                                           base_model,
                                                                                                           loss_qry_avg_batch_total,
                                                                                                           loss_qry_all_total,
                                                                                                           map_qry_all_total,
                                                                                                           checkpoints_dir,
                                                                                                           writer)


        for split_name in SPLIT_NAMES+["eval_multi"]:
            with open(os.path.join(runs_dir, split_name+"_scores_lang.pickle"), "wb") as file:
                pickle.dump(scores_lang_dict[split_name], file)

    sys.stdout.close()
    sys.stdout = stdoutOrigin