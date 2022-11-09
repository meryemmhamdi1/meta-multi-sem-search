from tkinter import E
from pyparsing import str_type
# import logging as logger
from multi_meta_ssd.log import *
from multi_meta_ssd.commands.get_arguments import get_train_params, get_path_options, get_adam_optim_params, get_base_model_options, get_language_options, get_meta_task_options, get_translate_train_params
from multi_meta_ssd.models.downstream.dual_encoders.sym_sent_trans import SBERTForRetrieval
from multi_meta_ssd.commands.train_evaluate import load_and_cache_examples, train, evaluate, set_device
from multi_meta_ssd.processors.downstream import utils_lareqa
from multi_meta_ssd.processors.upstream.meta_task import MetaDataset
from multi_meta_ssd.models.upstream.maml import MetaLearner
from sentence_transformers import SentenceTransformer, util

from tqdm import tqdm
import numpy as np
import os, json, random, torch, pickle, importlib, sys, configparser, gc, csv
from transformers import (
    AdamW, BertConfig, BertTokenizer, WEIGHTS_NAME, XLMRobertaTokenizer,
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup)
# Tensorboard
try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

from scipy.stats import pearsonr

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

def get_config_params(args):
    paths = configparser.ConfigParser()
    paths.read('multi_meta_ssd/config/paths.ini')

    location = "ENDEAVOUR_SYM"

    root_dir = str(paths.get(location, "ROOT"))

    args.data_root = root_dir + str(paths.get(location, "DATA_ROOT"))
    args.train_file = root_dir + str(paths.get(location, "TRAIN_FILE"))
    args.predict_file = root_dir + str(paths.get(location, "PREDICT_FILE"))
    args.out_dir = root_dir + str(paths.get(location, "OUT_DIR"))
    args.load_pre_finetune_path = root_dir + str(paths.get(location, "LOAD_PRE_FINETUNE_PATH"))

    print("OUT_DIR:", args.out_dir)

    params = configparser.ConfigParser()
    params.read('multi_meta_ssd/config/down_model_param.ini')
    
    args.pre_finetune_language = str(params.get("PARAMS", "PRE_FINETUNE_LANGUAGE"))

    args.max_seq_length = int(params.get("PARAMS", "MAX_SEQ_LENGTH"))
    args.max_query_length = int(params.get("PARAMS", "MAX_QUERY_LENGTH"))
    args.max_answer_length = int(params.get("PARAMS", "MAX_ANSWER_LENGTH"))

    params = configparser.ConfigParser()
    params.read('multi_meta_ssd/config/meta_task_param.ini')

    args.train_lang_pairs = str(params.get(args.mode_transfer, "TRAIN_LANG_PAIRS"))
    args.valid_lang_pairs = str(params.get(args.mode_transfer, "VALID_LANG_PAIRS"))
    args.test_lang_pairs = str(params.get(args.mode_transfer, "TEST_LANG_PAIRS"))

    return args

def set_out_dir(args):
    print(args.out_dir,
          args.model_type,
          args.meta_learn_alg if args.use_meta_learn else "finetune",
          args.mode_transfer)

    out_dir = os.path.join(args.out_dir,
                           args.model_type,
                           args.meta_learn_alg if args.use_meta_learn else "finetune",
                           args.mode_transfer)

    if args.do_pre_finetune:
        out_dir = os.path.join(out_dir, "PreFineTune")

    if args.translate_train:
        out_dir = os.path.join(out_dir, "TRANSLATE_TRAIN", args.translate_train_langs)

    if args.use_cross_val:
        out_dir = os.path.join(out_dir, "CrossVal_"+args.cross_val_split)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    runs_dir = os.path.join(out_dir, 'runs')
    if not os.path.isdir(runs_dir):
        os.makedirs(runs_dir)

    logs_dir = os.path.join(out_dir, 'logs')
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    #########################################

    meta_tasks_dir = os.path.join(args.out_dir,
                                  args.model_type,
                                  "meta_tasks",
                                  args.mode_transfer,
                                  ",".join(sorted(set(args.languages.split(","))))) # different path for meta_tasks


    if args.use_cross_val:
        meta_tasks_dir = os.path.join(meta_tasks_dir, "CrossVal_"+args.cross_val_split)

    if not os.path.isdir(meta_tasks_dir):
        os.makedirs(meta_tasks_dir)

    writer = SummaryWriter(runs_dir)

    return meta_tasks_dir, checkpoints_dir, runs_dir, logs_dir, writer

def save_torch_model(args, model, optimizer, checkpoints_dir, epoch):
    args_save_file = os.path.join(checkpoints_dir, "training_args_"+epoch+".bin")
    model_save_file = os.path.join(checkpoints_dir, "pytorch_model_"+epoch+".bin")
    optim_save_file = os.path.join(checkpoints_dir, "optimizer_"+epoch+".pt")
    torch.save(args, args_save_file)
    torch.save(model.state_dict(), model_save_file)
    torch.save(optimizer.state_dict(), optim_save_file)

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

def load_torch_model(args, optimizer, model, optim_load_file, model_load_file):
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    # print("State of loaded optimizer:")
    # loaded_optimizer = torch.load(optim_load_file)
    # for var_name, val in loaded_optimizer.items():
    #     print(var_name, "\t", loaded_optimizer[var_name])

    optimizer.load_state_dict(torch.load(optim_load_file, map_location=args.device)) # TODO see what's going on with the optimizer here
    # loaded_model = torch.load(model_load_file)
    # loaded_model_dict = [k for k,v in loaded_model.items()]
    # base_model_dict = model.state_dict().keys()
    # print("******************************")
    # print("Intersection:", list(set(loaded_model_dict) & set(base_model_dict)))
    # print("Not intersection:", set(loaded_model_dict) ^ set(base_model_dict))
    # print(loaded_model.state_dict)
    model.load_state_dict(torch.load(model_load_file, map_location=args.device), strict=False)

    return optimizer, model

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
                    sentences_pair[split_name][lang]["scores"].append(float(score))
    return sentences_pair

def read_multi_file(root_path):
    sentences_pair = {"test": {lang_pair: {} for lang_pair in LANG_TRACK_DICT}}

    for lang_pair in LANG_TRACK_DICT:
        with open(os.path.join(root_path, "STS2017.eval.v1.1", "STS.input.track"+LANG_TRACK_DICT[lang_pair]+"."+lang_pair+".txt"), "r") as file:
            data = file.read().splitlines()

        sentences1 = []
        sentences2 = []

        for line in data:
            sent1, sent2 = line.split("\t")
            sentences1.append(sent1)
            sentences2.append(sent2)

        with open(os.path.join(root_path, "STS2017.gs", "STS.gs.track"+LANG_TRACK_DICT[lang_pair]+"."+lang_pair+".txt"), "r") as file:
            scores = file.read().splitlines()

        scores = [float(score) for score in scores]

        sentences_pair["test"][lang_pair] = {"sentences1": sentences1, "sentences2": sentences2, "scores": scores}

    return sentences_pair

def create_meta_dataset_stsb(args, meta_learn_split_config, meta_tasks_dir, tokenizer, embedder=None):
    meta_dataset = {split:{} for split in SPLIT_NAMES}

    sentences_pair = read_csv_file(args.data_root)
    eval_sentences_pair = read_multi_file(args.data_root)

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

    meta_tasks = None # TODO Later
    # meta_file_names = [os.path.join(meta_tasks_dir, split_name+"_meta_dataset.pickle") for split_name in SPLIT_NAMES]

    # if not args.update_meta_data and all([os.path.isfile(meta_file_names[i]) for i in range(len(SPLIT_NAMES))]):
    #     print("Metadatasets exit so will load them ONLYYY ", meta_tasks_dir)
    #     for split_name in SPLIT_NAMES:
    #         with open(os.path.join(meta_tasks_dir, split_name+"_meta_dataset.pickle"), "rb") as file:
    #             meta_dataset[split_name] = pickle.load(file)
    # else:
    #     print("META DATASETS DON'T EXIST SO WILL BE CREATED FROM SCRATCH ", meta_tasks_dir)
    #     for split_name in SPLIT_NAMES:

    #         top_results = None # get_all_embeddings_similarities(question_set[split_name]) # TODO Later

    #         # Create meta-dataset tasks for that split
    #         logger.info("------>Constructing the meta-dataset for {} number of tasks {} .... ".format(split_name,
    #                                                                                                   meta_learn_split_config[split_name]["n_tasks_total"]))
    #         meta_dataset[split_name] = MetaDataset(meta_learn_split_config,
    #                                                sentences_pair,
    #                                                split_name,
    #                                                tokenizer,
    #                                                top_results)

    #         with open(os.path.join(meta_tasks_dir, split_name+"_meta_dataset.pickle"), "wb") as file:
    #             pickle.dump(meta_dataset[split_name], file)

    # meta_tasks = {split_name: meta_dataset[split_name].meta_tasks for split_name in SPLIT_NAMES}

    # logger.info({split_name: len(meta_tasks[split_name]) for split_name in SPLIT_NAMES})

    return meta_dataset, meta_tasks, sentences_pair, eval_sentences_pair

def zero_shot_evaluation(sentences_pair, lang, tokenizer, base_model,  meta_learn_split_config, args, split_name):
    computed_scores = []
    for i in range(len(sentences_pair[split_name][lang]["sentences1"])):
        # Get the encoding of the question using the base model
        sentences1 = sentences_pair[split_name][lang]["sentences1"]
        sentences2 = sentences_pair[split_name][lang]["sentences2"]
        scores_gs = sentences_pair[split_name][lang]["scores"]

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
    print("scores_normalized:", scores_normalized, "computed_scores:", computed_scores)

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
    # TODO ADD PREFINETUNING
    if args.translate_train:
        #     model_load_file = args.load_pre_finetune_path + "pytorch_model.bin"
        #     optim_load_file = args.load_pre_finetune_path + "optimizer.pt"

        #     logger.info("Finished Loading SQUAD torch model")
        #     opt, base_model = load_torch_model(args, opt, base_model, optim_load_file, model_load_file)

        #     logger.info("Finished Loading SQUAD torch model")
        #     print("Finished LOADING PREFINETUNING MODEL")

        #     for lang_pair in LANG_PAIRS:
        #         test_evaluation = zero_shot_evaluation(eval_sentences_pair, lang_pair, tokenizer, base_model,  meta_learn_split_config, args, "test")
        #         print("scores_lang:", test_evaluation)

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

                q_inputs_ids = [X[j]["input_ids"] for j in indices]
                q_attention_mask = [X[j]["attention_mask"] for j in indices]
                q_token_type_ids = [X[j]["token_type_ids"] for j in indices]

                a_inputs_ids = [Y[j]["input_ids"] for j in indices]
                a_attention_mask = [Y[j]["attention_mask"] for j in indices]
                a_token_type_ids = [Y[j]["token_type_ids"] for j in indices]

                scores_gs = [S[j] for j in indices]


                inputs = {"q_input_ids": torch.tensor(q_inputs_ids, dtype=torch.long),
                          "q_attention_mask": torch.tensor(q_attention_mask, dtype=torch.long),
                          "q_token_type_ids": torch.tensor(q_token_type_ids, dtype=torch.long),
                          "a_input_ids": torch.tensor(a_inputs_ids, dtype=torch.long),
                          "a_attention_mask": torch.tensor(a_attention_mask, dtype=torch.long),
                          "a_token_type_ids": torch.tensor(a_token_type_ids, dtype=torch.long),
                          "scores_gs": torch.tensor(scores_gs, dtype=torch.float)} # 

                inputs = {k:v.to(args.device) for k, v in inputs.items()}
                base_model = base_model.to(args.device)

                outputs = base_model(**inputs)

                loss, q_encodings, a_encodings = outputs

                loss.backward()
                opt.step()

                base_model.zero_grad()

            # logger.info("Test Evaluation from that model at the End of the epoch")
            # for split_name in SPLIT_NAMES:
            #     for lang in LANGUAGES:
            #         scores_lang = zero_shot_evaluation(sentences_pair, lang, tokenizer, base_model,  meta_learn_split_config, args, split_name)
            #         print("lang:", lang, " scores_lang:", scores_lang)

            #         scores_lang_dict[split_name].append({lang: scores_lang})


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

def run_main(args):
    # Setting up the configuration parameters by reading them from their paths
    args = get_config_params(args)

    # Setting the logger
    logger.info(args)

    # Setting the device
    set_device(args)

    # Setting the output directory
    meta_tasks_dir, checkpoints_dir, runs_dir, logs_dir, writer = set_out_dir(args)

    # Sys stdout options
    # stdoutOrigin = sys.stdout
    # sys.stdout = open(os.path.join(logs_dir, args.logs_file), "w")

    logger.info("Saving to logs_dir: {}".format(logs_dir))
    logstats_init(os.path.join(logs_dir, args.stats_file))

    config_path = os.path.join(logs_dir, 'config.json')
    logstats_add_args('config', args)

    args_var = {k: v for k, v in vars(args).items() if k not in ["func", "device"]}
    logstats_write_json(args_var, config_path)

    # Loading config, tokenizer, and downstream model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                               #   do_lower_case=True,
                                               cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_type == "sbert-retrieval":
        base_model = model_class(config=config,
                                 trans_model_name=args.model_name_or_path) #bert-base-multilingual-cased)
    else:
        base_model = model_class.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path,
                                                 config=config,
                                                 from_tf=bool(".ckpt" in args.model_name_or_path),
                                                 cache_dir=args.cache_dir if args.cache_dir else None)

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

    ## Create/load meta-dataset
    meta_dataset, meta_tasks, sentences_pair, eval_sentences_pair = create_meta_dataset_stsb(args,
                                                                                             meta_learn_split_config,
                                                                                             meta_tasks_dir,
                                                                                             tokenizer,
                                                                                             base_model)
    
    for lang_pair in ["tr-en"]: #LANG_PAIRS:
        test_evaluation = zero_shot_evaluation(eval_sentences_pair, lang_pair, tokenizer, base_model,  meta_learn_split_config, args, "test")
        print("INITIAL map_scores_lang: ", lang_pair, test_evaluation)

    exit(0)

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

    # sys.stdout.close()
    # sys.stdout = stdoutOrigin
