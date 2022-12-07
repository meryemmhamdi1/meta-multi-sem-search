import os, json, random, torch, pickle, importlib, gc, sys
from tqdm import tqdm
import numpy as np

# import logging as logger
from multi_meta_ssd.log import *
from multi_meta_ssd.commands.get_arguments import get_train_params, get_path_options, get_adam_optim_params, get_base_model_options, get_language_options, get_meta_task_options, get_translate_train_params
from multi_meta_ssd.commands.train_evaluate_utils import set_device, set_seed, get_config_params, save_torch_model, load_torch_model, optimizer_to, get_config_tokenizer_model, get_split_dict, split_ep_dict, set_out_dir
from multi_meta_ssd.models.downstream.dual_encoders.bert import BertForRetrieval
from multi_meta_ssd.models.downstream.dual_encoders.sent_trans import SBERTForRetrieval
from multi_meta_ssd.models.downstream.dual_encoders.xlm_roberta import XLMRobertaConfig, XLMRobertaForRetrieval
from multi_meta_ssd.processors.downstream import utils_lareqa
from multi_meta_ssd.processors.upstream.meta_task import MetaDataset
from sentence_transformers import SentenceTransformer, util

from transformers import (
    AdamW, BertConfig, BertTokenizer, WEIGHTS_NAME, XLMRobertaTokenizer,
    AutoTokenizer, AutoConfig)
# Tensorboard
try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

MODEL_CLASSES = {
    "bert-retrieval": (BertConfig, BertForRetrieval, BertTokenizer),
    "xlmr-retrieval": (XLMRobertaConfig, XLMRobertaForRetrieval, XLMRobertaTokenizer),
    "sbert-retrieval": (AutoConfig, SBERTForRetrieval, AutoTokenizer)
}

SPLIT_NAMES = ["train", "valid", "test"]
DATASET_TO_DIR = {"xquad": "xquad-r",
                  "mlqa": "mlqa-r"}

EMBEDDER = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def create_a_sym_search_parser(subparser):
    parser = subparser.add_parser("asymsearch", help="Evaluate on Asymmetric Search")
    get_train_params(parser)
    get_path_options(parser)
    get_adam_optim_params(parser)
    get_base_model_options(parser)
    get_language_options(parser)
    get_meta_task_options(parser)
    get_translate_train_params(parser)
    parser.set_defaults(func=run_main)

def load_trans_questions_candidates(args, meta_learn_split_config, meta_task_dir, tokenizer, base_model):
    translation_path = os.path.join(args.data_root, "Translations/XQUAD/translate.pseudo-test/")
    squad_dir = os.path.join(args.data_root, "lareqa/Raw/xquad-r/cross_val/", str(args.cross_val_split))

    question_set = get_split_dict(SPLIT_NAMES)
    candidate_set = get_split_dict(SPLIT_NAMES)

    def read_squad_load_questions(languages, root_dir, file_name):
        squad_per_lang = {lang: {} for lang in languages}
        for lang in languages:
            with open(os.path.join(root_dir, file_name+lang+".json"), "r") as f:
                squad_per_lang[lang] = json.load(f)
            logger.info("Loaded %s" % lang)
        
        return utils_lareqa.load_data(squad_per_lang, base_model, tokenizer, args.device, meta_learn_split_config)

    ## Loading train question/candidate sets from translate-train
    languages = args.translate_train_langs.split(",")
    question_set["train"], candidate_set["train"] = read_squad_load_questions(languages, \
                                                                              translation_path, \
                                                                              "xquad.translate.pseudo-test.en-")

    ## Loading test question/candidate sets for evaluation purposes
    languages = set(args.languages.split(","))
    for split_name in ["valid", "test"]:
        question_set[split_name], candidate_set[split_name] = read_squad_load_questions(languages, \
                                                                                        os.path.join(squad_dir, split_name), \
                                                                                        "")

        with open(os.path.join(meta_task_dir , split_name+"_question_set.pickle"), "wb") as file:
            pickle.dump(question_set[split_name], file)

        with open(os.path.join(meta_task_dir, split_name+"_candidate_set.pickle"), "wb") as file:
            pickle.dump(candidate_set[split_name], file)

    meta_dataset = {split:{} for split in question_set}
    for split_name in question_set:
        top_results = get_all_embeddings_similarities(question_set[split_name])

        # Create meta-dataset tasks for that split
        logger.info("------>Constructing the meta-dataset for {} number of tasks {} .... ".format(split_name,
                                                                                                  meta_learn_split_config[split_name]["n_tasks_total"]))
        meta_dataset[split_name] = MetaDataset(meta_learn_split_config[split_name]["n_tasks_total"],
                                               meta_learn_split_config[split_name]["lang_pairs"],
                                               question_set[split_name],
                                               candidate_set[split_name],
                                               split_name,
                                               meta_learn_split_config,
                                               tokenizer,
                                               top_results)
        
        with open(os.path.join(meta_task_dir, split_name+"_meta_dataset.pickle"), "wb") as file:
            pickle.dump(meta_dataset[split_name], file)


    meta_tasks = {split_name: meta_dataset[split_name].meta_tasks for split_name in SPLIT_NAMES}

    return question_set, candidate_set, meta_dataset, meta_tasks
        
def create_meta_dataset_lareqa(args, meta_learn_split_config, meta_tasks_dir, tokenizer, embedder=None):
    print("CREATE META DATASET LAREQA WITHOUT TRANSLATION")
    print("SOMETHING TO TEST HERE")
    if args.use_triplet_loss:
        args.n_neg_eg = 1

    meta_dataset = get_split_dict(SPLIT_NAMES)
    question_set = get_split_dict(SPLIT_NAMES)
    candidate_set = get_split_dict(SPLIT_NAMES)

    question_file_names = [os.path.join(meta_tasks_dir, split_name+"_question_set.pickle") for split_name in SPLIT_NAMES]
    if all([os.path.isfile(question_file_names[i]) for i in range(len(SPLIT_NAMES))]):
        print("LOADING PRE-EXISTING", meta_tasks_dir)
        for split_name in SPLIT_NAMES:
            with open(os.path.join(meta_tasks_dir, split_name+"_question_set.pickle"), "rb") as file:
                question_set[split_name] = pickle.load(file)

            with open(os.path.join(meta_tasks_dir, split_name+"_candidate_set.pickle"), "rb") as file:
                candidate_set[split_name] = pickle.load(file)
    else:
        print("LOADING FROM SCRATCH")
        if args.use_cross_val:
            squad_dir = os.path.join(args.data_root, 'lareqa', 'Raw', DATASET_TO_DIR["xquad"], "cross_val", args.cross_val_split)
        else:
            squad_dir = os.path.join(args.data_root, 'lareqa', 'Raw', DATASET_TO_DIR["xquad"], "splits")

        meta_dataset = {split:{} for split in SPLIT_NAMES}
        for split_name in SPLIT_NAMES:
            # Load splits for each language.
            logger.info("Loading the dataset for {} .... ".format(split_name))
            if split_name == "pre_finetune":
                languages = set(args.pre_finetune_language.split(","))
                split_lang_pairs =  "|".join([lang+"_"+lang+"-"+lang+"_"+lang for lang in languages])
            else:
                languages = set(args.languages.split(","))
                split_lang_pairs = meta_learn_split_config[split_name]["lang_pairs"]

            squad_per_lang = {}
            for language in languages:
                with open(os.path.join(squad_dir, split_name, language+".json"), "r") as f:
                    squad_per_lang[language] = json.load(f)
                logger.info("Loaded %s" % language)

            # Load the question set and candidate set.
            question_set[split_name], candidate_set[split_name] = utils_lareqa.load_data(squad_per_lang, embedder, tokenizer, args.device, meta_learn_split_config)

    print("Reading for split_names:", SPLIT_NAMES)

    meta_file_names = [os.path.join(meta_tasks_dir, split_name+"_meta_dataset.pickle") for split_name in SPLIT_NAMES]

    if not args.update_meta_data and all([os.path.isfile(meta_file_names[i]) for i in range(len(SPLIT_NAMES))]):
        print("Metadatasets exit so will load them ONLYYY ", meta_tasks_dir)
        for split_name in SPLIT_NAMES:
            with open(os.path.join(meta_tasks_dir, split_name+"_meta_dataset.pickle"), "rb") as file:
                meta_dataset[split_name] = pickle.load(file)
    else:
        print("META DATASETS DON'T EXIST SO WILL BE CREATED FROM SCRATCH ", meta_tasks_dir)
        for split_name in SPLIT_NAMES:

            top_results = get_all_embeddings_similarities(question_set[split_name])

            # Create meta-dataset tasks for that split
            logger.info("------>Constructing the meta-dataset for {} number of tasks {} .... ".format(split_name,
                                                                                                      meta_learn_split_config[split_name]["n_tasks_total"]))
            meta_dataset[split_name] = MetaDataset(meta_learn_split_config[split_name]["n_tasks_total"],
                                                   split_lang_pairs,
                                                   question_set[split_name],
                                                   candidate_set[split_name],
                                                   split_name,
                                                   meta_learn_split_config,
                                                   tokenizer,
                                                   top_results)

            with open(os.path.join(meta_tasks_dir, split_name+"_meta_dataset.pickle"), "wb") as file:
                pickle.dump(meta_dataset[split_name], file)

            with open(os.path.join(meta_tasks_dir, split_name+"_question_set.pickle"), "wb") as file:
                pickle.dump(question_set[split_name], file)

            with open(os.path.join(meta_tasks_dir, split_name+"_candidate_set.pickle"), "wb") as file:
                pickle.dump(candidate_set[split_name], file)

    meta_tasks = {split_name: meta_dataset[split_name].meta_tasks for split_name in SPLIT_NAMES}

    logger.info({split_name: len(meta_tasks[split_name]) for split_name in SPLIT_NAMES})

    return question_set, candidate_set, meta_dataset, meta_tasks

def multilingual_zero_shot_evaluation(question_set, candidate_set, query_languages, tokenizer, base_model,  meta_learn_split_config, args, split_name):
    # Get all candidates in answer_languages and convert them to features
    print("candidates: ", {lang: len(cand) for lang, cand in candidate_set[split_name].by_lang.items()})
    use_base_model = True
    if use_base_model:
        c_features = [tokenizer.encode_plus((candidate.sentence+candidate.context).replace("\n", ""),
                                            max_length=meta_learn_split_config["max_answer_length"],
                                            pad_to_max_length=True,
                                            return_token_type_ids=True) for candidate in candidate_set[split_name].as_list()]

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

            all_candidate_vecs_list.append(candidate_encoding)

        all_candidate_vecs = np.concatenate(all_candidate_vecs_list, axis=0)

        print(np.sum(all_candidate_vecs))
    else:
        all_candidate_vecs = np.concatenate([np.expand_dims(candidate.encoding, 0) for candidate in candidate_set[split_name].as_list()], axis=0)
        print(np.sum(all_candidate_vecs))


    map_scores_lang = {lang: 0.0 for lang in query_languages}
    for query_lang in tqdm(query_languages):
        # For each query language, we compute map for each query in all answer languages
        print("Computing map for language %s ..."%query_lang)
        map_scores = []
        for question in tqdm(question_set[split_name].by_lang[query_lang]):
            # Get the encoding of the question using the base model
            if use_base_model:
                base_model = base_model.to(args.device)
                q_features = tokenizer.encode_plus(question.question,
                                                   max_length=meta_learn_split_config["max_query_length"],
                                                   pad_to_max_length=True,
                                                   return_token_type_ids=True)

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
                question_encoding = question.encoding
                question_encoding = np.expand_dims(question_encoding, 0)

            scores = question_encoding.dot(all_candidate_vecs.T)

            y_true = np.zeros(scores.shape[1])
            all_correct_cands = set(candidate_set[split_name].by_xling_id[question.xling_id])
            for ans in all_correct_cands:
                y_true[candidate_set[split_name].pos[ans]] = 1

            map_scores.append(utils_lareqa.average_precision_at_k(np.where(y_true == 1)[0], np.squeeze(scores).argsort()[::-1]))
        map_scores_lang[query_lang] = np.mean(map_scores)
    return map_scores_lang

def train_validate(args, meta_learn_split_config, meta_tasks, meta_tasks_dir, question_set, candidate_set, tokenizer, base_model, checkpoints_dir, writer):
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

    if args.do_pre_finetune: 
        print("DO PRE-FINE-TUNING")
        if True: # Always loading SQUAD EN Model instead of pre-fine-tuning from scratch in this case
            model_load_file = args.load_pre_finetune_path + "pytorch_model.bin"
            optim_load_file = args.load_pre_finetune_path + "optimizer.pt"

            logger.info("Finished Loading SQUAD torch model")
            opt, base_model = load_torch_model(args, opt, base_model, optim_load_file, model_load_file)

            logger.info("Finished Loading SQUAD torch model")
            print("Finished LOADING PREFINETUNING MODEL")

            test_multilingual_evaluation = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
            print("map_scores_lang:", test_multilingual_evaluation, " MEAN:", np.mean([test_multilingual_evaluation[lang] for lang in test_multilingual_evaluation if lang!= "en"]))
        else:
            # OR PRE-FINE-TUNE FROM SCRATCH
            split_name = "pre_finetune"
            n_tasks_batch =  args.n_prefinetune_tasks_batch
            pbar = tqdm(range(args.num_prefinetune_epochs))
            pbar.set_description("Prefinetuning Epoch Progress")
            for ep in pbar:
                pbar = tqdm(range(0, len(meta_tasks[split_name])//n_tasks_batch, n_tasks_batch))
                pbar.set_description(" --- Batch Progress")
                for batch_step in pbar: # number of train batches
                    meta_tasks_batch = meta_tasks[split_name][batch_step: batch_step+args.n_train_tasks_batch] # list of train MetaTasks
                    for j in range(len(meta_tasks_batch)):
                        opt.zero_grad()
                        qry_set = {k:v.to(args.device) for k, v in meta_tasks_batch[j].qry[0]}

                        qry_outputs = base_model(**qry_set)

                        loss_qry, q_encodings_qry, a_encodings_qry, n_encodings_qry = qry_outputs

                        loss_qry_avg_batch += loss_qry

                        map_at_20_qry = utils_lareqa.mean_avg_prec_at_k_meta([meta_tasks_batch[j].qry[q_n].question_cluster for q_n in range(len(meta_tasks_batch[j].qry))], # question_list
                                                                            q_encodings_qry,
                                                                            [meta_tasks_batch[j].qry[q_n].all_candidates for q_n in range(len(meta_tasks_batch[j].qry))],
                                                                            np.concatenate((a_encodings_qry, n_encodings_qry), axis=0),
                                                                            k=20)

                        writer.add_scalar(split_name+"_loss", loss_qry, ep * (len(meta_tasks[split_name])//n_tasks_batch) + (batch_step+j))
                        writer.add_scalar(split_name+"_map_at_20", map_at_20_qry, ep * (len(meta_tasks[split_name])//n_tasks_batch) + (batch_step+j))

                        loss_qry.backward()
                        opt.step()
                        base_model.zero_grad()

            logger.info("Zero-shot Multilingual Evaluation from that model")
            map_scores_lang = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
            print("map_scores_lang:", map_scores_lang, " MEAN:", np.mean([map_scores_lang[lang] for lang in map_scores_lang if lang!= "en"]))

    if args.use_meta_learn:
        Model = importlib.import_module('multi_meta_ssd.models.upstream.' + args.meta_learn_alg)
        meta_learner = Model.MetaLearner(base_model,
                                         args.device,
                                         meta_learn_split_config,
                                         opt)

    train_map_scores_lang = []
    valid_map_scores_lang = []
    test_map_scores_lang = []

    loss_qry_all_total, map_qry_all_total, loss_qry_avg_batch_total = split_ep_dict(args, SPLIT_NAMES) , \
                                                                      split_ep_dict(args, SPLIT_NAMES), \
                                                                      split_ep_dict(args, SPLIT_NAMES)

    for ep in tqdm(range(args.num_train_epochs)):
        for split_name in SPLIT_NAMES:
            if split_name != "test":
                random.shuffle(meta_tasks[split_name])
                if args.use_meta_learn:
                    logger.info("Meta %s"%split_name)
                    n_tasks_batch = meta_learner.meta_learn_config[split_name]["n_tasks_batch"]
                else:
                    logger.info("Fine-tuning %s"%split_name)
                    if split_name == "train":
                        n_tasks_batch = args.n_train_tasks_batch
                    else:
                        n_tasks_batch = args.n_valid_tasks_batch

                if args.use_triplet_loss:
                    n_triplets = args.triplet_batch_size
                else:
                    n_triplets = 1

                print("args.n_neg_eg:", args.n_neg_eg)
                print("meta_tasks[split_name]:", {k:v.shape for k, v in meta_tasks[split_name][0].spt_features.items()})

                for batch_step in tqdm(range(0, len(meta_tasks[split_name])//(n_tasks_batch*n_triplets), n_tasks_batch*n_triplets)): # number of train batches TODO change this
                    meta_tasks_batch = meta_tasks[split_name][batch_step: batch_step+(n_tasks_batch*n_triplets)]
                    if args.use_triplet_loss:
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
                            concatenated_spt_features_triplets_all = concatenated_spt_features_triplets_extend_l

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

                    if args.use_meta_learn:
                        loss_qry_avg_batch, loss_qry_all, map_qry_all = meta_learner(split_name, meta_tasks_batches, ep, batch_step, writer)

                        # print("AFTER Meta-Learning ...")
                        # for n, p in meta_learner.base_model.named_parameters():
                        #     if n == "trans_model.encoder.layer.0.attention.self.query.weight": # output.LayerNorm.weight intermediate.dense.bias
                        #         print("p:", p)

                        # for n, p in meta_learner.maml.named_parameters():
                        #     print(n)
                        #     if n == "module.trans_model.encoder.layer.0.attention.self.query.weight": # output.LayerNorm.weight intermediate.dense.bias
                        #         print("p:", p)
                        test_multilingual_evaluation = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
                        print("test_multilingual_evaluation:", test_multilingual_evaluation, " MEAN:", np.mean([test_multilingual_evaluation[lang] for lang in test_multilingual_evaluation if lang!= "en"]))
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        base_model = base_model.to(args.device)
                        loss_qry_avg_batch, loss_qry_all, map_qry_all = 0.0, [], []
                        for j in range(len(meta_tasks_batches)):
                            opt.zero_grad()
                            loss_spt_avg = 0.0
                            map_at_1_spt_avg = 0.0
                            for s_n in range(len(meta_tasks_batches[j]["spt_features"])):
                                spt_set = {k:v.to(args.device) for k, v in meta_tasks_batches[j]["spt_features"][s_n].items()}
                                spt_outputs = base_model(**spt_set)
                                loss_spt, q_encodings_spt, a_encodings_spt, n_encodings_spt = spt_outputs

                                loss_spt.backward()
                                opt.step()
                                base_model.zero_grad()

                                map_at_1_spt = utils_lareqa.mean_avg_prec_at_k_meta(meta_tasks_batches[j]["spt_questions"][s_n], # question_list
                                                                                    q_encodings_spt,
                                                                                    meta_tasks_batches[j]["spt_candidates"][s_n],
                                                                                    np.concatenate((a_encodings_spt, n_encodings_spt), axis=0),
                                                                                    k=1)

                                logger.info("ep: {},  batch: {}, sn: {}, loss_spt: {}, map_at_20_spt: {}".format(ep, j, s_n, loss_spt, map_at_1_spt))
                                loss_spt_avg += loss_spt
                                map_at_1_spt_avg += map_at_1_spt

                                del spt_set

                            writer.add_scalar(split_name+"_loss_spt", loss_spt_avg/len(meta_tasks_batches[j]["spt_features"]), ep * (len(meta_tasks[split_name])) + (batch_step+j))
                            writer.add_scalar(split_name+"_map_at_1", map_at_1_spt_avg/len(meta_tasks_batches[j]["spt_features"]), ep * (len(meta_tasks[split_name])) + (batch_step+j))

                            q_encodings_qry_all = []
                            loss_qry_avg = 0.0
                            map_at_1_qry_avg = 0.0
                            for q_n in tqdm(range(min(len(meta_tasks_batches[j]["qry_features"]), 4))):
                                qry_set = {k:v.to(args.device) for k, v in meta_tasks_batches[j]["qry_features"][q_n].items()}
                                qry_outputs = base_model(**qry_set)
                                loss_qry, q_encodings_qry, a_encodings_qry, n_encodings_qry = qry_outputs
                                q_encodings_qry_all.append(q_encodings_qry)

                                loss_qry.backward()
                                opt.step()
                                base_model.zero_grad()


                                del qry_set
                                loss_qry_avg_batch += loss_qry
                                map_at_1_qry = utils_lareqa.mean_avg_prec_at_k_meta(meta_tasks_batches[j]["qry_questions"][q_n], # question_list
                                                                                    q_encodings_qry,
                                                                                    meta_tasks_batches[j]["qry_candidates"][q_n],
                                                                                    np.concatenate((a_encodings_qry, n_encodings_qry), axis=0),
                                                                                    k=1)

                                loss_qry_avg += loss_qry
                                map_at_1_qry_avg += map_at_1_qry

                                logger.info("ep: {}, batch: {}, loss_qry: {}, map_at_20_qry: {}".format(ep, j, loss_qry, map_at_1_qry))

                            writer.add_scalar(split_name+"_loss_qry", loss_qry/len(meta_tasks_batches[j]["qry_features"]), ep * (len(meta_tasks[split_name])) + (batch_step+j))
                            writer.add_scalar(split_name+"_map_at_1", map_at_1_qry_avg/len(meta_tasks_batches[j]["qry_features"]), ep * (len(meta_tasks[split_name])) + (batch_step+j))
                            loss_qry_all.append(loss_qry)
                            map_qry_all.append(map_at_1_qry_avg/len(meta_tasks_batches[j]["qry_features"]))

                            del spt_outputs
                            del qry_outputs

                        map_scores_lang = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
                        print("map_scores_lang:", map_scores_lang, " MEAN:", np.mean([map_scores_lang[lang] for lang in map_scores_lang if lang!= "en"]))
                        loss_qry_avg_batch = loss_qry_avg_batch / len(meta_tasks_batch)
                        writer.add_scalar(split_name+"_avg_batch_loss", loss_qry_avg_batch, ep * (len(meta_tasks[split_name])//n_tasks_batch) + (batch_step+j))
                    loss_qry_avg_batch_total[split_name][ep].append(loss_qry_avg_batch)
                    loss_qry_all_total[split_name][ep].append(loss_qry_all)
                    map_qry_all_total[split_name][ep].append(map_qry_all)

                    if args.use_meta_learn:
                        save_torch_model(args, meta_learner.base_model, meta_learner.opt, checkpoints_dir, split_name+str(ep))
                        save_torch_model(args, meta_learner.maml, meta_learner.opt, checkpoints_dir, "maml_"+split_name+str(ep))
                        save_torch_model(args, meta_learner.learner, meta_learner.opt, checkpoints_dir, "learner_"+split_name+str(ep))
                    else:
                        save_torch_model(args, base_model, opt, checkpoints_dir, split_name+str(ep))
                # TODO Add checkpointing for train using tensorboard

            train_multilingual_evaluation = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "train")
            print("train_multilingual_evaluation:", train_multilingual_evaluation, " MEAN:", np.mean([train_multilingual_evaluation[lang] for lang in train_multilingual_evaluation if lang!= "en"]))
            train_map_scores_lang.append(train_multilingual_evaluation)
            valid_multilingual_evaluation = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "valid")
            print("valid_multilingual_evaluation:", valid_multilingual_evaluation, " MEAN:", np.mean([valid_multilingual_evaluation[lang] for lang in valid_multilingual_evaluation if lang!= "en"]))
            valid_map_scores_lang.append(valid_multilingual_evaluation)
            test_multilingual_evaluation = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
            print("test_multilingual_evaluation:", test_multilingual_evaluation, " MEAN:", np.mean([test_multilingual_evaluation[lang] for lang in test_multilingual_evaluation if lang!= "en"]))
            test_map_scores_lang.append(test_multilingual_evaluation)

        if args.update_meta_data:
            meta_dataset, meta_tasks, question_set, candidate_set = create_meta_dataset_lareqa(args,
                                                                                               meta_learn_split_config,
                                                                                               meta_tasks_dir,
                                                                                               tokenizer,
                                                                                               base_model)
    ## Save FINAL MODEL
    if args.use_meta_learn:
        save_torch_model(args, meta_learner.base_model, meta_learner.opt, checkpoints_dir, "final")
        save_torch_model(args, meta_learner.maml, meta_learner.opt, checkpoints_dir, "final_maml")
        save_torch_model(args, meta_learner.learner, meta_learner.opt, checkpoints_dir, "final_learner")
    else:
        save_torch_model(args, base_model, opt, checkpoints_dir, "final")

    return {"loss_qry_avg_batch_total": loss_qry_avg_batch_total,\
            "loss_qry_all_total": loss_qry_all_total, \
            "map_qry_all_total": map_qry_all_total, \
            "train_map_scores_lang": train_map_scores_lang, \
            "valid_map_scores_lang": valid_map_scores_lang, \
            "test_map_scores_lang": test_map_scores_lang}

def get_all_embeddings_similarities(question_set):
    languages = question_set.by_lang
    all_embeddings = {lang: [question.encoding for question in question_set.by_lang[lang]] for lang in languages}

    top_results_xling_ids = {lang1: {lang2: {} for lang2 in languages} for lang1 in languages}
    top_results_scores = {lang1: {lang2: {} for lang2 in languages} for lang1 in languages}
    xling_ids =  {lang: {idx: question.xling_id for idx, question in enumerate(question_set.by_lang[lang])} for lang in languages}
    for lang1 in languages:
        for lang2 in languages:
            top_k = min(5, len(all_embeddings[lang2]))
            query_embeddings = np.concatenate(all_embeddings[lang1], axis=0)
            corpus_embeddings = np.concatenate(all_embeddings[lang2], axis=0)
            cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)
            _top_results = [torch.topk(cos_scores[i], k=top_k) for i in range(len(cos_scores))]
            for i in range(len(cos_scores)):
                # matrix of scores
                top_results_xling_ids[lang1][lang2].update({xling_ids[lang1][i]: [xling_ids[lang2][idx.cpu().detach().numpy().item()] for idx in _top_results[i][1]]})
                top_results_scores[lang1][lang2].update({xling_ids[lang1][i]:  _top_results[i][0].cpu().detach().numpy()})

    return {"xling_ids": top_results_xling_ids, "scores": top_results_scores}

def run_main(args):
    print("Running main Asymmetric Semantic Search")
    SPLIT_NAMES = ["train", "valid", "test"]

    # Setting up the configuration parameters by reading them from their paths
    args, meta_learn_split_config = get_config_params(args, 'asym')
    logger.info(args)

    # Setting the seed and the device
    set_seed(args)
    set_device(args)

    # Setting the output directory
    meta_tasks_dir, checkpoints_dir, runs_dir, logs_dir, writer = set_out_dir(args, "asym")

    if args.do_pre_finetune:
        SPLIT_NAMES += ["pre_finetune"]
        logger.info(SPLIT_NAMES)

    # if args.no_debug:
    stdoutOrigin = sys.stdout
    sys.stdout = open(os.path.join(logs_dir, args.logs_file), "w")

    # Setting logstats and writing config to json file
    logger.info("Saving to logs_dir: {}".format(logs_dir))
    logstats_init(os.path.join(logs_dir, args.stats_file))

    config_path = os.path.join(logs_dir, 'config.json')
    logstats_add_args('config', args)
    args_var = {k: v for k, v in vars(args).items() if k not in ["func", "device"]}
    logstats_write_json(args_var, config_path)

    # Load config, tokenizer, and downstream model
    tokenizer, base_model = get_config_tokenizer_model(MODEL_CLASSES, args)

    # Create/load meta-dataset
    if args.translate_train:
        load_data_func = load_trans_questions_candidates
    else:
        load_data_func = create_meta_dataset_lareqa

    question_set, candidate_set, meta_dataset, meta_tasks = load_data_func(args,  
                                                                           meta_learn_split_config,
                                                                           meta_tasks_dir, 
                                                                           tokenizer,
                                                                           base_model)


    # Initial Multilingual Evaluation 
    test_multilingual_evaluation = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
    print("INITIAL map_scores_lang:", test_multilingual_evaluation, " MEAN:", np.mean([test_multilingual_evaluation[lang] for lang in test_multilingual_evaluation if lang!= "en"]))

    if args.do_evaluate:
        print("Evaluation mode")
        opt = torch.optim.Adam(base_model.parameters(),
                               lr=meta_learn_split_config["train"]["beta_lr"])

        optim_load_file = os.path.join(checkpoints_dir, "optimizer.pt")
        model_load_file =os.path.join(checkpoints_dir, "pytorch_model.bin")
        opt, base_model = load_torch_model(args, opt, base_model, optim_load_file, model_load_file)


        logger.info("Multilingual Evaluation for that model")
        map_scores_lang = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
        print("map_scores_lang:", map_scores_lang, " MEAN:", np.mean([map_scores_lang[lang] for lang in map_scores_lang if lang!= "en"]))
    else:
        print("Training mode")
        metrics = train_validate(args,
                                 meta_learn_split_config,
                                 meta_tasks,
                                 meta_tasks_dir,
                                 question_set,
                                 candidate_set,
                                 tokenizer,
                                 base_model,
                                 checkpoints_dir,
                                 writer)

        for n, v in metrics.items():
            with open(os.path.join(runs_dir, n+".pickle"), "wb") as file:
                pickle.dump(v, file)

    # if args.no_debug:
    sys.stdout.close()
    sys.stdout = stdoutOrigin
