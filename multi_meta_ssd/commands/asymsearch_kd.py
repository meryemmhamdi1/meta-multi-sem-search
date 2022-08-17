from pyparsing import str_type
from multi_meta_ssd.log import *
from multi_meta_ssd.commands.get_arguments import get_train_params, get_path_options, get_adam_optim_params, get_base_model_options, get_language_options, get_meta_task_options
from multi_meta_ssd.models.downstream.dual_encoders.bert import BertForRetrieval
from multi_meta_ssd.models.downstream.dual_encoders.sent_trans import SBERTForRetrieval
from multi_meta_ssd.models.downstream.dual_encoders.xlm_roberta import XLMRobertaConfig, XLMRobertaForRetrieval
from multi_meta_ssd.commands.train_evaluate import load_and_cache_examples, train, evaluate, set_device
from multi_meta_ssd.processors.downstream import utils_lareqa
from multi_meta_ssd.processors.upstream.meta_task_merged import MetaDataset
from multi_meta_ssd.models.upstream.maml import MetaLearner
from sentence_transformers import SentenceTransformer, util

from tqdm import tqdm
import numpy as np
import os, json, random, torch, pickle, importlib, sys, configparser, gc
from transformers import (
    AdamW, BertConfig, BertTokenizer, WEIGHTS_NAME, XLMRobertaTokenizer,
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup)
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

split_names = ["train", "valid", "test"]
embedder = None #SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def get_config_params(args):
    paths = configparser.ConfigParser()
    paths.read('multi_meta_ssd/config/paths.ini')

    location = "SENSEI"
    # location = "LOCAL"

    args.data_root = str(paths.get(location, "DATA_ROOT"))
    args.train_file = str(paths.get(location, "TRAIN_FILE"))
    args.predict_file = str(paths.get(location, "PREDICT_FILE"))
    args.out_dir = str(paths.get(location, "OUT_DIR"))

    params = configparser.ConfigParser()
    params.read('multi_meta_ssd/config/down_model_param.ini')
    
    args.pre_finetune_language = str(params.get("PARAMS", "PRE_FINETUNE_LANGUAGE"))

    args.max_seq_length = int(params.get("PARAMS", "MAX_SEQ_LENGTH"))
    args.max_query_length = int(params.get("PARAMS", "MAX_QUERY_LENGTH"))
    args.max_answer_length = int(params.get("PARAMS", "MAX_ANSWER_LENGTH"))

    # TODO FIX THOSE AS CONFIG
    # args.adam_eps = float(params.get("HYPER", "ADAM_EPS"))
    # args.beta_1 = float(params.get("HYPER", "BETA_1"))
    # args.beta_2 = float(params.get("HYPER", "BETA_2"))
    # args.epsilon = float(params.get("HYPER", "EPSILON"))
    # args.step_size = float(params.get("HYPER", "STEP_SIZE"))
    # args.gamma = float(params.get("HYPER", "GAMMA"))
    # args.test_steps = int(params.get("HYPER", "TEST_STEPS"))
    # args.num_intent_tasks = int(params.get("HYPER", "NUM_INTENT_TASKS"))  # only in case of CILIA setup
    # args.num_lang_tasks = int(params.get("HYPER", "NUM_LANG_TASKS"))  # only in case of CILIA setup
    # args.test_steps = int(params.get("HYPER", "TEST_STEPS"))

    params = configparser.ConfigParser()
    params.read('multi_meta_ssd/config/meta_task_param.ini')

    args.train_lang_pairs = str(params.get(args.mode_transfer, "TRAIN_LANG_PAIRS"))
    args.valid_lang_pairs = str(params.get(args.mode_transfer, "VALID_LANG_PAIRS"))
    args.test_lang_pairs = str(params.get(args.mode_transfer, "TEST_LANG_PAIRS"))

    return args

def set_out_dir(args):
    out_dir = os.path.join(args.out_dir,
                           args.model_type,
                           args.meta_learn_alg if args.use_meta_learn else "finetune",
                           args.mode_transfer)

    if args.do_pre_finetune:
        out_dir = os.path.join(out_dir, "PreFineTune")

    if args.use_triplet_loss:
        out_dir = os.path.join(out_dir, "TripletLoss")

    if args.use_cross_val:
        out_dir = os.path.join(out_dir, "CrossVal_"+args.cross_val_split)

    if args.mode_qry == "random":
        out_dir = os.path.join(out_dir, "random")

    if args.neg_sampling_approach != "random":
        out_dir = os.path.join(out_dir, "neg_sampling_"+args.neg_sampling_approach)

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
                                  "meta_tasks",
                                  "TripletLoss" if args.use_triplet_loss else "NegEg"+str(args.n_neg_eg),
                                  args.mode_transfer,
                                  args.mode_qry if args.mode_qry == "random" else args.sim_model_name if args.use_sim_embedder else "NoSIM",
                                  ",".join(sorted(set(args.languages.split(","))))) # different path for meta_tasks


    if args.use_cross_val:
        meta_tasks_dir = os.path.join(meta_tasks_dir, "CrossVal_"+args.cross_val_split)

    if args.neg_sampling_approach != "random":
        meta_tasks_dir = os.path.join(meta_tasks_dir, "neg_sampling_"+args.neg_sampling_approach)

    print("Reading from meta_tasks_dir:", meta_tasks_dir)

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

def create_a_sym_search_kd_parser(subparser):
    parser = subparser.add_parser("asymsearch_kd", help="Evaluate on Asymmetric Search")
    get_train_params(parser)
    get_path_options(parser)
    get_adam_optim_params(parser)
    get_base_model_options(parser)
    get_language_options(parser)
    get_meta_task_options(parser)
    parser.set_defaults(func=run_main)

def create_meta_dataset_lareqa(args, meta_learn_split_config, meta_tasks_dir, tokenizer, embedder=None):
    if args.use_triplet_loss:
        args.n_neg_eg = 1

    meta_dataset = {split:{} for split in split_names}
    question_set = {split:{} for split in split_names}
    candidate_set = {split:{} for split in split_names}

    question_file_names = [os.path.join(meta_tasks_dir, split_name+"_question_set.pickle") for split_name in split_names]
    if all([os.path.isfile(question_file_names[i]) for i in range(len(split_names))]):
        for split_name in split_names:
            with open(os.path.join(meta_tasks_dir, split_name+"_question_set.pickle"), "rb") as file:
                question_set[split_name] = pickle.load(file)

            with open(os.path.join(meta_tasks_dir, split_name+"_candidate_set.pickle"), "rb") as file:
                candidate_set[split_name] = pickle.load(file)
    else:
        dataset_to_dir = {
        "xquad": "xquad-r",
        "mlqa": "mlqa-r"
        }

        if args.use_cross_val:
            squad_dir = os.path.join(args.data_root, 'lareqa', dataset_to_dir["xquad"], "cross_val", args.cross_val_split)
        else:
            squad_dir = os.path.join(args.data_root, 'lareqa', dataset_to_dir["xquad"], "splits")

        meta_dataset = {split:{} for split in split_names}
        for split_name in split_names:
            # Load splits for each language.
            logger.info("Loading the dataset for {} .... ".format(split_name))
            if split_name == "pre_finetune":
                languages = set(args.pre_finetune_language.split(","))
                split = "train"
                split_lang_pairs =  "|".join([lang+"_"+lang+"-"+lang+"_"+lang for lang in languages])
            else:
                languages = set(args.languages.split(","))
                split = split_name
                split_lang_pairs = meta_learn_split_config[split]["lang_pairs"]

            squad_per_lang = {}
            for language in languages:
                with open(os.path.join(squad_dir, split, language+".json"), "r") as f:
                    squad_per_lang[language] = json.load(f)
                logger.info("Loaded %s" % language)

            # Load the question set and candidate set.
            question_set[split_name], candidate_set[split_name] = utils_lareqa.load_data(squad_per_lang, embedder, tokenizer, args.device, meta_learn_split_config)

    print("Reading for split_names:", split_names)

    meta_file_names = [os.path.join(meta_tasks_dir, split_name+"_merged_meta_dataset.pickle") for split_name in split_names]

    if not args.update_meta_data and all([os.path.isfile(meta_file_names[i]) for i in range(len(split_names))]):
        for split_name in split_names:
            with open(os.path.join(meta_tasks_dir, split_name+"_merged_meta_dataset.pickle"), "rb") as file:
                meta_dataset[split_name] = pickle.load(file)
    else:
        logger.info("META DATASETS DON'T EXIST SO WILL BE CREATED FROM SCRATCH")
        for split_name in split_names:

            top_results = get_all_embeddings_similarities(question_set[split_name])

            # Create meta-dataset tasks for that split
            logger.info("------>Constructing the meta-dataset for {} number of tasks {} .... ".format(split_name,
                                                                                                      meta_learn_split_config[split_name]["n_tasks_total"]))
            meta_dataset[split_name] = MetaDataset(meta_learn_split_config[split]["n_tasks_total"],
                                                   split_lang_pairs,
                                                   question_set[split_name],
                                                   candidate_set[split_name],
                                                   split_name,
                                                   meta_learn_split_config,
                                                   tokenizer,
                                                   top_results)

            with open(os.path.join(meta_tasks_dir, split_name+"_merged_meta_dataset.pickle"), "wb") as file:
                pickle.dump(meta_dataset[split_name], file)

            with open(os.path.join(meta_tasks_dir, split_name+"_question_set.pickle"), "wb") as file:
                pickle.dump(question_set[split_name], file)

            with open(os.path.join(meta_tasks_dir, split_name+"_candidate_set.pickle"), "wb") as file:
                pickle.dump(candidate_set[split_name], file)

    meta_tasks = {split_name: meta_dataset[split_name].meta_tasks for split_name in split_names}

    logger.info({split_name: len(meta_tasks[split_name]) for split_name in split_names})

    return meta_dataset, meta_tasks, question_set, candidate_set

def multilingual_zero_shot_evaluation(question_set, candidate_set, query_languages, tokenizer, base_model,  meta_learn_split_config, args, split_name):
    # Get all candidates in answer_languages and convert them to features
    print("candidates: ", {lang: len(cand) for lang, cand in candidate_set[split_name].by_lang.items()})
    use_base_model = True
    candidates_list = []
    all_uuids = []
    for candidate_language in candidate_set[split_name].by_lang:
        print("candidate_language:", candidate_language, "candidate_set[split_name].by_lang[candidate_language]:", len(candidate_set[split_name].by_lang[candidate_language]))
        candidates_list.extend(candidate_set[split_name].by_lang[candidate_language])
        all_uuids.extend([candidate.uid for candidate in candidate_set[split_name].by_lang[candidate_language]])
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
        # print("all_candidate_vecs sum:", np.sum(all_candidate_vecs))

        # print("candidate.encoding.shape:", np.squeeze(candidate_set[split_name].as_list()[0].encoding).shape)

        # print("ENCODER all_candidate_vecs.shape:", all_candidate_vecs.shape)
        print(np.sum(all_candidate_vecs))
    else:
        all_candidate_vecs = np.concatenate([np.expand_dims(candidate.encoding, 0) for candidate in candidate_set[split_name].as_list()], axis=0)
        # print("all_candidate_vecs.shape:", all_candidate_vecs.shape)
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
                question_encoding = question.encoding
                question_encoding = np.expand_dims(question_encoding, 0)

            scores = question_encoding.dot(all_candidate_vecs.T)

            # print("scores.shape:", scores.shape)
            y_true = np.zeros(scores.shape[1])
            # print("y_true.shape:", y_true.shape)
            all_correct_cands = set(candidate_set[split_name].by_xling_id[question.xling_id])
            # for ans in all_correct_cands:
            #     # print("pos:", candidate_set[split_name].pos[ans])
            #     y_true[candidate_set[split_name].pos[ans]] = 1

            all_correct_uuids = [candidate.uid for candidate in all_correct_cands]
            other_positions = []
            for c_idx, uuid in enumerate(all_uuids):
                if uuid in all_correct_uuids:
                    other_positions.append(c_idx)
                    y_true[c_idx] = 1

            map_scores.append(utils_lareqa.average_precision_at_k(np.where(y_true == 1)[0], np.squeeze(scores).argsort()[::-1]))
        map_scores_lang[query_lang] = np.mean(map_scores)
    return map_scores_lang

def train_validate(args, meta_learn_split_config, meta_tasks, meta_tasks_dir, question_set, candidate_set, tokenizer, base_model, loss_qry_avg_batch_total, loss_qry_all_total, map_qry_all_total, checkpoints_dir, writer):
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
        if True:
            # EITHER LOAD THE MODEL
            # model_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs-temp/squad/bert-base-multilingual-cased_LR3e-5_EPOCH3.0_maxlen384_batchsize4_gradacc8/pytorch_model.bin"
            # optim_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs-temp/squad/bert-base-multilingual-cased_LR3e-5_EPOCH3.0_maxlen384_batchsize4_gradacc8/training_args.bin"
            model_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs/lareqa/bert-base-multilingual-cased_LR2e-5_EPOCH3.0_LEN352/checkpoint-9000/pytorch_model.bin"
            optim_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs/lareqa/bert-base-multilingual-cased_LR2e-5_EPOCH3.0_LEN352/checkpoint-9000/optimizer.pt"

            logger.info("Finished Loading SQUAD torch model")
            opt, base_model = load_torch_model(args, opt, base_model, optim_load_file, model_load_file)

            logger.info("Finished Loading SQUAD torch model")
            # logger.info("Zero-shot Multilingual Evaluation from that model")
            # map_scores_lang = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
            # print("map_scores_lang:", map_scores_lang, " MEAN:", np.mean([map_scores_lang[lang] for lang in map_scores_lang if lang!= "en"]))
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

    model_load_file = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/maml/MONO_BIL/TripletLoss/CrossVal_0/checkpoints/pytorch_model_0.bin"
    optim_load_file = "/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/sbert-retrieval/maml/MONO_BIL/TripletLoss/CrossVal_0/checkpoints/optimizer_0.pt"

    logger.info("Finished Loading SQUAD torch model")
    opt, base_model = load_torch_model(args, opt, base_model, optim_load_file, model_load_file)
    opt = optimizer_to(opt, args.device)
    if args.use_meta_learn:
        Model = importlib.import_module('multi_meta_ssd.models.upstream.' + args.meta_learn_alg)
        meta_learner = Model.MetaLearner(tokenizer,
                                        base_model,
                                        args.device,
                                        meta_learn_split_config,
                                        opt)


    train_map_scores_lang = []
    valid_map_scores_lang = []
    test_map_scores_lang = []
    for ep in tqdm(range(args.num_train_epochs)):
        for split_name in split_names:
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

                for batch_step in tqdm(range(0, len(meta_tasks[split_name])//(n_tasks_batch*n_triplets), n_tasks_batch*n_triplets)): # number of train batches TODO change this
                    meta_tasks_batch = meta_tasks[split_name][batch_step: batch_step+(n_tasks_batch*n_triplets)]
                    print("len(meta_tasks_batch):", len(meta_tasks_batch))
                    print("list:", list(range(batch_step, batch_step+(n_tasks_batch*n_triplets), n_triplets)))
                    if args.use_triplet_loss:
                        # Concatenate n_tasks_batch meta_tasks
                        meta_task_triplets = []
                        meta_tasks_batches = []
                        for k in range(0, n_tasks_batch*n_triplets, n_triplets):
                            meta_trip = meta_tasks_batch[k: k+n_triplets]
                            meta_task_triplets.append(meta_trip)
                            spt_features_triplets = []
                            qry_features_triplets = {rank: [] for rank in range(2)}
                            spt_questions = []
                            spt_candidates = []
                            qry_questions = {rank: [] for rank in range(2)}
                            qry_candidates = {rank: [] for rank in range(2)}
                            for k1 in range(n_triplets):
                                spt_features_triplets.append(meta_trip[k1].spt_features.items())

                                len_q = meta_trip[k1].spt_features["q_input_ids"].shape[0]
                                len_a = meta_trip[k1].spt_features["a_input_ids"].shape[0]
                                for s_n in range(len_a//len_q):
                                    spt_questions.append([meta_trip[k1].spt.question_cluster])
                                    spt_candidates.append([meta_trip[k1].spt.all_candidates])

                                for rank in range(2):
                                    qry_features_triplets_parts = []
                                    for qr_k in range(len(meta_trip[k1].qry_features[rank])):
                                        qry_features_triplets_parts.append(meta_trip[k1].qry_features[rank][qr_k].items())

                                    len_q = meta_trip[k1].qry_features[rank][qr_k]["q_input_ids"].shape[0]
                                    len_a = meta_trip[k1].qry_features[rank][qr_k]["a_input_ids"].shape[0]
                                    for q_n in range(len_a//len_q):
                                        qry_questions[rank].append([meta_trip[k1].qry[rank][q_n].question_cluster for q_n in range(len(meta_trip[k1].qry[rank]))])
                                        qry_candidates[rank].append([meta_trip[k1].qry[rank][q_n].all_candidates for q_n in range(len(meta_trip[k1].qry[rank]))])

                                    qry_features_triplets[rank].append(qry_features_triplets_parts)

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

                            concatenated_qry_features_triplets_all = {rank: [] for rank in range(2)}
                            for rank in range(2):
                                for qry_features in qry_features_triplets[rank]:
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
                                    concatenated_qry_features_triplets_all[rank].extend(concatenated_qry_features_triplets_extend_l)

                            meta_tasks_batches.append({"spt_features": concatenated_spt_features_triplets_all,
                                                       "qry_features": concatenated_qry_features_triplets_all,
                                                       "spt_questions": spt_questions,
                                                       "spt_candidates": spt_candidates,
                                                       "qry_questions": qry_questions,
                                                       "qry_candidates": qry_candidates})
                    else:
                        spt_questions = []
                        spt_candidates = []
                        qry_questions = {rank: [] for rank in range(2)}
                        qry_candidates = {rank: [] for rank in range(2)}
                        for k in range(len(meta_tasks_batch)):
                            spt_questions.append([meta_tasks_batch[k].spt.question_cluster])
                            spt_candidates.append([meta_tasks_batch[k].spt.all_candidates])
                            for rank in range(2):
                                qry_questions[rank].append([meta_tasks_batch[k].qry[rank][q_n].question_cluster for q_n in range(len(meta_tasks_batch[k].qry[rank]))])
                                qry_candidates[rank].append([meta_tasks_batch[k].qry[rank][q_n].all_candidates for q_n in range(len(meta_tasks_batch[k].qry[rank]))])


                        meta_tasks_batches = [{"spt_features": meta_tasks_batch[k2].spt_features.items(), 
                                               "qry_features": meta_tasks_batch[k2].qry_features,
                                               "spt_questions": spt_questions[k2], 
                                               "spt_candidates": spt_candidates, 
                                               "qry_questions": qry_questions[k2],
                                               "qry_candidates": qry_candidates} for k2 in range(len(meta_tasks_batch))]

                    if args.use_meta_learn:
                        loss_qry_avg_batch, loss_qry_all, map_qry_all = meta_learner(split_name, meta_tasks_batches, ep, batch_step, writer)
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
                            for q_n in range(len(meta_tasks_batches[j]["qry_features"])):
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
    else:
        save_torch_model(args, base_model, opt, checkpoints_dir, "final")

    return loss_qry_avg_batch_total, loss_qry_all_total, map_qry_all_total, train_map_scores_lang, valid_map_scores_lang, test_map_scores_lang
    # TODO ADD DIFFERENT MODES OF EVALUATION SEPARATELY

def train_validate_debug(args, meta_learn_split_config, meta_tasks, question_set, candidate_set, tokenizer, base_model, loss_qry_avg_batch_total, loss_qry_all_total, map_qry_all_total, checkpoints_dir, writer):
    # EITHER LOAD THE MODEL
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
    # model_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs-temp/squad/bert-base-multilingual-cased_LR3e-5_EPOCH3.0_maxlen384_batchsize4_gradacc8/pytorch_model.bin"
    # optim_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs-temp/squad/bert-base-multilingual-cased_LR3e-5_EPOCH3.0_maxlen384_batchsize4_gradacc8/training_args.bin"
    model_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs/lareqa/bert-base-multilingual-cased_LR2e-5_EPOCH3.0_LEN352/checkpoint-9000/pytorch_model.bin"
    optim_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs/lareqa/bert-base-multilingual-cased_LR2e-5_EPOCH3.0_LEN352/checkpoint-9000/optimizer.pt"

    logger.info("Finished Loading SQUAD torch model")
    opt, base_model = load_torch_model(args, opt, base_model, optim_load_file, model_load_file)

    opt = optimizer_to(opt, args.device)

    logger.info("Zero-shot Multilingual Evaluation from that model")
    map_scores_lang = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
    print("map_scores_lang:", map_scores_lang, " MEAN:", np.mean([map_scores_lang[lang] for lang in map_scores_lang if lang!= "en"]))

    split_name = "train"
    n_tasks_batch =  args.n_prefinetune_tasks_batch
    pbar = tqdm(range(args.num_prefinetune_epochs))
    pbar.set_description("Training Epoch Progress")
    base_model = base_model.to(args.device)
    for ep in pbar:
        pbar = tqdm(range(0, len(meta_tasks[split_name])//n_tasks_batch, n_tasks_batch))
        pbar.set_description(" --- Batch Progress")
        loss_qry_avg_batch = 0.0
        for batch_step in pbar: # number of train batches
            meta_tasks_batch = meta_tasks[split_name][batch_step: batch_step+args.n_train_tasks_batch] # list of train MetaTasks
            for j in range(len(meta_tasks_batch)):
                opt.zero_grad()
                qry_set = {k:v.to(args.device) for k, v in meta_tasks_batch[j].qry_features[0].items()}

                qry_outputs = base_model(**qry_set)

                loss_qry, q_encodings_qry, a_encodings_qry, n_encodings_qry = qry_outputs

                loss_qry_avg_batch += loss_qry

                # map_at_20_qry = utils_lareqa.mean_avg_prec_at_k_meta([meta_tasks_batch[j].qry[q_n].question_cluster for q_n in range(len(meta_tasks_batch[j].qry))], # question_list
                #                                                      q_encodings_qry,
                #                                                     [meta_tasks_batch[j].qry[q_n].all_candidates for q_n in range(len(meta_tasks_batch[j].qry))],
                #                                                     np.concatenate((a_encodings_qry, n_encodings_qry), axis=0),
                #                                                     k=20)

                # writer.add_scalar(split_name+"_map_at_20", map_at_20_qry, ep * (len(meta_tasks[split_name])//n_tasks_batch) + (batch_step+j))

                loss_qry.backward()
                opt.step()
                base_model.zero_grad()

                print("loss_qry:", loss_qry)

                writer.add_scalar(split_name+"_loss", loss_qry, ep * (len(meta_tasks[split_name])//n_tasks_batch) + (batch_step+j))

                map_scores_lang = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
                print("batch_step:", batch_step, "j:", j, " map_scores_lang:", map_scores_lang, " MEAN:", np.mean([map_scores_lang[lang] for lang in map_scores_lang if lang!= "en"]))

    logger.info("Zero-shot Multilingual Evaluation from that model")
    map_scores_lang = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
    print("map_scores_lang:", map_scores_lang, " MEAN:", np.mean([map_scores_lang[lang] for lang in map_scores_lang if lang!= "en"]))

def get_all_embeddings_similarities(question_set):
    languages = question_set.by_lang
    all_embeddings = {lang: [question.encoding for question in question_set.by_lang[lang]] \
        for lang in languages}

    top_results_xling_ids = {lang1: {lang2: {} for lang2 in languages} for lang1 in languages}
    top_results_scores = {lang1: {lang2: {} for lang2 in languages} for lang1 in languages}
    xling_ids =  {lang: {idx: question.xling_id for idx, question in enumerate(question_set.by_lang[lang])} for lang in languages}
    for lang1 in languages:
        for lang2 in languages:
            # print("Processing lang1:", lang1, "lang2:", lang2)
            top_k = min(5, len(all_embeddings[lang2]))
            query_embeddings = np.concatenate(all_embeddings[lang1], axis=0)
            corpus_embeddings = np.concatenate(all_embeddings[lang2], axis=0)
            print("query_embeddings:", query_embeddings.shape)
            print("corpus_embeddings:", corpus_embeddings.shape)
            cos_scores = util.cos_sim(query_embeddings, corpus_embeddings) #for i in range(len(query_embeddings))
            _top_results = [torch.topk(cos_scores[i], k=top_k) for i in range(len(cos_scores))]
            # print("Top results:", _top_results)
            # top_results[lang1][lang2] = _top_results
            for i in range(len(cos_scores)):
                # matrix of scores
                # top_results[lang1][lang2].append(_top_results[i][0])
                # matrix of indices -> mapped to uuids
                # print("_top_results[i][0]:", _top_results[i][0])
                # print("_top_results[i][1]:", _top_results[i][1])
                top_results_xling_ids[lang1][lang2].update({xling_ids[lang1][i]: [xling_ids[lang2][idx.cpu().detach().numpy().item()] for idx in _top_results[i][1]]})
                top_results_scores[lang1][lang2].update({xling_ids[lang1][i]:  _top_results[i][0].cpu().detach().numpy()})

            # for score, idx in zip(_top_results[0], _top_results[1]):
            #     # top_results[lang1][lang2].append()
            #     print("lang1:", lang1, " lang2:", lang2, corpus_embeddings[idx], "(Score: {:.4f})".format(score))

            # print("top_results_scores[lang1][lang2]:", top_results_scores[lang1][lang2])
            # print("top_results_xling_ids[lang1][lang2]:", top_results_xling_ids[lang1][lang2])

    return {"xling_ids": top_results_xling_ids, "scores": top_results_scores}

def run_main(args):
    split_names = ["train", "valid", "test"]
    args = get_config_params(args)
    logger.info(args)

    if args.use_triplet_loss:
        args.n_neg_eg = 1

    set_device(args)
    meta_tasks_dir, checkpoints_dir, runs_dir, logs_dir, writer = set_out_dir(args)

    if args.do_pre_finetune:
        split_names += ["pre_finetune"]
        logger.info(split_names)

    # if args.no_debug:
    stdoutOrigin = sys.stdout
    sys.stdout = open(os.path.join(logs_dir, args.logs_file), "w")

    logger.info("Saving to logs_dir: {}".format(logs_dir))
    logstats_init(os.path.join(logs_dir, args.stats_file))
    # config_path = os.path.join(logs_dir, 'config.json')
    # logstats_add_args('config', args)
    # logstats_write_json(vars(args), config_path)


    # Load config, tokenizer, and downstream model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = tokenizer_class.from_pretrained(
              args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
              #   do_lower_case=True,
              cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_type == "sbert-retrieval":
        base_model = model_class(config=config,
                                 trans_model_name=args.model_name_or_path) #bert-base-multilingual-cased)
    else:
        base_model = model_class.from_pretrained(args.model_name_or_path,
                                                 from_tf=bool(".ckpt" in args.model_name_or_path),
                                                 config=config,
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
                                "mode": args.mode_qry,
                                "kd_lambda": args.kd_lambda,
                              }

    ## Create/load meta-dataset
    meta_dataset, meta_tasks, question_set, candidate_set = create_meta_dataset_lareqa(args,
                                                                                       meta_learn_split_config,
                                                                                       meta_tasks_dir,
                                                                                       tokenizer,
                                                                                       base_model)

    test_multilingual_evaluation = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
    print("map_scores_lang:", test_multilingual_evaluation, " MEAN:", np.mean([test_multilingual_evaluation[lang] for lang in test_multilingual_evaluation if lang!= "en"]))



    if args.do_evaluate:
        ## Train and validation
        opt = torch.optim.Adam(base_model.parameters(),
                            lr=meta_learn_split_config["train"]["beta_lr"])

        #
        # model_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs-temp/squad/bert-base-multilingual-cased_LR3e-5_EPOCH3.0_maxlen384_batchsize4_gradacc8/pytorch_model.bin"
        # optim_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs-temp/squad/bert-base-multilingual-cased_LR3e-5_EPOCH3.0_maxlen384_batchsize4_gradacc8/training_args.bin"
        # model_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs/lareqa/bert-base-multilingual-cased_LR2e-5_EPOCH3.0_LEN352/checkpoint-9000/pytorch_model.bin"
        # optim_load_file = "/sensei-fs/users/mhamdi/xtreme/outputs/lareqa/bert-base-multilingual-cased_LR2e-5_EPOCH3.0_LEN352/checkpoint-9000/optimizer.pt"
        optim_load_file = os.path.join(checkpoints_dir, "optimizer.pt")
        model_load_file =os.path.join(checkpoints_dir, "pytorch_model.bin")
        opt, base_model = load_torch_model(args, opt, base_model, optim_load_file, model_load_file)


        logger.info("Multilingual Evaluation for that model")
        map_scores_lang = multilingual_zero_shot_evaluation(question_set, candidate_set, args.languages.split(","), tokenizer, base_model,  meta_learn_split_config, args, "test")
        print("map_scores_lang:", map_scores_lang, " MEAN:", np.mean([map_scores_lang[lang] for lang in map_scores_lang if lang!= "en"]))
    else:
        def split_ep_dict(args):
            return {split_name: {ep:[] for ep in range(args.num_train_epochs)} for split_name in split_names}

        loss_qry_all_total, map_qry_all_total, loss_qry_avg_batch_total = split_ep_dict(args) , \
                                                                          split_ep_dict(args), \
                                                                          split_ep_dict(args)


        loss_qry_avg_batch_total, loss_qry_all_total, map_qry_all_total, train_map_scores_lang, valid_map_scores_lang, test_map_scores_lang = train_validate(args,
                                                                                                                                       meta_learn_split_config,
                                                                                                                                       meta_tasks,
                                                                                                                                       meta_tasks_dir,
                                                                                                                                       question_set,
                                                                                                                                       candidate_set,
                                                                                                                                       tokenizer,
                                                                                                                                       base_model,
                                                                                                                                       loss_qry_avg_batch_total,
                                                                                                                                       loss_qry_all_total,
                                                                                                                                       map_qry_all_total,
                                                                                                                                       checkpoints_dir,
                                                                                                                                       writer)


        for split_name in split_names:
            with open(os.path.join(runs_dir, split_name+"loss_qry_avg_batch_total.pickle"), "wb") as file:
                pickle.dump(loss_qry_avg_batch_total[split_name], file)

            with open(os.path.join(runs_dir, split_name+"_loss_qry_all_total.pickle"), "wb") as file:
                pickle.dump(loss_qry_all_total[split_name], file)

            with open(os.path.join(runs_dir, split_name+"_map_qry_all_total.pickle"), "wb") as file:
                pickle.dump(map_qry_all_total[split_name], file)


        with open(os.path.join(runs_dir, "train_map_scores_lang.pickle"), "wb") as file:
            pickle.dump(train_map_scores_lang, file)

        with open(os.path.join(runs_dir, "valid_map_scores_lang.pickle"), "wb") as file:
            pickle.dump(valid_map_scores_lang, file)

        with open(os.path.join(runs_dir, "test_map_scores_lang.pickle"), "wb") as file:
            pickle.dump(test_map_scores_lang, file)

    
    # if args.no_debug:
    sys.stdout.close()
    sys.stdout = stdoutOrigin


