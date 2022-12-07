try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (AdamW, WEIGHTS_NAME, get_linear_schedule_with_warmup)
import os
import torch
import logging
import random
import timeit
import numpy as np
from tqdm import tqdm, trange
import itertools
from multi_meta_ssd.processors.downstream import lareqa
import configparser

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def set_device(args):
    if args.local_rank == -1 or args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_order
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        # device = available_gpus[0]
        # device = torch.device("cuda:"+args.gpu_order.split(",")[0] if torch.cuda.is_available() and not args.no_cuda else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

def get_config_params(args, task_type):
    config_path = 'multi_meta_ssd/config/'

    paths = configparser.ConfigParser()
    paths.read(config_path+'paths.ini')

    location = "ENDEAVOUR"

    if task_type == "sym":
        location += "_SYM"

    root_dir = str(paths.get(location, "ROOT"))

    args.data_root = root_dir + str(paths.get(location, "DATA_ROOT"))
    args.train_file = root_dir + str(paths.get(location, "TRAIN_FILE"))
    args.predict_file = root_dir + str(paths.get(location, "PREDICT_FILE"))
    args.out_dir = root_dir + str(paths.get(location, "OUT_DIR"))
    args.load_pre_finetune_path = root_dir + str(paths.get(location, "LOAD_PRE_FINETUNE_PATH"))

    params = configparser.ConfigParser()
    params.read(config_path+'down_model_param.ini')
    
    args.pre_finetune_language = str(params.get("PARAMS", "PRE_FINETUNE_LANGUAGE"))
    args.max_seq_length = int(params.get("PARAMS", "MAX_SEQ_LENGTH"))
    args.max_query_length = int(params.get("PARAMS", "MAX_QUERY_LENGTH"))
    args.max_answer_length = int(params.get("PARAMS", "MAX_ANSWER_LENGTH"))

    params = configparser.ConfigParser()
    meta_task_config_path = config_path+'meta_task_param'
    if task_type == "sym":
        meta_task_config_path += "_stsb" 
    params.read(meta_task_config_path+'.ini')

    if args.translate_train:
        tlangs = args.translate_train_langs.split(",")
        if task_type == "asym":
            t_lang_pairs = "|".join([lang+"_"+lang+"-"+lang+"_"+lang for lang in tlangs])
        else:
            # For now we are only sampling from the same language
            t_lang_pairs = "|".join([lang+"-"+lang+"_"+lang+"-"+lang for lang in tlangs])
        args.train_lang_pairs = args.valid_lang_pairs = args.test_lang_pairs =  t_lang_pairs
    else:
        args.train_lang_pairs = str(params.get(args.mode_transfer, "TRAIN_LANG_PAIRS"))
        args.valid_lang_pairs = str(params.get(args.mode_transfer, "VALID_LANG_PAIRS"))
        args.test_lang_pairs = str(params.get(args.mode_transfer, "TEST_LANG_PAIRS"))

    if args.use_triplet_loss:
        args.n_neg_eg = 1

    if task_type == "sym":
        dev_valid_name = "dev"
    else:
        dev_valid_name = "valid"

    print("args.train_lang_pairs:", args.train_lang_pairs, " args.valid_lang_pairs:", args.valid_lang_pairs, " args.test_lang_pairs:", args.test_lang_pairs)
    meta_learn_split_config = {"train": {"n_tasks_total": args.n_train_tasks,
                                         "n_tasks_batch": args.n_train_tasks_batch,
                                         "n_up_steps": args.n_up_train_steps,
                                         "alpha_lr": args.alpha_lr_train,
                                         "beta_lr": args.beta_lr_train,
                                         "lang_pairs": args.train_lang_pairs},

                                dev_valid_name: {"n_tasks_total": args.n_valid_tasks,
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
                                "k_spt": args.k_spt,
                                "q_qry": args.q_qry
                                }
    return args, meta_learn_split_config

def set_out_dir(args, task_name):
    out_dir = os.path.join(args.out_dir,
                           "SEED_"+str(args.seed),
                           args.model_type,
                           args.meta_learn_alg if args.use_meta_learn else "finetune",
                           args.mode_transfer)

    if args.do_pre_finetune:
        out_dir = os.path.join(out_dir, "PreFineTune")

    if args.translate_train:
        out_dir = os.path.join(out_dir, "TranslateTrain", args.translate_train_langs)
    
    if args.use_cross_val:
        out_dir = os.path.join(out_dir, "CrossVal_"+args.cross_val_split)

    if task_name == "asym":
        if args.use_triplet_loss:
            out_dir = os.path.join(out_dir, "TripletLoss")

        if args.neg_sampling_approach != "random":
            out_dir = os.path.join(out_dir, "neg_sampling_"+args.neg_sampling_approach)

        if args.mode_qry == "random":
            out_dir = os.path.join(out_dir, "random")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    checkpoints_dir = build_dir(out_dir, "checkpoints")
    runs_dir = build_dir(out_dir, "runs")
    logs_dir = build_dir(out_dir, "logs")

    #########################################
    # Constructing Meta-Tasks Directory

    meta_tasks_dir = os.path.join(args.out_dir,
                                  args.model_type,
                                  "meta_tasks",
                                  args.mode_transfer,
                                  ",".join(sorted(set(args.languages.split(","))))) # different path for meta_tasks

    if task_name == "asym":
        meta_tasks_dir = os.path.join(meta_tasks_dir,
                                      "TripletLoss" if args.use_triplet_loss else "NegEg"+str(args.n_neg_eg),
                                      args.mode_qry if args.mode_qry == "random" else args.sim_model_name if args.use_sim_embedder else "NoSIM")

        if args.neg_sampling_approach != "random":
            meta_tasks_dir = os.path.join(meta_tasks_dir, "neg_sampling_"+args.neg_sampling_approach)

    if args.translate_train:
        meta_tasks_dir = os.path.join(meta_tasks_dir, "TranslateTrain", args.translate_train_langs)

    if args.use_cross_val:
        meta_tasks_dir = os.path.join(meta_tasks_dir, "CrossVal_"+args.cross_val_split)

    if not os.path.isdir(meta_tasks_dir):
        os.makedirs(meta_tasks_dir)

    writer = SummaryWriter(runs_dir)

    print("meta_tasks_dir:", meta_tasks_dir, " runs_dir:", runs_dir)

    return meta_tasks_dir, checkpoints_dir, runs_dir, logs_dir, writer

def get_config_tokenizer_model(model_spec, args):
    config_class, model_class, tokenizer_class = model_spec[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = tokenizer_class.from_pretrained(
              args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
              cache_dir=args.cache_dir if args.cache_dir else None
    )

    if args.model_type == "sbert-retrieval":
        base_model = model_class(config=config,
                                 trans_model_name=args.model_name_or_path)
    else:
        base_model = model_class.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path,
                                                 config=config,
                                                 from_tf=bool(".ckpt" in args.model_name_or_path),
                                                 cache_dir=args.cache_dir if args.cache_dir else None)

    return tokenizer, base_model

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_split_dict(SPLIT_NAMES):
    return {split_name: {} for split_name in SPLIT_NAMES}

def split_ep_dict(args, SPLIT_NAMES):
    return {split_name: {ep:[] for ep in range(args.num_train_epochs)} for split_name in SPLIT_NAMES}

def build_dir(root_dir, name):
    sub_dir = os.path.join(root_dir, name)
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    return sub_dir

def circle_batch(iterable, batchsize):
    it = itertools.cycle(iterable)
    while True:
        yield list(itertools.islice(it, batchsize))

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

def save_torch_model(args, model, optimizer, checkpoints_dir, epoch):
    args_save_file = os.path.join(checkpoints_dir, "training_args_"+epoch+".bin")
    model_save_file = os.path.join(checkpoints_dir, "pytorch_model_"+epoch+".bin")
    optim_save_file = os.path.join(checkpoints_dir, "optimizer_"+epoch+".pt")
    torch.save(args, args_save_file)
    torch.save(model.state_dict(), model_save_file)
    torch.save(optimizer.state_dict(), optim_save_file)
