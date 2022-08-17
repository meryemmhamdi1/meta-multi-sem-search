def get_path_options(parser):
    """ Path Options """
    # path_params = parser.add_argument_group('Path Parameters')
    parser.add_argument('--data_root', type=str,
                              help='Path of data directory root.')

    parser.add_argument('--train_file',  type=str,
                              help='Path of pre-fine-tuning train file.')

    parser.add_argument("--predict_file", type=str, default=None,
                              help="The path of pre-fine-tuning evaluation file. If a data dir is specified, will look for the file there."
                              + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")

    parser.add_argument('--out_dir', type=str,
                              help='Path of results directory root.')

    parser.add_argument("--cache_dir", type=str, default=None,
                              help="Where do you want to store the pre-trained models downloaded from s3.")

    parser.add_argument("--overwrite_cache",action="store_true",
                              help="Overwrite the cached training and evaluation sets.")

    parser.add_argument("--no_debug", action="store_false",
                              help="If true, save training and testing logs to disk.")

    parser.add_argument("--stats_file", type=str, default="stats.txt",
                              help="Filename of the stats file.")  # TODO CHECK WHAT THIS DOES EXACTLY

    parser.add_argument("--logs_file", type=str, default="log.txt",
                              help="Filename of the log file.")  # TODO DO PROPER CHECKPOINTING

    parser.add_argument('--load_pre_finetune_path',  type=str, default="",
                              help='The path from which to load the pre-finetuning path')

# TODO CHANGE get_base_model_options TO MAKE IT COMPLY WITH THE MODEL
def get_base_model_options(parser):

    parser.add_argument("--use_sim_embedder", action="store_true",
                              help="Whether to use similarity embedder model.")

    parser.add_argument("--sim_model_name", type=str, default="paraphrase-multilingual-mpnet-base-v2",
                              help="Sentence transformers model to be used.")

    parser.add_argument("--model_type", type=str, choices=['bert-retrieval', 'xlmr-retrieval', 'sbert-retrieval'],
                                        help="Model type selected in the list: bert-retrieval, xlmr-retrieval")

    parser.add_argument("--tokenizer_name", default="", type=str,
                                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument( "--model_name_or_path", default=None, type=str,
                                                 help="Path to pre-trained model or shortcut name selected in the list ")

    parser.add_argument('--vocab_size', type=int, default=50265,
                                   help='Vocabulary size of the MBART model.')

    parser.add_argument('--d_model', type=int, default=1024,
                                   help='Dimensionality of the layers and the pooler layer.')

    parser.add_argument('--encoder_layers', type=int, default=12,
                                   help='Number of encoder layers.')

    parser.add_argument('--dropout', type=float, default=0.1,
                                   help='The dropout probability for all fully connected layers in the'
                                        'embeddings, encoder, and pooler.')

    parser.add_argument('--attention_dropout', type=float, default=0.0,
                                   help='The dropout ratio for the attention probabilities.')

    parser.add_argument('--activation_dropout', type=float, default=0.0,
                                   help='The dropout ratio for activations inside the fully connected layer.')

    parser.add_argument('--classifier_dropout', type=float, default=0.0,
                                   help='The dropout ratio for classifier.')

    parser.add_argument('--max_position_embeddings', type=int, default=1024,
                                   help='The maximum sequence length that this model might ever be used with.')

    parser.add_argument('--init_std', type=float, default=0.02,
                                   help='The standard deviation of the truncated_normal_initializer for'
                                        'initializing all weight matrices.')

    parser.add_argument('--encoder_layerdrop', type=float, default=0.0,
                                   help='The LayerDrop probability for the encoder.')

    parser.add_argument('--decoder_layerdrop', type=float, default=0.0,
                                   help='The LayerDrop probability for the decoder.')

    parser.add_argument('--gradient_checkpointing', action='store_true',
                                   help='If True, use gradient checkpointing to save memory at the expense of'
                                        'slower backward pass.')

    parser.add_argument('--scale_embedding', action='store_true',
                                   help='Scale embeddings by diving by sqrt(d-model).')

    parser.add_argument('--not_use_cache', action='store_true',
                                   help='Whether or not the model should return the last'
                                        'key/values attentions (not used by all models)')

    parser.add_argument('--forced_eos_token_id', type=int, default=2,
                                   help='The id of the token to force as the last generated token when '
                                        '`max_length` is reached. Usually set to `eos_token_id`.')

    parser.add_argument('--max_seq_length', type=int, default=384,
                                   help='The max total input sequence length after WordPiece tokenization. Sequences longer'
                                        'than this will be truncated, and sequences shorter than this will be padded.')

    parser.add_argument("--max_query_length", default=64, type=int,
                                  help="The maximum number of tokens for the question. Questions longer than this will "
                                       "be truncated to this length.")

    parser.add_argument("--max_answer_length", default=128, type=int,
                             help="The maximum number of tokens for the answer and context. longer than this will "
                                  "be truncated to this length.")

    parser.add_argument("--do_lower_case", action="store_false",
                                     help="Set this flag if you are using an uncased model.")


    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")

def get_language_options(parser):
    """ Multi-lingual Semantic Search """
    parser.add_argument("--mode_transfer", type=str, choices=["MONO_MONO", "MONO_BIL",  "MONO_MULTI", "BIL_MULTI", "MIXT", "TRANS", "MONO_BIL_MULTI"])

    parser.add_argument("--do_evaluate", action="store_true", help="Whether to evaluate the model.")

    parser.add_argument('--do_pre_finetune',  action="store_true",
                              help='Whether to pre-fine-tune or not before fine-tuning on LAREQA')

    parser.add_argument('--pre_finetune_language', type=str, default='en',
                              help='Pre-fine-tuning language(s) separated by comma (one or multiple languages separated by comma) of the pre-finetuning stage.')

    parser.add_argument('--num_prefinetune_epochs', type=int, default=10,
                              help="The number of epochs in pre-fine-tuning stage.")

    parser.add_argument('--languages', type=str, default='ar,de,el,hi,ru,th,tr', #default="ar,el",
                              help='The list of languages.')

    parser.add_argument('--train_lang_pairs', type=str, default='ar_ar-de_de',#default=['ar_ar-ar_de', 'de_de-de_el'], # monolingual to bilingual
                              help='Train/parent languages pairs list (most high-resourced pairs). Each language pair is separated by a dash'
                                   ' between the language of the support and the languages of the query separated by a comma.')

    parser.add_argument('--valid_lang_pairs', type=str, default='el_ar-el_ar,el|hi_de-hi_de,hi|el_ar-el_ar,de,hi|hi_de-hi_de,ar,el', # bilingual to multilingual and multilingual
                              help='Language pairs list for validation (intermediate resource pairs). Each language pair is separated by a dash'
                                   ' between the language of the support and the languages of the query separated by a comma.')

    parser.add_argument('--test_lang_pairs', type=str, default='ar_ar-ar_de,th,hi,ru,el|hi_hi-hi_th,ar,ru,el|th_th-th_ru,ar,hi|el_el-el_hi,ar,el', # monolingual to multilingual for unseen languages
                              help='Test/children language pairs list (most low-resource pairs), testing on different combinations of.'
                                   ' Each language pair is separated by a dash between the language of the support and the languages of the query'
                                   ' separated by a comma.')

def get_meta_task_options(parser):
    """ Meta-Tasks Construction Setup Options """
    parser.add_argument('--prop', type=float, default=0.1,
                              help='To simulate different low-resource/downsampling setups.')

    parser.add_argument('--k_spt', type=int, default=8,
                              help='Total number of support examples where example=(source sent, target sent).') # could be adjusted proportionally

    parser.add_argument('--q_qry', type=int, default=4,
                              help='Total number of query examples.')

    parser.add_argument('--n_train_tasks', type=int, default=7000,
                              help='Total number of tasks for meta-train dataset.')

    parser.add_argument('--n_valid_tasks', type=int, default=2000,
                              help='Total number of tasks for meta-validation dataset.')

    parser.add_argument('--n_test_tasks', type=int, default=1000,
                              help='Total number of tasks for meta-testing dataset.')

    parser.add_argument("--mode_qry", type=str, default="similar", choices=["similar", "random"],
                              help='The mode of sampling of the query.')
     
    parser.add_argument("--update_meta_data", action='store_true', help='Whether to update the meta-dataset.')
                         

    """ Meta-Learning Algorithms Options """ ## TODO Meryem: SHOULD ADD PARAMS FOR MORE META ALGORITHMS ## SNAIL or Another Meta-learning Algorithm Hyperparameters
    parser.add_argument('--use_meta_learn', action='store_true', help='Whether not to use meta-learning or not.')

    parser.add_argument('--meta_learn_alg', type=str, choices=['maml', 'meta-sgd', 'reptile', 'maml_align'], # default is none
                              help='The choice of the upstream meta-learning algorithm')

    parser.add_argument('--kd_lambda', type=float, default=0.5,
                              help='Lambda weight for the contribution of the kd loss.')

    parser.add_argument('--n_prefinetune_tasks_batch', type=int, default=4,
                              help='Number of meta-tasks per batch for pre-finetuning.')

    parser.add_argument('--n_train_tasks_batch', type=int, default=4,
                              help='Number of meta-tasks per batch for meta-training.')

    parser.add_argument('--n_valid_tasks_batch', type=int, default=4,
                              help='Number of meta-tasks per batch for meta-validation')

    parser.add_argument('--n_test_tasks_batch', type=int, default=4,
                                   help='Number of meta-tasks per batch for meta-testing')

    ##
    parser.add_argument('--n_up_prefinetune_steps', type=int, default=5,
                                   help='Number of inner loop update steps in the pre-finetuning stage')

    parser.add_argument('--n_up_train_steps', type=int, default=5,
                                   help='Number of inner loop update steps in the meta-training stage')

    parser.add_argument('--n_up_valid_steps', type=int, default=5,
                                   help='Number of inner loop update steps in the meta-validation stage')

    parser.add_argument('--n_up_test_steps', type=int, default=5,
                                   help="Number of inner loop update steps in the meta-testing stage")

    ##
    parser.add_argument('--alpha_lr_prefinetune', type=float, default=1e-2,
                                   help='Learning rate during the prefinetuning.')

    parser.add_argument('--alpha_lr_train', type=float, default=1e-3,
                                   help='Learning rate during the task-specific stage (inner loop).')

    parser.add_argument('--beta_lr_train', type=float, default=1e-3,
                                   help='Learning rate during the optimization of the meta-knowledge (outer loop).')

    #
    parser.add_argument('--alpha_lr_valid', type=float, default=1e-3,
                                   help='Learning rate during the inner loop with the validation meta-dataset.')

    #
    parser.add_argument('--alpha_lr_test', type=float, default=1e-3,
                                   help='Learning rate during the inner loop with the testing meta-dataset.')

def get_adam_optim_params(parser):
    """ Optimizer/Training Options """
    parser.add_argument('--adam_lr', type=float, default=3e-5,
                                   help='learning rate of adam optimizer when training base model from scratch')

    parser.add_argument('--adam_eps', type=float, default=1e-08,
                                   help='epsilon of adam optimizer when training base model from scratch')

    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay if we apply some.')

    parser.add_argument('--warmup_steps', type=int, default=0, help='Linear warmup over warmup_steps.')

    parser.add_argument('--step_size', type=int, default=7, help='The step size for the scheduler.')

    parser.add_argument('--gamma', type=float, default=0.1, help='gamma for the scheduler')

def get_train_params(parser):
    """ Training Arguments """
    parser.add_argument("--use_triplet_loss", action='store_true',
                              help='Whether to use triplet loss or contrastive loss.')

    parser.add_argument("--use_cross_val", action='store_true',
                              help='Whether to use cross-validation.')

    parser.add_argument("--cross_val_split", type=str, default="0",
                              help='The split number to be tested in cross-validation.')

    parser.add_argument("--n_neg_eg", type=int, default=5,
                              help='The number of negative examples.')

    parser.add_argument("--neg_sampling_approach", type=str, default="random", choices=["random", "semi-hard", "hard", "easy"],
                              help="The sampling approach used in triplet loss where to sample randomly or semi-hard or hard or easy")

    parser.add_argument("--triplet_batch_size", type=int, default=3,
                              help='The number of negative examples.')

    parser.add_argument('--pre_train_steps', type=int, default=2000,
                              help='# of iterations if pre-training of base model before any upstream training')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size in the pre-training process')

    parser.add_argument("--num_train_epochs", default=5, type=float, help="Total number of training epochs to perform.")

    parser.add_argument('--save_every', type=int, default=1, help='How often to save a checkpoint of the model')

    parser.add_argument('--early_stopping', type=int, default=10,
                              help='Number of epochs after which we stop running the model if the model stops improving'
                                   'on the meta-validation.')

    parser.add_argument('--n_valid_epochs', type=int, default=2,
                              help='The number of epochs after which we need to choose best model or hyperparameters.')

    parser.add_argument('--n_test_epochs', type=int, default=2,
                              help='The number of epochs after which we need to test.')

    parser.add_argument('--seed', type=int, default=109883242, help='Random seed for initialization')

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")

    ## Parallel Training
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--gpu_order", type=str, default="0", help="Which GPU to use in the node.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")

    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training.")

    parser.add_argument('--no_cuda', action='store_true', help='Whether not to use CUDA when available')

    parser.add_argument('--use_dpp', action='store_true', help='Whether to use DPP or not')

    parser.add_argument('--valid_steps', type=int, default=10, help='The frequency of the steps to run meta-validation.')

    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
