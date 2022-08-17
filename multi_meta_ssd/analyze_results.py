import os
import pickle
import torch
import numpy as np
import json
from tqdm import tqdm
from multi_meta_ssd.processors.downstream import utils_lareqa
from transformers import BertConfig, BertTokenizer, XLMRobertaTokenizer
from multi_meta_ssd.models.downstream.dual_encoders.bert import BertForRetrieval
from multi_meta_ssd.processors.downstream.utils_lareqa import mean_avg_prec_at_k_eval

results_dir="/sensei-fs/users/mhamdi/Results/meta-multi-sem-search/asym/"
data_root="/sensei-fs/users/mhamdi/Datasets/meta-multi-sem-search/"


def read_data(model, mode,  languages):
    base_path = os.path.join(data_root, 'lareqa')
    dataset_to_dir = {
      "xquad": "xquad-r",
    }

    squad_dir = os.path.join(base_path, dataset_to_dir["xquad"], "splits")
    # languages="ar,de,el,hi,ru,th,tr,en"
    split_names = ["test"]
    question_set, candidate_set = {}, {}
    # Load the question set and candidate set.
    print("Loading the dataset ....")
    squad_per_lang = {} #set(languages.split(","))
    for filename in os.listdir(os.path.join(squad_dir, "test")):
        language = os.path.splitext(filename)[0]
        # languages.add(language)
        if language in languages:
            with open(os.path.join(squad_dir, "test", filename), "r") as f:
                squad_per_lang[language] = json.load(f)
            print("Loaded %s" % filename)

    # Full Multilingual evaluation
    question_set, candidate_set = utils_lareqa.load_data(squad_per_lang, sim_model=None)
    map_at_20_test_all = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased",
        do_lower_case=True
    )

    c_features = [tokenizer.encode_plus((candidate.sentence+candidate.context).replace("\n", ""),
                                            max_length=64,
                                            pad_to_max_length=True,
                                            return_token_type_ids=True) for candidate in candidate_set.as_list()]

    input_ids_total = []
    attention_mask_total = []
    token_type_ids_total = []
    for i in range(len(c_features)):
        input_ids_total.append(c_features[i]["input_ids"])
        attention_mask_total.append(c_features[i]["attention_mask"])
        token_type_ids_total.append(c_features[i]["token_type_ids"])

    c_features_new = {"input_ids": input_ids_total, "attention_mask": attention_mask_total, "token_type_ids": token_type_ids_total}

    q_input_ids_=torch.tensor(c_features_new["input_ids"], dtype=torch.long)
    q_attention_mask_=torch.tensor(c_features_new["attention_mask"], dtype=torch.long)
    q_token_type_ids_=torch.tensor(c_features_new["token_type_ids"], dtype=torch.long)

    print("CANDIDATES q_input_ids_.shape:", q_input_ids_.shape, "q_attention_mask_.shape:", q_attention_mask_.shape, " q_token_type_ids_.shape:", q_token_type_ids_.shape)

    c_encodings = model(q_input_ids=q_input_ids_,
                        q_attention_mask=q_attention_mask_,
                        q_token_type_ids=q_token_type_ids_,
                        inference=True)
    for question in tqdm(question_set.as_list()):
        q_features = tokenizer.encode_plus(question.question,
                                           max_length=64,
                                           pad_to_max_length=True,
                                           return_token_type_ids=True)

        q_input_ids_=torch.unsqueeze(torch.tensor(q_features["input_ids"], dtype=torch.long), dim=0)
        q_attention_mask_=torch.unsqueeze(torch.tensor(q_features["attention_mask"], dtype=torch.long), dim=0)
        q_token_type_ids_=torch.unsqueeze(torch.tensor(q_features["token_type_ids"], dtype=torch.long), dim=0)
        q_encodings = model(q_input_ids=q_input_ids_,
                            q_attention_mask=q_attention_mask_,
                            q_token_type_ids=q_token_type_ids_,
                            inference=True)


        map_at_20_test = mean_avg_prec_at_k_eval([question],
                                                 q_encodings,
                                                 candidate_set,
                                                 c_encodings,
                                                 k=20)
        map_at_20_test_all.append(map_at_20_test)


    print(" mode: ", mode, " languages:", ",".join(languages), np.mean(map_at_20_test_all))

    return np.mean(map_at_20_test_all)

def zero_shot_test(mode):
    with open(os.path.join(results_dir, mode, "checkpoints/bert-retrieval/maml", "pytorch_model.bin"), "rb") as file:
        model_dict = torch.load(file)

    config = BertConfig.from_pretrained("bert-base-multilingual-cased")
    base_model = BertForRetrieval.from_pretrained("bert-base-multilingual-cased",
                from_tf=bool(".ckpt" in "bert-base-multilingual-cased"),
                config=config,
    )
    base_model.load_state_dict(model_dict)
    precision = []
    precision.append(read_data(base_model, mode,  ["ar"]))
    precision.append(read_data(base_model, mode,  ["de"]))
    precision.append(read_data(base_model, mode,  ["el"]))
    precision.append(read_data(base_model, mode,  ["hi"]))
    precision.append(read_data(base_model, mode,  ["ru"]))
    precision.append(read_data(base_model, mode,  ["th"]))
    precision.append(read_data(base_model, mode,  ["tr"]))

    precision.append(read_data(base_model, mode,  ["ar", "de"]))
    precision.append(read_data(base_model, mode,  ["el", "ar"]))
    precision.append(read_data(base_model, mode,  ["hi", "de"]))
    precision.append(read_data(base_model, mode,  ["el", "ar", "hi"]))

    print("***************************************")
    print(precision)

zero_shot_test("mixt")

def print_train_qry_results():
    for mode in ["mono_mono", "mono_bil", "bil_multi", "mixt"]:
        train_loss_qry_file = os.path.join(results_dir, mode, "runs", "train_loss_qry_all_total.pickle")
        train_precision_qry_file = os.path.join(results_dir, mode, "runs", "train_precision_qry_all_total.pickle")

        with open(train_loss_qry_file, "rb") as file:
            train_loss_qry = pickle.load(file)

        with open(train_precision_qry_file, "rb") as file:
            train_precision_qry = pickle.load(file)

        ep = 0
        # print("len(train_loss_qry[ep]):", len(train_loss_qry[ep]))
        # print("train_precision_qry[1][0]:", train_precision_qry[1][0])
        # print("train_loss_qry[1][0]:", train_loss_qry[1][0])
        new_train_loss_qry = []
        for i in range(len(train_loss_qry)):
            new_train_loss_qry_sublist = []
            for j in range(len(train_loss_qry[i])):
                new_train_loss_qry_sublist_ = []
                for k in range(len(train_loss_qry[i][j])):
                    new_train_loss_qry_sublist_.append(train_loss_qry[i][j][k].cpu().detach().numpy())
                new_train_loss_qry_sublist.append(new_train_loss_qry_sublist_)
            new_train_loss_qry.append(new_train_loss_qry_sublist)

        print(mode, " train_loss_qry:", np.mean([new_train_loss_qry[ep][i] for i in range(len(new_train_loss_qry[ep]))]), " train_precision_qry:", np.mean([train_precision_qry[1][i] for i in range(len(train_precision_qry[1]))]))

# print_train_qry_results()