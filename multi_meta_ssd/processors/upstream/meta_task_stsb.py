import collections, random
from operator import pos
from tqdm import tqdm
import torch
from sentence_transformers import util
import numpy as np
from itertools import cycle

MAX_SENT_LEN = 100

def circ1(lst):
    i = -1
    while lst:
        if i is len(lst)-1:
            i = 0
            # Shuffle the list of sentence pairs by index again
            random.shuffle(lst)
            return lst[i]
        else:
            i += 1
            return lst[i]

def circ(lst, i):
    j = i % len(lst)
    return lst[j]

class MetaPoint(object):
    """ This class is the smallest unit of sentence pair"""
    def __init__(self, sentence1, sentence2, sentence1_feat, sentence2_feat, score):
        """
            @sentence1: sentence 1 
            @sentence2: sentence 2
            @score: similarity score of sentences 1 and 2
        """
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.sentence1_feat = sentence1_feat
        self.sentence2_feat = sentence2_feat
        self.score = score

class MetaSet(object):
    """ This class is to organize the format of the support or query set. Each set consists of a set of sentence pairs and their corresponding scores."""
    def __init__(self, meta_points, id):
        """
            @meta_points: list of meta-points
        """
        self.id = id
        self.meta_points = meta_points
        self.sentences1 = [meta_point.sentence1 for meta_point in meta_points]
        self.sentences2 = [meta_point.sentence2 for meta_point in meta_points]

        input_ids_total = []
        attention_mask_total = []
        token_type_ids_total = []
        for i in range(len(meta_points)):
            input_ids_total.append(meta_points[i].sentence1_feat["input_ids"])
            attention_mask_total.append(meta_points[i].sentence1_feat["attention_mask"])
            token_type_ids_total.append(meta_points[i].sentence1_feat["token_type_ids"])

        self.sentences1_feat = {"input_ids": torch.tensor(input_ids_total, dtype=torch.long),
                                "attention_mask": torch.tensor(attention_mask_total, dtype=torch.long),
                                "token_type_ids": torch.tensor(token_type_ids_total, dtype=torch.long)
                               }

        input_ids_total = []
        attention_mask_total = []
        token_type_ids_total = []
        for i in range(len(meta_points)):
            input_ids_total.append(meta_points[i].sentence2_feat["input_ids"])
            attention_mask_total.append(meta_points[i].sentence2_feat["attention_mask"])
            token_type_ids_total.append(meta_points[i].sentence2_feat["token_type_ids"])

        self.sentences2_feat = {"input_ids": torch.tensor(input_ids_total, dtype=torch.long),
                                "attention_mask": torch.tensor(attention_mask_total, dtype=torch.long),
                                "token_type_ids": torch.tensor(token_type_ids_total, dtype=torch.long)
                               }

        self.scores = torch.tensor([meta_point.score for meta_point in meta_points], dtype=torch.float)

        self.inputs = {"sent1_"+k: v for k, v in self.sentences1_feat.items()}
        self.inputs.update({"sent2_"+k: v for k, v in self.sentences2_feat.items()})
        self.inputs.update({"scores_gs": self.scores})
        

class MetaTask(object):
    """ This class is to organize the structure of each task consisting of support and query.
        To start with, we will use corresponding questions in other language and off-the-shelf tool to get the most similar questions in the same language. """
    def __init__(self, id, sentences_pair, translate_train, spt_sent1_lang, spt_sent2_lang, qry_sent1_lang, qry_sent2_lang, meta_learn_split_config, tokenizer, spt_indices, qry_indices):
        """
            @sentences_pair: dictionary of sentence pairs per split and language pair
            @qry_lang: the language(s) of the query meta set
            @k_qry: the number of sentences1 to be retrieved for the given support
        """
        self.id = id
        self.sentences_pair = sentences_pair
        self.translate_train = translate_train

        self.spt_sent1_lang = spt_sent1_lang
        self.spt_sent2_lang = spt_sent2_lang

        self.qry_sent1_lang = qry_sent1_lang
        self.qry_sent2_lang = qry_sent2_lang

        self.meta_learn_split_config = meta_learn_split_config
        self.tokenizer = tokenizer
        
        self.spt_indices = spt_indices
        self.qry_indices = qry_indices 

        # self.spt_indices = circ(self.spt_indices)
        # self.qry_indices = circ(self.qry_indices)

        self.spt = self.create_meta_set("train", meta_learn_split_config["k_spt"], self.spt_sent1_lang, self.spt_sent2_lang, self.spt_indices)
        self.qry = self.create_meta_set("dev", meta_learn_split_config["q_qry"], self.qry_sent1_lang, self.qry_sent2_lang, self.qry_indices)
        
    def create_meta_set(self, split_name, len_meta_task, sent1_lang, sent2_lang, indices): # lang_pair
        # Pick k-shot or q_qry sentence pairs at a time
        meta_points = []
        for _ in range(len_meta_task):
            # rnd_idx = next(indices)
            # circ(indices, )
            rnd_idx = random.choice(indices)

            # print("rnd_idx:", rnd_idx)

            if self.translate_train:
                lang_pair_1 = sent1_lang
                lang_pair_2 = sent2_lang
            else:
                lang_pair_1 = lang_pair_2 = sent1_lang + "-" + sent2_lang
            
            sent1 = self.sentences_pair[split_name][lang_pair_1]["sentences1"][rnd_idx]
            sent2 = self.sentences_pair[split_name][lang_pair_2]["sentences2"][rnd_idx]
            score = self.sentences_pair[split_name][lang_pair_1]["scores"][rnd_idx]
            sent1_feat = self.sentences_pair[split_name][lang_pair_1]["sentences1_feat"][rnd_idx]
            sent2_feat = self.sentences_pair[split_name][lang_pair_2]["sentences2_feat"][rnd_idx]

            meta_point = MetaPoint(sentence1=sent1,
                                   sentence2=sent2,
                                   sentence1_feat=sent1_feat,
                                   sentence2_feat=sent2_feat,
                                   score=score)
            
            meta_points.append(meta_point)

        return MetaSet(meta_points, self.id)

    def shuffle_candidate_xling(self):
        candidates = list(self.candidate_xling.items())
        random.shuffle(candidates)
        self.candidate_xling = collections.OrderedDict(candidates)

class MetaDataset(object):
    """ This class holds sets of tasks for different meta-datasets (train, dev, and test)"""
    def __init__(self, meta_learn_split_config, sentences_pair, split_name, tokenizer, translate_train): # n_train_tasks, n_valid_tasks, n_test_tasks, , train_lang_pairs, valid_lang_pairs, test_lang_pairs):
        self.n_tasks = meta_learn_split_config[split_name]["n_tasks_total"]
        self.lang_pairs = meta_learn_split_config[split_name]["lang_pairs"]
        self.sentences_pair = sentences_pair
        self.meta_learn_split_config = meta_learn_split_config
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.translate_train = translate_train
        self.meta_tasks = []

        self.create_meta_tasks()

    def create_meta_tasks(self):
        n_tasks_per_spt_lang = self.n_tasks // len(self.lang_pairs.split("|"))

        for lang_pair in self.lang_pairs.split("|"):
            pbar = tqdm(range(n_tasks_per_spt_lang))
            pbar.set_description("Processing %s" % lang_pair)
            for metatask_n in pbar:
                spt_langs, qry_langs = lang_pair.split("_")

                spt_sent1_lang, spt_sent2_lang = spt_langs.split("-")
                qry_sent1_lang, qry_sent2_lang = qry_langs.split("-")

                if self.translate_train:
                    spt_langs = spt_sent1_lang
                    qry_langs = qry_sent1_lang

                spt_indices = list(range(len(self.sentences_pair["train"][spt_langs]["sentences1"])))
                qry_indices = list(range(len(self.sentences_pair["dev"][qry_langs]["sentences1"])))

                # Shuffle sentences_pairs indices
                random.shuffle(spt_indices)
                random.shuffle(qry_indices)

                # Construct the support and query sets
                meta_task = MetaTask(lang_pair+"_"+str(metatask_n),
                                     self.sentences_pair,
                                     self.translate_train,
                                     spt_sent1_lang,
                                     spt_sent2_lang,
                                     qry_sent1_lang,
                                     qry_sent2_lang,
                                     self.meta_learn_split_config,
                                     self.tokenizer, 
                                     spt_indices,
                                     qry_indices)

                self.meta_tasks.append(meta_task)

