# from multi_meta_ssd.downstream_processors.utils_lareqa import CandidateSet
# from sentence_transformers import SentenceTransformer, util
import collections, random
from operator import pos
from tqdm import tqdm
import torch
from multi_meta_ssd.processors.downstream.utils_lareqa import convert_features
from sentence_transformers import util
import numpy as np

# MetaDataset -> batches of meta-tasks -> 1 support set + 2 query sets
# LAREQA: question -> candidate sentences, answers


class MetaSet(object):
    """ This class is to organize the format of the support or query set. Each set consists of question or its cluster, positive and negative candidate retrieved sentences."""
    """ To start with, we have each question is a cluster so we have N-way with 1 shot TODO we can implement semantic similarity or use off-the-shelf semantic similarity to come up with clusters of questions so we have N-way, K-shot"""
    def __init__(self, question_cluster, positive_candidates=[], negative_candidates=[]):
        """
            @question_cluster: question or cluster of questions of type Question
            @positive_candidates: list of positive candidates (answers to the question_cluster)
            @negative_candidates: list of negative candidates (not answers to the question)
        """
        self.question_cluster = question_cluster
        self.positive_candidates = positive_candidates
        self.negative_candidates = negative_candidates
        self.all_candidates = positive_candidates + negative_candidates

    def add_positive_candidate(self, candidate):
        self.positive_candidates.append(candidate)
        self.all_candidates.append(candidate)

    def add_negative_candidate(self, candidate):
        self.negative_candidates.append(candidate)
        self.all_candidates.append(candidate)


class MetaTaskMerged(object):
    """ This class is to organize the structure of each task consisting of support and query.
        To start with, we will use corresponding questions in other language and off-the-shelf tool to get the most similar questions in the same language. """
    def __init__(self, spt_qst, spt_qst_lang, spt_cand_langs, qry1_qst_lang, qry1_cand_langs, qry2_qst_lang, qry2_cand_langs, question_set, candidate_set, meta_learn_split_config, tokenizer, top_results):
        """
            WE WANT TO DO META-TRANSFR FROM BILINGUAL SUPPORT TO MULTILINGUAL SUPPORT
            @spt_qst: support question or question cluster
            @qry_lang: the language(s) of the query meta set
            @k_qry: the number of questions to be retrieved for the given support
        """
        self.spt_qst = spt_qst
        self.spt_qst_lang = spt_qst_lang ## Assuming the question can be presented in only one language
        self.spt_cand_langs = spt_cand_langs
        self.qry_qst_lang = {0: qry1_qst_lang, 1: qry2_qst_lang}
        self.qry_cand_langs = {0: qry1_cand_langs, 1: qry2_cand_langs}
        self.n_neg_eg = meta_learn_split_config["n_neg_eg"]
        self.question_set = question_set
        self.candidate_set = candidate_set
        # Get randomly shuffled candidate_set
        self.candidate_xling = candidate_set.by_xling_id
        self.meta_learn_split_config = meta_learn_split_config
        self.tokenizer = tokenizer
        self.top_results = top_results
        self.mode = meta_learn_split_config["mode"]
        self.neg_sampling_approach = meta_learn_split_config["neg_sampling_approach"]
        self.neg_trip_margin = 1.0
        self.create_spt_meta_set()
        self.qry = {i: [] for i in range(2)} # List of MetaSet(s)
        self.qry_features = {i: [] for i in range(2)} # List of MetaSet(s)
        if self.mode == "similar":
            for i in range(2):
                self.create_qry_meta_sets(i) # could be one qry set or multiple qry sets depending on how many languages are included
        else:
            for i in range(2):
                self.create_rnd_qry_meta_sets(i)

    def pick_rnd_qst(self, qry_qst_lang, rank):
        questions_qry_lang = self.question_set.by_lang[qry_qst_lang[rank]]
        random.shuffle(questions_qry_lang)

        return questions_qry_lang[0]

    def create_spt_meta_set(self):
        positive_candidates = self.candidate_set.by_xling_id_get_langs(self.spt_qst.xling_id, self.spt_cand_langs.split(","))
        d_list = [np.mean(positive_candidates[i].encoding - self.spt_qst.encoding) for i in range(len(positive_candidates))]
        self.spt = MetaSet(self.spt_qst,
                           positive_candidates=positive_candidates,
                           negative_candidates=self.get_negative_candidates(self.spt_qst, self.spt_cand_langs.split(","), np.mean(d_list)))
        self.spt_features = convert_features(self.meta_learn_split_config, self.spt, self.tokenizer)

    def create_qry_meta_sets(self, rank):
        """ Given the support meta set, we compute the semantic similarity between the question and other questions in self.qry_lang.
        If self.spt_lang == self.qry_lang, then find a question set different .
        To start with we just pick the translation of the question to different languages in self.qry_lang.
        @rank: whether to do the first or the second query
        """
        if self.qry_qst_lang[rank] != self.spt_qst_lang:
            for equivalent_question in self.question_set.by_xling_id[self.spt_qst.xling_id]:
                # print("question:", question.language, " self.qry_qst_lang[rank]=", self.qry_qst_lang[rank], " self.spt_qst.language:", self.spt_qst.language, " condition:", question.language == self.qry_qst_lang[rank] and question.language != self.spt_qst.language)
                if equivalent_question.language == self.qry_qst_lang[rank]:
                    positive_candidates = self.candidate_set.by_xling_id_get_langs(equivalent_question.xling_id, self.qry_cand_langs[rank].split(","))
                    d_list = [np.mean(positive_candidates[i].encoding - equivalent_question.encoding) for i in range(len(positive_candidates))]
                    meta_set = MetaSet(equivalent_question,
                                       positive_candidates=positive_candidates,
                                       negative_candidates=self.get_negative_candidates(equivalent_question, self.qry_cand_langs[rank].split(","), np.mean(d_list)))
                    self.qry[rank].append(meta_set)
                    self.qry_features[rank].append(convert_features(self.meta_learn_split_config, meta_set, self.tokenizer))

        # Find most similar questions in the language of interest
        for xling_id in self.top_results["xling_ids"][self.spt_qst_lang][self.qry_qst_lang[rank]][self.spt_qst.xling_id]:
            if xling_id != self.spt_qst.xling_id:
                for similar_question in self.question_set.by_xling_id[xling_id]:
                    if similar_question.language == self.qry_qst_lang[rank]:
                        positive_candidates = self.candidate_set.by_xling_id_get_langs(xling_id, self.qry_cand_langs[rank].split(","))
                        d_list = [np.mean(positive_candidates[i].encoding - similar_question.encoding) for i in range(len(positive_candidates))]
                        meta_set = MetaSet(similar_question,
                                           positive_candidates=positive_candidates,
                                           negative_candidates=self.get_negative_candidates(similar_question, self.qry_cand_langs[rank].split(","), np.mean(d_list)))

                        self.qry[rank].append(meta_set)
                        self.qry_features[rank].append(convert_features(self.meta_learn_split_config, meta_set, self.tokenizer))



        ## find similar question rather than equivalent questions mapped by language
        ### Get the list of question embeddings from a similarity measure
        # for question in self.question_set[self.qry_qst_lang[rank]]:
        #     question.encoding

    def create_rnd_qry_meta_sets(self, rank):
        random_question = self.pick_rnd_qst(self.qry_qst_lang, rank)
        positive_candidates = self.candidate_set.by_xling_id_get_langs(random_question.xling_id, self.qry_cand_langs[rank].split(","))
        d_list = [np.mean(positive_candidates[i].encoding - random_question.encoding) for i in range(len(positive_candidates))]
        meta_set = MetaSet(random_question,
                           positive_candidates=positive_candidates,
                           negative_candidates=self.get_negative_candidates(random_question, self.qry_cand_langs[rank].split(","), np.mean(d_list)))
        self.qry[rank].append(meta_set)
        self.qry_features[rank].append(convert_features(self.meta_learn_split_config, meta_set, self.tokenizer))

    def shuffle_candidate_xling(self):
        candidates = list(self.candidate_xling.items())
        random.shuffle(candidates)
        self.candidate_xling = collections.OrderedDict(candidates)

    def get_negative_candidates(self, question, langs, positive_dist):
        negative_candidates = []

        # Shuffling each time we build negative candidates to ensure randomization and variety in the candidate set
        # TODO different techniques for picking negative examples
        self.shuffle_candidate_xling()

        for xling, candidates in self.candidate_xling.items():
            if xling != question.xling_id:
                # print("*********** xling:", xling, " len(candidates):", len(candidates))
                if len(negative_candidates) == self.n_neg_eg:
                    break
                for candidate in candidates:
                    # print("xling:", xling, " question.xling_id:", question.xling_id, " candidate.language:", candidate.language, "langs:", langs)
                    if len(negative_candidates) == self.n_neg_eg:
                        break
                    if candidate.language in langs:
                        neg_dist = np.mean(candidate.encoding - question.encoding)
                        if self.neg_sampling_approach == "random":
                            negative_candidates.append(candidate)
                        else:
                            if self.neg_sampling_approach == "hard":
                                if neg_dist < positive_dist: # Hard triplet
                                    negative_candidates.append(candidate)
                            elif self.neg_sampling_approach == "semi-hard":
                                if neg_dist < positive_dist + self.neg_trip_margin:
                                    negative_candidates.append(candidate)
                                    print("Semi-hard triplet")
                            elif self.neg_sampling_approach == "easy":
                                if positive_dist + self.neg_trip_margin < neg_dist:
                                    negative_candidates.append(candidate)
                                    print("Semi-hard triplet")
                # else:
                #     # Continue if the inner loop wasn't broken.
                #     continue

                # # Inner loop was broken, break the outer.
                # break

        assert len(negative_candidates) == min(len(negative_candidates), self.n_neg_eg)
        return negative_candidates


class MetaDataset(object):
    """ This class holds sets of tasks for different meta-datasets (train, dev, and test)"""
    def __init__(self, n_tasks, lang_pairs, question_set, candidate_set, split, meta_learn_split_config, tokenizer, top_results): # n_train_tasks, n_valid_tasks, n_test_tasks, , train_lang_pairs, valid_lang_pairs, test_lang_pairs):
        self.n_tasks = n_tasks
        self.lang_pairs = lang_pairs
        self.question_set = question_set
        self.candidate_set = candidate_set
        self.meta_learn_split_config = meta_learn_split_config
        self.tokenizer = tokenizer
        self.split = split
        self.meta_tasks = []
        self.top_results = top_results

        self.create_meta_tasks()

    def pick_rnd_qst(self, spt_qst_lang):
        questions_spt_lang = self.question_set.by_lang[spt_qst_lang]
        random.shuffle(questions_spt_lang)

        return questions_spt_lang[0]

    def create_meta_tasks(self):
        # Compute the number of meta-tasks by language pair For a number of meta-tasks (the number of meta-tasks can be less than the number of available question so we need to come up with a combination)
        counter = {item: 0 for item in range(9)}
        for _, question_list in self.question_set.by_xling_id.items():
            count = 0
            for _ in question_list:
                count += 1
            counter.update({count:counter[count]+1})

        n_tasks_per_spt_lang = self.n_tasks // len(self.lang_pairs)

        for lang_pair in self.lang_pairs.split("|"):
            pbar = tqdm(range(n_tasks_per_spt_lang))
            pbar.set_description("Processing %s" % lang_pair)
            for _ in pbar:
                spt_langs, qry1_langs, qry2_langs = lang_pair.split("-")

                spt_qst_lang, spt_cand_langs = spt_langs.split("_")
                qry1_qst_lang, qry1_cand_langs = qry1_langs.split("_")
                qry2_qst_lang, qry2_cand_langs = qry2_langs.split("_")

                # Pick a question at random using spt_qst_lang
                spt_qst = self.pick_rnd_qst(spt_qst_lang)
                # Construct the support and query sets

                meta_task = MetaTaskMerged(spt_qst,
                                           spt_qst_lang,
                                           spt_cand_langs,
                                           qry1_qst_lang,
                                           qry1_cand_langs,
                                           qry2_qst_lang,
                                           qry2_cand_langs,
                                           self.question_set,
                                           self.candidate_set,
                                           self.meta_learn_split_config,
                                           self.tokenizer,
                                           self.top_results)

                self.meta_tasks.append(meta_task)

