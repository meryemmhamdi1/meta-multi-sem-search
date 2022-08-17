import collections
import os
import numpy as np
import torch
from tqdm import tqdm
from multi_meta_ssd.processors.downstream.sb_sed import infer_sentence_breaks
from multi_meta_ssd.log import *

class Question():
    """Question class holding information about a single question.

    Attributes:
    question (str): The text of the question.
    xling_id (str): An id that identifies the same QA pair across different
        languages.
    uid (str): A unique identifier for each question.
    language (str): The language code of the question.
    encoding (np.array): The encoding of the question.
    """

    def __init__(self, question, xling_id, lang, encoding):
        self.question = question
        self.xling_id = xling_id
        self.uid = "{}_{}".format(xling_id, lang)
        self.language = lang
        self.encoding = encoding

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if isinstance(other, Question):
            return self.uid == other.uid
        return False

    def __str__(self):
        return "Question: uid ({}), lang ({}), xling_id ({})".format(
            self.uid, self.language, self.xling_id)


class Candidate():
    """Candidate class holding info about a single answer candidate.

    Attributes:
        uid (str): A unique identifier for each candidate answer.
        sentence (str): The text of the candidate answer sentence.
        context (str): The text of the paragraph of the candidate answer sentence.
        language (str): The language code of the candidate answer.
        encoding (np.array): The encoding of the candidate answer.
    """

    def __init__(self, sentence, context, lang, context_id, sent_pos, encoding):
        self.uid = "{}_{}".format(context_id, sent_pos)
        self.sentence = sentence
        self.context = context
        self.language = lang
        self.encoding = encoding

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if isinstance(other, Candidate):
            return self.uid == other.uid
        return False

    def __str__(self):
        return "Candidate: uid ({}), lang ({})".format(self.uid, self.language)


class QuestionSet():
    """A set of questions with several mappings that track relations between them.

    Attributes:
    by_xling_id: A mapping of xling_id to a list of Question objects with that
        id.
    by_lang: A mapping of language code to a list of Question objects in that
        language.
    by_uid: An OrderedDict mapping uid to Question.
    pos: A dictionary that maps a Question object to its position in the
        OrderedDict `by_uid`.
    """

    def __init__(self):
        self.by_uid = collections.OrderedDict()
        self.by_xling_id = collections.defaultdict(list)
        self.by_lang = collections.defaultdict(list)
        self.pos = {}

    def add(self, question):
        self.pos[question] = len(self.by_uid)
        assert question.uid not in self.by_uid
        self.by_uid[question.uid] = question
        self.by_lang[question.language].append(question)
        self.by_xling_id[question.xling_id].append(question)

    def as_list(self):
        return list(self.by_uid.values())

    def filter_by_langs(self, langs):
        new_question_set = QuestionSet()
        for q in self.as_list():
            if q.language in langs:
                new_question_set.add(q)
        return new_question_set

    def get_encodings(self):
        return np.concatenate([
            np.expand_dims(q.encoding, 0) for q in self.as_list()])


class CandidateSet():
    """A set of candidates with several mappings that track relations.

    Attributes:
        by_xling_id: A mapping of xling_id to a list of Candidate objects with that
        id.
        by_lang: A mapping of language code to a list of Candidate objects in that
        language.
        by_uid: An OrderedDict mapping uid to Candidate.
        pos: A dictionary that maps a Candidate object to its position in the
        OrderedDict `by_uid`.
    """

    def __init__(self):
        self.by_uid = collections.OrderedDict()
        self.by_xling_id = collections.defaultdict(list)
        self.by_lang = collections.defaultdict(list)
        self.pos = {}

    def add_or_retrieve_candidate(self, candidate):
        # Don't add candidates that already exist, just return them.
        if candidate.uid in self.by_uid:
            return self.by_uid[candidate.uid]
        self.pos[candidate] = len(self.by_uid)
        self.by_uid[candidate.uid] = candidate
        self.by_lang[candidate.language].append(candidate)
        return candidate

    def update_xling_id(self, candidate, xling_id):
        """Given an already created candidate, update the by_xling_id mapping."""
        assert candidate.uid in self.by_uid
        assert xling_id
        assert candidate not in self.by_xling_id[xling_id], (
            "Candidate {} already updated xling_id {}".format(
                candidate.uid, xling_id))
        self.by_xling_id[xling_id].append(candidate)

    def as_list(self):
        return list(self.by_uid.values())

    def filter_by_langs(self, langs):
        """Generates new candidate set of candidates with desired languages."""
        new_candidate_set = CandidateSet()
        for c in self.as_list():
            if c.language in langs:
                new_candidate_set.add_or_retrieve_candidate(c)
        # Although we've added all the relevant candidates to the new candidate_set,
        # we need to update the by_xling_id dictionary in the new candidate_set.
        for xling_id, candidates in self.by_xling_id.items():
            for c in candidates:
                if c.uid in new_candidate_set.by_uid:
                    new_candidate_set.update_xling_id(c, xling_id)
        return new_candidate_set

    def get_encodings(self):
        return np.concatenate([
            np.expand_dims(c.encoding, 0) for c in self.as_list()])

    def by_xling_id_get_langs(self, xling_id, langs):
        """Gets answers with a given xling_id according to langs.

        Args:
        xling_id: The desired xling_id of the answers.
        langs: The desired languages of the answers.

        Returns:
        A list of answers (filtered_answers) with the desired xling_id such that
        filtered_answers[idx].language == langs[idx]. If no answer exists for a
        language lang[idx], then filtered_answers[idx] = None.
        """
        all_answers = self.by_xling_id[xling_id]
        filtered_answers = []
        for lang in langs:
            selected_answer = None
            for a in all_answers:
                if a.language == lang:
                    selected_answer = a
            filtered_answers.append(selected_answer)
        return filtered_answers

def convert_features(meta_learn_split_config, meta_set, tokenizer):
     ## Define batches of meta-tasks
    q_features = tokenizer.encode_plus(meta_set.question_cluster.question,
                                       max_length=meta_learn_split_config["max_query_length"],#meta_learn_split_config["max_query_length"],
                                       pad_to_max_length=True,
                                       return_token_type_ids=True)

    q_features_new = {"input_ids": torch.unsqueeze(torch.tensor(q_features["input_ids"], dtype=torch.long), dim=0),
                      "attention_mask": torch.unsqueeze(torch.tensor(q_features["attention_mask"], dtype=torch.long), dim=0),
                      "token_type_ids": torch.unsqueeze(torch.tensor(q_features["token_type_ids"], dtype=torch.long), dim=0)
                     }

    # Note, this concatenation of sentence and paragraph matches the current preprocessing in evaluate_retrieval.py.
    # TODO(nconstant): Adjust preprocessing here and in evaluate_retrieval.py to pass sentence and paragraph as separate inputs to encode_plus().
    a_features = [tokenizer.encode_plus((candidate.sentence + candidate.context).replace("\n", ""),
                                        max_length=meta_learn_split_config["max_answer_length"],#meta_learn_split_config["max_answer_length"],
                                        pad_to_max_length=True,
                                        return_token_type_ids=True)
                  for candidate in meta_set.positive_candidates]

    input_ids_total = []
    attention_mask_total = []
    token_type_ids_total = []
    for i in range(len(a_features)):
        input_ids_total.append(a_features[i]["input_ids"])
        attention_mask_total.append(a_features[i]["attention_mask"])
        token_type_ids_total.append(a_features[i]["token_type_ids"])

    a_features_new = {"input_ids": torch.tensor(input_ids_total, dtype=torch.long),
                      "attention_mask": torch.tensor(attention_mask_total, dtype=torch.long),
                      "token_type_ids": torch.tensor(token_type_ids_total, dtype=torch.long)
                     }

    # Note, this concatenation of sentence and paragraph matches the current preprocessing in evaluate_retrieval.py.
    # TODO(nconstant): Adjust preprocessing here and in evaluate_retrieval.py to pass sentence and paragraph as separate
    # inputs to encode_plus().
    n_features = [tokenizer.encode_plus((candidate.sentence + candidate.context).replace("\n", ""),
                                        max_length=meta_learn_split_config["max_answer_length"],#meta_learn_config["max_answer_length"],
                                        pad_to_max_length=True,
                                        return_token_type_ids=True)
                  for candidate in meta_set.negative_candidates]

    input_ids_total = []
    attention_mask_total = []
    token_type_ids_total = []
    for i in range(len(n_features)):
        input_ids_total.append(n_features[i]["input_ids"])
        attention_mask_total.append(n_features[i]["attention_mask"])
        token_type_ids_total.append(n_features[i]["token_type_ids"])

    n_features_new = {"input_ids": torch.tensor(input_ids_total, dtype=torch.long),
                      "attention_mask": torch.tensor(attention_mask_total, dtype=torch.long),
                      "token_type_ids": torch.tensor(token_type_ids_total, dtype=torch.long)
                     }

    # return {"q_features": q_features_new, "a_features": a_features_new, "n_features": n_features_new}
    return {"q_input_ids": q_features_new["input_ids"], "q_attention_mask": q_features_new["attention_mask"], "q_token_type_ids": q_features_new["token_type_ids"], \
           "a_input_ids": a_features_new["input_ids"], "a_attention_mask": a_features_new["attention_mask"], "a_token_type_ids": a_features_new["token_type_ids"], \
           "n_input_ids": n_features_new["input_ids"], "n_attention_mask": n_features_new["attention_mask"], "n_token_type_ids": n_features_new["token_type_ids"]}

def set_encodings(questions, candidates,
                question_encoder, response_encoder):
    """Set encoding fields within questions and candidates.

    Args:
        questions: List of Question objects.
        candidates: List of Candidate objects.
        question_encoder: The question encoder function, mapping from
        Sequence[Question] to an np.ndarray matrix holding question encodings.
        response_encoder: The candidate answer encoder function, mapping from
        Sequence[Candidate] to an np.ndarray matrix holding answer encodings.
    """
    for i, chunk in enumerate(chunks(questions, 100)):
        question_strs = [q.question for q in chunk]
        question_encodings = np.array(question_encoder(question_strs))

        for j in range(question_encodings.shape[0]):
            questions[i * 100 + j].encoding = question_encodings[j]
        if i % 50 == 0:
            logger.info("Questions: encoded %d of %d ..." % i * 100, len(questions))

    for i, chunk in enumerate(chunks(candidates, 50)):
        candidate_strs = [(c.sentence, c.context) for c in chunk]
        candidate_encodings = np.array(response_encoder(candidate_strs))
        for j in range(candidate_encodings.shape[0]):
            candidates[i * 50 + j].encoding = candidate_encodings[j]
        if i % 100 == 0:
            logger.info("Candidates: encoded %s of %s..." % i * 50, len(candidates))

def encode_sentence(sentence, tokenizer, device, max_length, base_model):
    q_features = tokenizer.encode_plus(sentence,
                                       max_length=max_length,#max_length,
                                       pad_to_max_length=True,
                                       return_token_type_ids=True)

    q_features_new = {"input_ids": torch.unsqueeze(torch.tensor(q_features["input_ids"], dtype=torch.long), dim=0),
                      "attention_mask": torch.unsqueeze(torch.tensor(q_features["attention_mask"], dtype=torch.long), dim=0),
                      "token_type_ids": torch.unsqueeze(torch.tensor(q_features["token_type_ids"], dtype=torch.long), dim=0)}

    # print("Device:", device)
    # available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    # print("available_gpus:", available_gpus)

    q_features_new = {k:v.to(device) for k, v in q_features_new.items()}

    base_model = base_model.to(device)

    encoding = base_model(q_input_ids=q_features_new["input_ids"],
                          q_attention_mask=q_features_new["attention_mask"],
                          q_token_type_ids=q_features_new["token_type_ids"],
                          inference=True)

    return encoding



def load_data(squad_per_lang, base_model, tokenizer, device, meta_learn_split_config):
    """Load and encode SQuAD-format data from parsed JSON documents.

    Args:
    squad_per_lang: A map from language code to SQuAD-format data, as returned
        by json.load(...).

    Returns:
    All questions and candidates.
    """
    question_set = QuestionSet()
    candidate_set = CandidateSet()

    for lang, squad in squad_per_lang.items():
        counter_qst = 0
        for question, answer, context, context_sentences, xling_id, context_id in (generate_examples(squad, lang)):
            counter_qst += 1
            q_encoding = None
            if base_model:
                # q_encoding = sim_model.encode(question)
                q_encoding = encode_sentence(question, tokenizer, device, meta_learn_split_config["max_query_length"], base_model)
            question = Question(question, xling_id, lang, q_encoding)
            question_set.add(question)
            assert answer in context_sentences, ("answer doesn't appear in context_sentences")
            for sent_pos, sentence in enumerate(context_sentences):
                c_encoding = None
                if base_model:
                    # c_encoding = sim_model.encode((sentence + context).replace("\n", ""))
                    c_encoding = encode_sentence((sentence + context).replace("\n", ""), tokenizer, device, meta_learn_split_config["max_answer_length"], base_model)
                candidate = Candidate(sentence, context, lang, context_id, sent_pos, c_encoding)
                candidate = candidate_set.add_or_retrieve_candidate(candidate)
                if sentence == answer:
                    candidate_set.update_xling_id(candidate, xling_id)
    print("Totals across languages: questions={}, candidates={}".format(
        len(question_set.as_list()), len(candidate_set.as_list())))

    return question_set, candidate_set


def generate_examples(data, language):
    """Generates SQuAD examples.

    Args:
        data: (object) An object returned by json.load(...)
        language: (str) Language code for the QA pairs in the dataset.

    Yields:
        question: (str) The question.
        answer: (str) The answer sentence.
        context (str) The answer context, as raw text.
        context_sentences: (List[str]) The answer context as a list of sentences.
        xling_id: (str) The unique question ID, e.g. "56beb4343aeaaa14008c925b".
    """
    # Loop through the SQuAD-format data and perform the conversion. The basic
    # outline, which is mirrored in the for-loops below, is as follows:
    #
    # data ---< Passage ---< Paragraph ---< Question
    #
    # The loop below goes through each paragraph and yields each question
    # along with the enclosing sentence of its answer and its context.
    num_passages = 0
    num_paragraphs = 0
    num_questions = 0
    count_skipped_qst = 0
    titles = []
    for passage in data["data"]:
        title = passage["title"]
        if title not in titles:
            titles.append(title)

    for passage in tqdm(data["data"]):
        num_passages += 1
        for paragraph in passage["paragraphs"]:
            num_paragraphs += 1
            context_id = "{}_{}".format(language, num_paragraphs)
            context = paragraph["context"]
            if language == "en":
                sentence_breaks = list(infer_sentence_breaks(context))
            else:
                sentence_breaks = paragraph["sentence_breaks"]
            context_sentences = [
                str(context[start:end]) for (start, end) in sentence_breaks]
            for qas in paragraph["qas"]:
                num_questions += 1
                # If there are more than one answer per question then skip that question
                if len(qas["answers"]) > 1:
                    count_skipped_qst += 1
                    break
                answer = qas["answers"][0]
                answer_start = answer["answer_start"]
                # Map the answer fragment back to its enclosing sentence, or to None
                # if there is no single sentence containing the answer.
                sentence = None
                for start, end in sentence_breaks:
                    if start <= answer_start < end:
                        sentence = context[start:end]
                        break

                if sentence:
                    assert sentence in context_sentences
                    yield (qas["question"], str(sentence), context, context_sentences,
                           qas["id"], context_id)
                else:
                    count_skipped_qst += 1
                # else:
                #     print("Warning: Skipped an example where the answer start was not "
                #         "contained within any sentence. This is likely due to a "
                #         "sentence breaking error.")

    logger.info("Language %s: Processed %s passages, %s titles, %s paragraphs, %s questions, and %s skipped."
                % (language, num_passages,  len(titles), num_paragraphs, num_questions, count_skipped_qst))


def save_encodings(output_dir, questions, candidates):
    """Outputs encodings to files in output directory."""
    def write_output(output_dir, prefix, objects):
        encodings = np.array([x.encoding for x in objects])
        ids = [x.uid for x in objects]
        with open(os.path.join(output_dir,
                            "{}_encodings.npz".format(prefix)), "wb") as f:
            np.save(f, encodings)
        with open(os.path.join(output_dir,
                            "{}_uids.txt".format(prefix)), "w") as f:
            for id_ in ids:
                f.write(id_ + "\n")

    write_output(output_dir, "question", questions)
    write_output(output_dir, "candidate", candidates)


def load_encodings(output_dir, question_set, candidate_set):
    """Loads encodings into objects from output directory."""
    def read_input(output_dir, prefix, uid_to_obj):
        with open(os.path.join(output_dir,
                            "{}_encodings.npz".format(prefix)), "rb") as f:
            encodings = np.load(f)
        with open(os.path.join(output_dir,
                            "{}_uids.txt".format(prefix)), "r") as f:
            ids = f.read().splitlines()
        for i, id_ in enumerate(ids):
            uid_to_obj[id_].encoding = encodings[i]
    read_input(output_dir, "question", question_set.by_uid)
    read_input(output_dir, "candidate", candidate_set.by_uid)


def chunks(lst, size):
    """Yield successive chunks of a given size from a list."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def average_precision_at_k(targets, ranked_predictions, k=None):
    """Computes AP@k given targets and ranked predictions."""
    if k:
        ranked_predictions = ranked_predictions[:k]
    score = 0.0
    hits = 0.0
    for i, pred in enumerate(ranked_predictions):
        if pred in targets and pred not in ranked_predictions[:i]:
            hits += 1.0
            score += hits / (i + 1.0)
    divisor = min(len(targets), k) if k else len(targets)
    return score / divisor


def mean_avg_prec_at_k(question_set, candidate_set, k=None):
    """Computes mAP@k on question_set and candidate_set with encodings."""
    # TODO(umaroy): add test for this method on a known set of encodings.
    # Current run_xreqa_eval.sh with X_Y encodings generates mAP of 0.628.
    all_questions = question_set.as_list()
    all_candidates = candidate_set.as_list()
    for embedding_type in ['sentences_and_contexts']:
        candidate_matrix = np.concatenate(
            [np.expand_dims(i.encoding[embedding_type], 0) for i in all_candidates],
            axis=0)

        ap_scores = []
        for q in all_questions:
            question_vec = np.expand_dims(q.encoding, 0)
            scores = question_vec.dot(candidate_matrix.T)
            y_true = np.zeros(scores.shape[1])
            all_correct_cands = set(candidate_set.by_xling_id[q.xling_id])
            for ans in all_correct_cands:
                y_true[candidate_set.pos[ans]] = 1
            ap_scores.append(average_precision_at_k(np.where(y_true == 1)[0], np.squeeze(scores).argsort()[::-1], k))
        logger.info(" %s : %s" % embedding_type, str(np.mean(ap_scores)))

def mean_avg_prec_at_k_meta(question_list, question_vecs, all_candidates_list, all_candidate_vecs, k=None):
    """Computes mAP@k on question_set and candidate_set with encodings."""
    # TODO(umaroy): add test for this method on a known set of encodings.
    # Current run_xreqa_eval.sh with X_Y encodings generates mAP of 0.628.
    print("len(question_list):", len(question_list), " len(question_vecs):", len(question_vecs))
    ap_scores = []
    for i, _ in enumerate(question_list):
        # print("question_vecs[i]:", question_vecs[i])
        question_vec = np.expand_dims(question_vecs[i], 0)
        scores = question_vec.dot(all_candidate_vecs.T)
        y_true = np.zeros(scores.shape[1])
        y_true[0] = 1
        ap_scores.append(average_precision_at_k(np.where(y_true == 1)[0], np.squeeze(scores).argsort()[::-1], k))

    return np.mean(ap_scores)