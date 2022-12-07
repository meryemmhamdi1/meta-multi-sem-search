import collections
import os
import numpy as np
import torch
from tqdm import tqdm

class Question():
    """Question class holding information about a single question.

    Attributes:
    question (str): The text of the question.
    xling_id (str): An id that identifies the same QA pair across different
        languages.
    uid (str): A unique identifier for each question.
    language (str): The language code of the question.
    encoding (np.array): The encoding of the question.
    trans (str): The translation of the question to English.
    """

    def __init__(self, question, xling_id, lang, encoding, trans=None):
        self.question = question
        self.xling_id = xling_id
        self.uid = "{}_{}".format(xling_id, lang)
        self.language = lang
        self.encoding = encoding
        self.trans = trans

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if isinstance(other, Question):
            return self.uid == other.uid
        return False

    def add_trans(self, trans):
        self.trans = trans

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
        trans_sent (str): The translation of the candidate sentence to English.
        trans_context (str): The translation of the candidate context to English.
    """

    def __init__(self, sentence, context, lang, uid, encoding, trans_sent=None, trans_context=None):
        # self.uid = "{}_{}".format(context_id, sent_pos)
        self.uid = uid
        self.sentence = sentence
        self.context = context
        self.language = lang
        self.encoding = encoding
        self.trans_sent = trans_sent
        self.trans_context = trans_context

    def __hash__(self):
        return hash(self.uid)

    def add_trans(self, trans_sent, trans_context):
        self.trans_sent = trans_sent
        self.trans_context = trans_context

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

def update_data(old_question_set, old_candidate_set):
    """Load and encode SQuAD-format data from parsed JSON documents.

    Args:
    squad_per_lang: A map from language code to SQuAD-format data, as returned
        by json.load(...).

    Returns:
    All questions and candidates.
    """
    question_set = QuestionSet()
    for uid, old_question in old_question_set.by_uid.items():
        # translate question to English
        question = Question(old_question.question, \
                            old_question.xling_id, \
                            old_question.language, \
                            old_question.encoding,
                            trans)

        question_set.add(question)
    
    candidate_set = CandidateSet()
    for xling_id, old_candidate in old_candidate_set.by_xling_id.items():
        context_id, sent_pos = old_candidate.uid.split("_")
        # translate candidate sentence to English
        # translate candidate context to English
        candidate = Candidate(old_candidate.sentence, \
                              old_candidate.context, \
                              old_candidate.language, \
                              context_id, \
                              sent_pos, \
                              candidate.encoding)

        candidate = candidate_set.add_or_retrieve_candidate(candidate)

        candidate_set.update_xling_id(candidate, xling_id)

    return question_set, candidate_set