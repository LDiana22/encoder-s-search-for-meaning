# -*- coding: utf-8 -*-

"""# Constants"""

# %% [code]
# -*- coding: utf-8 -*-
from abc import ABC
import argparse
from collections import OrderedDict 
from collections import ChainMap
from contextlib import redirect_stdout
import copy
import glob
import io
import itertools
import matplotlib.pyplot as plt
import os
import os.path
import pickle
import random
import re
import spacy
import string
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datetime import datetime

import torch

from torch import nn
from torch.utils import data as utils

import torch.nn.functional as F

import torch.optim as optim

from torchtext import datasets
from torchtext import data as data
from torchtext.vocab import GloVe
from torchtext.data import Pipeline

import numpy as np

import yake
from summa import keywords
from rake_nltk import Rake

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.enabled = False 
torch.cuda.manual_seed_all(0)

torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


VECTOR_CACHE = "../.vector_cache"
UCI_PATH = "../.data/uci"
IMDB_PATH = "../.data/imdb/aclImdb"
PREFIX_DIR = "experiments/dict_acquisition"

CONFIG = {
    "toy_data": False, # load only a small subset
    "cuda": False,
    "restore_checkpoint" : False,
    "dict_checkpoint_file": None,

    "max_words_dict": 300,
    "phrase_len":5,

    "prefix_dir" : PREFIX_DIR,

    "dirs": {
        "metrics": "metrics",
        "checkpoint": "snapshot",
        "dictionary": "dictionaries",
        },

    "max_vocab_size": 100000,
    "emb_dim": 300,
    "batch_size": 32,
    "load_dictionary": False,
    "filterpolarity": False
}


# %% [code]
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
DATE_REGEXP = '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}'

def remove_br_tag(token):
    return re.sub(r"br|(/><.*)|(</?(.*)/?>)|(<?(.*)/?>)|(<?/?(.*?)/?>?)", "", token)

def remove_non_alpha(token):
    if token.isalpha():
        return token
    return ""

def preprocess(token):
    token = remove_br_tag(token)
    token = remove_non_alpha(token)
    return token

preprocess_pipeline = Pipeline(preprocess)

class IMDBDataset:

  def __init__(self, args, max_length=250):
    self.args = args
    self.max_sent_len = 800
    TEXT = data.Field(lower=True, 
                      include_lengths=True,
                      sequential=True,
                      tokenize='spacy',
                      # use_vocab=False,
                      preprocessing=preprocess_pipeline)
    LABEL = data.LabelField(dtype = torch.float)#, use_vocab=False)
    print("Loading the IMDB dataset...")
    # self.train_data = self._load_data(TEXT, LABEL, "../.data/imdb/aclImdb", "train")
    # self.test_data = self._load_data(TEXT, LABEL, "../.data/imdb/aclImdb", "test")
    self.train_data = self._load_data(TEXT, LABEL, IMDB_PATH, "train")
    self.test_data = self._load_data(TEXT, LABEL, IMDB_PATH, "test")
#     self.train_data, self.test_data =  datasets.IMDB.splits(TEXT, LABEL)
    self.train_data, self.valid_data = self.train_data.split(random_state=random.getstate())

    # start = datetime.now()
    # formated_date = start.strftime(DATE_FORMAT)
    # with open(f"train-imdb-{formated_date}", "w") as f:
    #     f.write("\n".join([f"{' '.join(self.train_data[i].text)} ~  {self.train_data[i].label}" for i in range(len(self.train_data))]))
    # with open(f"val-imdb-{formated_date}", "w") as f:
    #     f.write("\n".join([f"{' '.join(self.valid_data[i].text)} ~  {self.valid_data[i].label}" for i in range(len(self.valid_data))]))
    # with open(f"test-imdb-{formated_date}", "w") as f:
    #     f.write("\n".join([f"{' '.join(self.test_data[i].text)} ~  {self.test_data[i].label}" for i in range(len(self.test_data))]))

    print("IMDB...")
    print(f"Train {len(self.train_data)}")
    print(f"Valid {len(self.valid_data)}")
    print(f"Test {len(self.test_data)}")
    TEXT.build_vocab(self.train_data,
                 max_size = args["max_vocab_size"])
    LABEL.build_vocab(self.train_data)

    self.TEXT = TEXT
    self.device = torch.device('cuda' if args["cuda"] else 'cpu')

  def _load_data(self, text_field, label_field, path, data_type="train"):
    fields = [('text', text_field), ('label', label_field)]
    examples = []
    path = os.path.join(path, data_type)
    for label in ['pos', 'neg']:
        print(f"{os.path.join(path, label, f'{label}.txt')}")
        fname = os.path.join(path, label, f'{label}.txt')
        with io.open(fname, 'r', encoding='utf-8', errors='replace') as f:
            for text in f:
                if text != '\n':
                    sent_len = len(text.split())
                    if sent_len > self.max_sent_len:
                        self.max_sent_len = sent_len
                    examples.append(data.Example.fromlist([text, label], fields))
                if self.args["toy_data"] and (len(examples)==50 or len(examples)==100):
                    break

    print(f'Loaded {len(examples)}')
    fields = dict(fields)
    # Unpack field tuples
    for n, f in list(fields.items()):
        if isinstance(n, tuple):
            fields.update(zip(n, f))
            del fields[n]
    return data.Dataset(examples, fields)

  def iterators(self):
    """
      Returns train_iterator, valid_iterator, test_iterator
    """
    return data.BucketIterator.splits(
      (self.train_data, self.valid_data, self.test_data), 
      batch_size = self.args["batch_size"],
      sort_within_batch = True,
      sort_key=lambda x: len(x.text),
      device = self.device)

  def training(self):
    return self.train_data

  def get_training_corpus(self):
    self.corpus = {"pos":[], "neg":[]}
    self.corpus["pos"] = [" ".join(example.text) for example in self.train_data if example.label == "pos"]
    self.corpus["neg"] = [" ".join(example.text) for example in self.train_data if example.label == "neg"]
    return self.corpus

  def dev(self):
    return self.valid_data

  def test(self):
    return self.test_data 

  def override(self, args):
    self.args.update(args)
    return self



###############################################################################
################################# ABSTRACT DICT ###############################
###############################################################################

class AbstractDictionary:
  def __init__(self, id, dataset, args):
    """
    A dictionary consists of a list of entries per each class.
    For a corpus dictionary, there is only one "dummy class" considered.
    """
    self.id = id
    self.dataset = dataset
    self.args = args 
    self.path = os.path.join(self.args["prefix_dir"], self.args["dirs"]["dictionary"], id)
    self.metrics = {}
    if not os.path.isdir(self.path):
        os.makedirs(self.path)
    if args["load_dictionary"]:
        self.dictionary = self.load_dict(args["dict_checkpoint"])


  def load_dict(self, file):
    return pickle.load(open(file, "rb"))

  def filter_by_sentiment_polarity(self, phrases, metric="compound", threshold=0.5):
    """
    in: list of phrases  [(score, "phrase")]
    metric: compound, pos, neg
    out: list of fitered phrases
    """
    sentiment = SentimentIntensityAnalyzer()
    return [(score, phrase) for score, phrase in phrases if abs(sentiment.polarity_scores(phrase)[metric])>threshold]


  def filter_phrases_max_words_by_occurence(self, phrases, corpus, max_phrases):
    """
    phrases: list of phrases
    max_words: maximum number of words per phrase
    corpus: used for phrase frequency
    max_phrases: maximum number of phrases
    """
    result = {}
    count = 0
    for i in range(len(phrases)):
      phrase = " ".join(phrases[i].split()[:self.max_words])
      freq = corpus.count(phrase)
      if freq > 0:
        result.update({phrase:freq})

    return result

  def _save_dict(self):
    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    file = os.path.join(self.path, f"dictionary-{formated_date}.h5")
    with open(file, "wb") as f: 
        f.write(pickle.dumps(self.dictionary))

    file = os.path.join(self.path, f"dictionary-{formated_date}.txt")
    with open(file, "w", encoding="utf-8") as f:
        for text_class, e_dict in self.dictionary.items():
            f.write("**************\n")
            f.write(f"{text_class}\n")
            f.write("**************\n")
            for expl, freq in e_dict.items():
                f.write(f"{expl}: {freq}\n")

    self.print_metrics()

  def _compute_metrics(self):
    overlap = 0 # number of overlapped entries for each label
    global_avg_w = 0 # global average words per instance
    global_count = 0
    class_avg_w = {}
    word_intersection = None
    for class_label in self.dictionary.keys():
        instances = list(self.dictionary[class_label].keys())
        no_instances = len(instances)
        if word_intersection is None:
            word_intersection = set(instances)
        else:
            word_intersection = set(instances).intersection(word_intersection)
            overlap = len(word_intersection)
        sum_number_of_words = sum([len(instance.split(" ")) for instance in instances])
        class_avg_w[class_label] = sum_number_of_words/no_instances
        global_avg_w += sum_number_of_words
        global_count += no_instances
    if global_count:
        global_avg_w = global_avg_w/global_count
    self.metrics = {
      "dictionary_entries": global_count,
      "overlap_count": overlap,
      "global_average_words_per_instance": global_avg_w,
      "class_average": class_avg_w,
      "overlap_words": word_intersection
    }

  def print_metrics(self):
    if not self.metrics:
        self._compute_metrics()
    metrics_path = os.path.join(self.path, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(str(self.metrics))
        f.write("\nOverlapping words:\n")
        f.write("\n".join(self.metrics["overlap_words"]))

  def get_dict(self):
    """
    Abstract method for building the dictionary
    """
    pass



################################# DICTIONARIES ################################

################################# RAKE-INST ################################

class RakeInstanceExplanations(AbstractDictionary):
  """ Rake max words per instance"""
  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    # self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    self.max_dict = args["max_words_dict"]
    self.max_words = args["phrase_len"]
    self.dictionary = self.get_dict()
    self.tokenizer = spacy.load("en")
    self._save_dict()

  def get_dict(self):
    """
    Builds a dictionary of keywords for each label.
    # {"all":{word:freq}} OR
    {"pos":{word:freq}, "neg":{word:freq}}
    """

    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    if hasattr(self, 'dictionary') and self.dictionary:
        return self.dictionary
    dictionary = OrderedDict()
    corpus = self.dataset.get_training_corpus()

    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    res_phrases = []
    for text_class in corpus.keys():
        phrases = []
        dictionary[text_class] = OrderedDict()
        for review in corpus[text_class]:
            rake = Rake(max_length=self.max_words)
            rake.extract_keywords_from_text(review)
            phrases += rake.get_ranked_phrases_with_scores()
        phrases.sort(reverse=True)
        if self.args["filterpolarity"]:
            print(f"Filtering by polarity {text_class}...")
            phrases = self.filter_by_sentiment_polarity(phrases)
        with open(os.path.join(self.path, f"phrases-{text_class}-{formated_date}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in phrases]))
        if max_per_class:
            phrases = phrases[:max_per_class]
        dictionary[text_class] = OrderedDict(ChainMap(*[{ph[1]:" ".join(corpus[text_class]).count(ph[1])} for ph in phrases]))
    return dictionary

################################# RAKE-CORPUS ################################

class RakeCorpusExplanations(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args["max_words_dict"]
    self.max_words = args["phrase_len"]
    # self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    self.dictionary = self.get_dict()
    self.tokenizer = spacy.load("en")
    self._save_dict()

  def get_dict(self):
    """
    Builds a dictionary of keywords for each label.
    # {"all":{word:freq}} OR
    {"pos":{word:freq}, "neg":{word:freq}}
    """
    if hasattr(self, 'dictionary') and self.dictionary:
        return self.dictionary
    
    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    dictionary = OrderedDict()
    corpus = self.dataset.get_training_corpus()

    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
        dictionary[text_class] = OrderedDict()
        class_corpus = ".\n".join(corpus[text_class])
        rake = Rake(max_length=self.max_words)
        rake.extract_keywords_from_sentences(corpus[text_class])
        phrases = rake.get_ranked_phrases_with_scores()
        phrases.sort(reverse=True)
        if self.args["filterpolarity"]:
            print(f"Filtering by polarity {text_class}...")
            phrases = self.filter_by_sentiment_polarity(phrases)
        with open(os.path.join(self.path, f"phrases-{text_class}-{formated_date}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in phrases]))
        if max_per_class:
            phrases = phrases[:max_per_class]
        dictionary[text_class] = OrderedDict(ChainMap(*[{ph[1]:" ".join(corpus[text_class]).count(ph[1])} for ph in phrases]))

        # result = []
        # count = 0
        # for phrase in phrases:
        #     freq = class_corpus.count(phrase)
        #     if freq > 0:
        #         result.append({phrase:freq})
        #         count+=1
        #     if count == max_per_class:
        #         break;
        
        # tok_words = self.tokenizer(class_corpus)
        # word_freq = Counter([token.text for token in tok_words if not token.is_punct])
        # dictionary[text_class] = OrderedDict(ChainMap(*result)) # len(re.findall(".*".join(phrase.split()), class_corpus))

    return dictionary

################################# YAKE ################################

class DefaultYAKE(AbstractDictionary):
  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args.get("max_dict", None)
    self.max_words = args["max_words_dict"]
    self.dictionary = self.get_dict()
    self.tokenizer = spacy.load("en")
    self._save_dict()
  
  def get_dict(self):
    """
    Builds a dictionary of keywords for each label.
    # {"all":{word:freq}} OR
    {"pos":{word:freq}, "neg":{word:freq}}
    """
    if hasattr(self, 'dictionary') and self.dictionary:
        return self.dictionary
    dictionary = OrderedDict()
    corpus = self.dataset.get_training_corpus()

    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
        dictionary[text_class] = OrderedDict()
        phrases = [yake.KeywordExtractor().extract_keywords(review) for review in corpus[text_class] if review]
        phrases = list(itertools.chain.from_iterable(phrases))
        phrases.sort(key=lambda x: x[1])
        rev_phrases = [(score, phrase) for score, phrase in phrases]
        if self.args["filterpolarity"]:
            print(f"Filtering by polarity {text_class}...")
            phrases = self.filter_by_sentiment_polarity(rev_phrases)
        with open(os.path.join(self.path, f"raw-phrases-{text_class}-{formated_date}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in phrases]))
        phrases = list(set([" ".join(ph[0].split()[:self.max_words]) for ph in phrases]))
        dictionary[text_class] = OrderedDict(ChainMap(*[{phrases[i]:" ".join(corpus[text_class]).count(phrases[i])} for i in range(min(max_per_class,len(phrases)))]))
    return dictionary

################################# TEXTRANK ################################


class TextRank(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args.get("max_dict", None)
    self.max_words = args["max_words_dict"]
    self.dictionary = self.get_dict()
    self.tokenizer = spacy.load("en")
    self._save_dict()
  
  def get_dict(self):
    """
    Builds a dictionary of keywords for each label.
    # {"all":{word:freq}} OR
    {"pos":{word:freq}, "neg":{word:freq}}
    """
    if hasattr(self, 'dictionary') and not self.dictionary:
        return self.dictionary
    dictionary = OrderedDict()
    corpus = self.dataset.get_training_corpus()

    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
        dictionary[text_class] = OrderedDict()
        phrases = [keywords.keywords(review, scores=True) for review in corpus[text_class]]
        phrases = list(itertools.chain.from_iterable(phrases))
        phrases.sort(reverse=True, key=lambda x: x[1])
        rev_phrases = [(score, phrase) for score, phrase in phrases]
        if self.args["filterpolarity"]:
            print(f"Filtering by polarity {text_class}...")
            phrases = self.filter_by_sentiment_polarity(rev_phrases)
        with open(os.path.join(self.path, f"raw-phrases-{text_class}-{formated_date}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in phrases]))
        phrases = list(set([" ".join(ph[0].split()[:self.max_words]) for ph in phrases]))
        dictionary[text_class] = OrderedDict(ChainMap(*[{phrases[i]:" ".join(corpus[text_class]).count(phrases[i])} for i in range(min(max_per_class,len(phrases)))]))
    return dictionary


parser = argparse.ArgumentParser(description='Config params.')
parser.add_argument('-p', metavar='phrase_len', type=int, default=CONFIG["phrase_len"],
                    help='Max number of words per phrase in explanations dictionary')
parser.add_argument('-size', metavar='max_words_dict', type=int, default=CONFIG["max_words_dict"],
                    help='Max number of words per phrase in explanations dictionary')

parser.add_argument('-d', metavar='dictionary_type', type=str,
                    help='Dictionary type: tfidf, rake-inst, rake-corpus, textrank, yake')
parser.add_argument('-cp', metavar='checkpoint_file', type=str)

parser.add_argument('--load', action='store_true')
parser.add_argument('--filterpolarity', action='store_true')

parser.add_argument('--td', action='store_true',
                    help='Toy data (load just a small data subset)')

args = parser.parse_args()

if args.load:
    CONFIG["load_dictionary"] = True
    CONFIG["dict_checkpoint_file"] = args.cp
if args.filterpolarity:
    CONFIG["filterpolarity"] = True
if args.td:
    CONFIG["toy_data"] = True
CONFIG["max_words_dict"] = args.size
CONFIG["phrase_len"] = args.p

print(CONFIG)
try:

    start = datetime.now()
    dataset = IMDBDataset(CONFIG)
    print(f"Time data load: {str(datetime.now()-start)}")
    
    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    if args.d=="tfidf":
        explanations = TFIDF(f"tfidf-filtered_{CONFIG['filterpolarity']}", dataset, CONFIG)
    elif args.d=="yake":
        explanations = DefaultYAKE(f"default-yake-filtered_{CONFIG['filterpolarity']}-p{CONFIG['phrase_len']}-d{CONFIG['max_words_dict']}", dataset, CONFIG)
    elif args.d=="textrank":
        explanations = TextRank(f"textrank-filtered_{CONFIG['filterpolarity']}-p{CONFIG['phrase_len']}-d{CONFIG['max_words_dict']}", dataset, CONFIG)
    elif args.d == "rake-inst":
        explanations = RakeInstanceExplanations(f"rake-instance-{CONFIG['max_words_dict']}-{args.p}-filtered{CONFIG ['filterpolarity']}", dataset, CONFIG)
    elif args.d == "rake-corpus":
        explanations = RakeCorpusExplanations(f"rake-corpus-{CONFIG['max_words_dict']}-{args.p}-filtered{CONFIG ['filterpolarity']}", dataset, CONFIG)
    elif args.d == "rake-polarity":
        explanations = RakeCorpusPolarityFiltered(f"rake-polarity", dataset, CONFIG)
    print(f"Time explanations: {str(datetime.now()-start)}")

except:
    import sys, traceback
    traceback.print_exc(file=sys.stdout)
