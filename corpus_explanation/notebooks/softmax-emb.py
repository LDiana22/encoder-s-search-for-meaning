# -*- coding: utf-8 -*-

"""# Constants"""

# %% [code]
# -*- coding: utf-8 -*-
from abc import ABC
from collections import OrderedDict 
from collections import ChainMap
from contextlib import redirect_stdout
import glob
import io
import itertools
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
PREFIX_DIR = "experiments/seed-test-2"
MODEL_MAPPING = "experiments/model_mappings/seed-test-6"

MODEL_NAME = "imdb-softmax-seed2"

CONFIG = {
    "toy_data": False, # load only a small subset

    "cuda": True,

    "embedding": "glove",

    "restore_checkpoint" : False,
    "checkpoint_file": None,
    "train": True,

    "dropout": 0.05,
    "weight_decay": 5e-06,

    "patience": 10,

    "epochs": 50,

    "objective": "cross_entropy",
    "init_lr": 0.0001,

    "gumbel_decay": 1e-5,


    "max_words_dict": 5,


    "prefix_dir" : PREFIX_DIR,

    "dirs": {
        "metrics": "metrics",
        "checkpoint": "snapshot",
        "dictionary": "dictionaries",
        "explanations": "explanations"
        },

    "aspect": "palate", # aroma, palate, smell, all
    "max_vocab_size": 100000,
    "emb_dim": 300,
    "batch_size": 32,
    "output_dim": 1,
}


# %% [code]
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
DATE_REGEXP = '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}'

"""# Helpers"""



def _extract_date(f):
    date_string = re.search(f"^{DATE_REGEXP}",f)[0]
    return datetime.strptime(date_string, DATE_FORMAT)

def _extract_date(f):
    date_string = re.search(f"^{DATE_REGEXP}",f)[0]
    return datetime.strptime(date_string, DATE_FORMAT)

def get_max_index_checkpoint(path):
  """
  Return int: suffix of checkpoint name
  """
  list_of_files = os.listdir(path)
  # list_of_files = ["checkpoint_1","checkpoint_10","checkpoint_2", "checkpoint_22"]

  n = max([_extract_number(f) for f in list_of_files]) if list_of_files else None
  if n is None:
    return 0

  return n

def get_last_checkpoint_by_date(path):
  """
  Return file_name with the largest suffix number
  """
  list_of_files = os.listdir(path)
  print(list_of_files)

  file_dates = {_extract_date(f): f for f in list_of_files}
  if file_dates:
    key = sorted(file_dates.keys(), reverse=True)[0]
    return file_dates[key]
  else:
    return None


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

"""# Experiment"""



class Experiment(object):
    """Holds all the experiment parameters and provides helper functions."""
    def __init__(self, e_id):
        self.id = e_id

    def restore_model(self):
        if self.config["restore_checkpoint"]:
            checkpoint = self.model.checkpoint_dir
            if self.config["checkpoint_file"] is None:
                last_checkpoint = get_last_checkpoint_by_date(checkpoint)
            else:
                last_checkpoint = self.config["checkpoint_file"]
            if last_checkpoint is not None:
                self.model.load_checkpoint(last_checkpoint)
                return True  
            else:
                print(f"No checkpoint found at {checkpoint}")
        return False

    def setup(self):
        self.restore_model()
        return self


  ### DECLARATIVE API ###

    def with_data(self, data):
        self.data = data
        return self

    def with_dictionary(self, dictionary):
        self.dictionary = dictionary
        return self

    def with_config(self, config):
        self.config = config.copy()
        return self

    def override(self, config):
        self.config.update(config)
        return self

    def with_model(self, model):
        self.model = model
        return self
  #### END API ######

    @property
    def experiment_name(self):
        return f'E-{self.id}_M-{self.model.id}'

    """ Dirs
    - *_dir - full path to dir
    """
    @property
    def experiments_dir(self):
        return "experiments"

    def train_model(self):
        training_start_time = datetime.now()

        training_losses, training_acc = [], []
        v_losses, v_acc = [], []

        best_valid_loss = float('inf')
        n_epochs = self.config["epochs"]
        patience, prev_loss = 0, 100
        for epoch in range(n_epochs):
            self.model.epoch = epoch
            start_time = datetime.now()

            train_metrics = self.model.train_model(self.train_iterator)
            valid_metrics = self.model.evaluate(self.valid_iterator, "valid")

            end_time = datetime.now()

            training_losses.append(train_metrics["train_loss"])
            training_acc.append(train_metrics["train_acc"])
            v_losses.append(valid_metrics["valid_loss"])
            v_acc.append(valid_metrics["valid_acc"])

            if valid_metrics["valid_loss"] < best_valid_loss:
                best_valid_loss = valid_metrics["valid_loss"]
                metrics = train_metrics
                metrics.update(valid_metrics)
                self.model.checkpoint(epoch, metrics)
            if prev_loss < valid_metrics["valid_loss"]:
                patience += 1
                if patience == self.config["patience"]:
                    print(f"Patience {patience} break, epoch {epoch+1}")
                    return
            prev_loss = valid_metrics["valid_loss"]


            print(f'Epoch: {epoch+1:02} | Epoch Time: {str(end_time-start_time)}')
            print(f'\tTrain Loss: {train_metrics["train_loss"]:.3f} | Train Acc: {train_metrics["train_acc"]*100:.2f}%')
            print(f'\t Val. Loss: {valid_metrics["valid_loss"]:.3f} |  Val. Acc: {valid_metrics["valid_acc"]*100:.2f}%')


        print(f'Training Time: {str(datetime.now()-training_start_time)}')
        print(f'Training losses: {training_losses}')
        print(f'Training acc: {training_acc}')
        print(f'Valid losses: {v_losses}')
        print(f'Valid acc: {v_acc}')
        metrics = {
            "training_time":  str(datetime.now()-training_start_time),
            "training_loss": training_losses,
            "training_acc": training_acc,
            "valid_loss": v_losses,
            "valid_acc": v_acc
         }
        self.model.save_results(metrics, "train")

    def run(self):
        if self.config["restore_checkpoint"]:
            loaded = self.restore_model()
            if not loaded:
                return
        self.train_iterator, self.valid_iterator, self.test_iterator = self.data.iterators()
        if self.config["train"]:
            print("Training...")
            self.train_model()
            print("Restoring best checkpoint...")
            self.model.load_best_model()

        print("Evaluating...")
        metrics = self.model.evaluate(self.test_iterator, "test_f")
        self.model.save_results(metrics, "test")

"""# Data

### IMDB
"""


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
    self.train_data, self.valid_data = self.train_data.split(random_state=np.random.seed(0))
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

"""### UCI"""


class UCIDataset:
  
  def __init__(self, args, max_length=250):
    self.args = args
    self.max_sent_len = 800
    TEXT = data.Field(lower=True, 
                      include_lengths=True,
                      tokenize='spacy',
                      preprocessing=preprocess_pipeline)
    LABEL = data.LabelField(dtype = torch.float)
    print("Loading the dataset...")
    self.train_data, self.test_data, self.valid_data = self._load_data(TEXT, LABEL, UCI_PATH) # 0.6,0.2,0.2
    print("UCI...")
    print(f"Train {len(self.train_data)}")
    print(f"Valid {len(self.valid_data)}")
    print(f"Test {len(self.test_data)}")
    TEXT.build_vocab(self.train_data, 
                 max_size = args["max_vocab_size"],
                 vectors = GloVe(name='6B', dim=args["emb_dim"], cache=VECTOR_CACHE), 
                 unk_init = torch.Tensor.normal_)


    LABEL.build_vocab(self.train_data)

    self.TEXT = TEXT
    self.device = torch.device('cuda' if args["cuda"] else 'cpu')

  def _load_data(self, text_field, label_field, path):
    files = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
    fields = [('text', text_field), ('label', label_field)]
    train_examples, test_examples, valid_examples = [], [], []
    for file_name in files:
        sample = []
        fname = os.path.join(path, file_name)
        with io.open(fname, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line != '\n':
                    text, label = line.split('\t')
                    sent_len = len(text.split())
                    if sent_len > self.max_sent_len:
                        self.max_sent_len = sent_len
                    sample.append(data.Example.fromlist([text.strip(), label], fields))
                if self.args["toy_data"] and (len(sample)==50 or len(sample)==100):
                    break
        train = int(0.8 * len(sample))
        test = int(0.2 * len(sample))
        valid = int(0.2*train)
        train = len(sample) - valid - test
        torch.manual_seed(0)
        train_subset, test_subset, valid_subset = utils.random_split(data.Dataset(sample, fields), (train,test, valid))
        train_data, test_data, valid_data = list(np.array(sample)[train_subset.indices]), list(np.array(sample)[test_subset.indices]), list(np.array(sample)[valid_subset.indices])
        print(file_name, len(train_data), len(test_data), len(valid_data))
        train_examples.extend(train_data)
        test_examples.extend(test_data)
        valid_examples.extend(valid_data)

    print(f'Loaded {len(train_examples)+len(test_examples)+len(valid_examples)}')
    fields = dict(fields)
    # Unpack field tuples
    for n, f in list(fields.items()):
        if isinstance(n, tuple):
            fields.update(zip(n, f))
            del fields[n]
    return data.Dataset(train_examples, fields),data.Dataset(test_examples, fields),data.Dataset(valid_examples, fields)

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
    return self.training_data

  def get_training_corpus(self):
    self.corpus = {"pos":[], "neg":[]}
    self.corpus["pos"] = [" ".join(example.text) for example in self.train_data if example.label == "1"]
    self.corpus["neg"] = [" ".join(example.text) for example in self.train_data if example.label == "0"]
    return self.corpus

  def dev(self):
    return self.valid_data

  def test(self):
    return self.test_data 

  def override(self, args):
    self.args.update(args)
    return self

"""# Dictionary

## Abstract
"""



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

    file = os.path.join(self.path, "dictionary.h5")
    with open(file, "wb") as f: 
        f.write(pickle.dumps(self.dictionary))

    file = os.path.join(self.path, "dictionary.txt")
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

"""## Rake"""

# !pip install rake_nltk


class RakePerClassExplanations(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args.get("max_dict", None)
    self.max_words = args["max_words_dict"]
    # self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    self.dictionary = self.get_dict()
    self.tokenizer = spacy.load("en")
    self._save_dict()
 
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

    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
        dictionary[text_class] = OrderedDict()
        class_corpus = ".\n".join(corpus[text_class])
        rake = Rake()
        rake.extract_keywords_from_sentences(corpus[text_class])
        phrases = rake.get_ranked_phrases()
#         with open(os.path.join(self.path, f"raw-phrases-{text_class}.txt"), "w", encoding="utf-8") as f:
#             f.write("\n".join(phrases))
        phrases = self.filter_phrases_max_words_by_occurence(phrases, class_corpus, max_per_class)
        # tok_words = self.tokenizer(class_corpus)
        # word_freq = Counter([token.text for token in tok_words if not token.is_punct])
        dictionary[text_class] = phrases # len(re.findall(".*".join(phrase.split()), class_corpus))

    return dictionary

class RakeMaxWordsPerInstanceExplanations(AbstractDictionary):
  """ Rake max words per instance"""
  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args.get("max_dict", None)
    self.max_words = args["max_words_dict"]
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
    if hasattr(self, 'dictionary') and not self.dictionary:
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
        phrases = list(set(phrases))
        phrases.sort(reverse=True)
        if max_per_class:
            phrases = phrases[:max_per_class]
        dictionary[text_class] = dict(ChainMap(*[{ph[1]:" ".join(corpus[text_class]).count(ph[1])} for ph in phrases]))
        with open(os.path.join(self.path, f"phrases-{text_class}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in phrases]))
    return dictionary

class RakeMaxWordsExplanations(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args.get("max_dict", None)
    self.max_words = args["max_words_dict"]
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
#     import ipdb
#     ipdb.set_trace(context=20)
    if hasattr(self, 'dictionary') and not self.dictionary:
        return self.dictionary
    dictionary = OrderedDict()
    corpus = self.dataset.get_training_corpus()

    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
        dictionary[text_class] = OrderedDict()
        class_corpus = ".\n".join(corpus[text_class])
        rake = Rake(max_length=self.max_words)
        rake.extract_keywords_from_sentences(corpus[text_class])
        phrases = rake.get_ranked_phrases()
#         with open(os.path.join(self.path, f"raw-phrases-{text_class}.txt"), "w", encoding="utf-8") as f:
#             f.write("\n".join(phrases))
        result = []
        count = 0
        for phrase in phrases:
            freq = class_corpus.count(phrase)
            if freq > 0:
                result.append({phrase:freq})
                count+=1
            if count == max_per_class:
                break;
        
        # tok_words = self.tokenizer(class_corpus)
        # word_freq = Counter([token.text for token in tok_words if not token.is_punct])
        dictionary[text_class] = dict(ChainMap(*result)) # len(re.findall(".*".join(phrase.split()), class_corpus))

    return dictionary

a=[("As",0), ("Z",12), ("b",4)]
a.sort(reverse=True, key=lambda x: x[1])
a

"""## TextRank"""

# pip install summa


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

    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
        dictionary[text_class] = OrderedDict()
        phrases = [keywords.keywords(review, scores=True) for review in corpus[text_class]]
        phrases = list(itertools.chain.from_iterable(phrases))
        phrases.sort(reverse=True, key=lambda x: x[1])
        with open(os.path.join(self.path, f"raw-phrases-{text_class}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in phrases]))
        phrases = list(set([" ".join(ph[0].split()[:self.max_words]) for ph in phrases]))
        dictionary[text_class] = dict(ChainMap(*[{phrases[i]:" ".join(corpus[text_class]).count(phrases[i])} for i in range(min(max_per_class,len(phrases)))]))
    return dictionary

"""## TF-IDF"""

class TFIDF(AbstractDictionary):

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

    for text_class in corpus.keys():
        dictionary[text_class] = OrderedDict()

        cv = CountVectorizer()

        # convert text data into term-frequency matrix
        data = cv.fit_transform(corpus[text_class])
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit_transform(data)
        word2tfidf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))
        sorted_keywords = sorted(word2tfidf, reverse=True, key=word2tfidf.get)

        max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None

        with open(os.path.join(self.path, f"raw-phrases-{text_class}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted_keywords))
        phrases = list(set(sorted_keywords))
        # dictionary[text_class] = [phrases[i] for i in range(min(max_per_class,len(phrases)))]
        dictionary[text_class] = dict(ChainMap(*[{sorted_keywords[i]:" ".join(corpus[text_class]).count(sorted_keywords[i])} for i in range(min(max_per_class,len(sorted_keywords)))]))
    return dictionary

"""## YAKE"""

# pip install git+https://github.com/LIAAD/yake


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
    if hasattr(self, 'dictionary') and not self.dictionary:
        return self.dictionary
    dictionary = OrderedDict()
    corpus = self.dataset.get_training_corpus()

    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
        dictionary[text_class] = OrderedDict()
        phrases = [yake.KeywordExtractor().extract_keywords(review) for review in corpus[text_class] if review]
        phrases = list(itertools.chain.from_iterable(phrases))
        phrases.sort(key=lambda x: x[1])
        with open(os.path.join(self.path, f"raw-phrases-{text_class}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in phrases]))
        phrases = list(set([" ".join(ph[0].split()[:self.max_words]) for ph in phrases]))
        dictionary[text_class] = dict(ChainMap(*[{phrases[i]:" ".join(corpus[text_class]).count(phrases[i])} for i in range(min(max_per_class,len(phrases)))]))
    return dictionary

"""# Models

## Abstract
"""



class AbstractModel(nn.Module):
    """
    Abstract Model
        - saves the mapping between the model-id and its parameters and
            model summary
        - creates the directories for the log files
    """
    def __init__(self, id, mapping_file_location, model_args):
        """
        id: Model id
        mapping_file_location: directory to store the file "model_id" 
                               that containes the hyperparameters values and 
                               the model summary
        logs_location: directory for the logs location of the model
        model_args: hyperparameters of the model
        """
        super().__init__()
        self.delim = "#################################"
        self.id = id
        self.mapping_location = mapping_file_location
        self.args = model_args
        self.epoch=-1
        self.device = torch.device('cuda' if model_args["cuda"] else 'cpu')
        self.model_dir = model_dir = os.path.join(self.args["prefix_dir"], self.id)
        self.__create_directories()

    def override(self, args):
        self.args.update(args)

    def __create_directories(self):
        """
        All the directories for a model are placed under the directory 
            prefix_dir / model_id / {dirs}
        """ 
        self.checkpoint_dir = os.path.join(self.model_dir, self.args["dirs"]["checkpoint"])
        for directory in self.args["dirs"].values():
            m_dir = os.path.join(self.model_dir, directory)
            if not os.path.isdir(m_dir):
                os.makedirs(m_dir)
        if not os.path.isdir(self.mapping_location):
            os.makedirs(self.mapping_location)

    def save_model_type(self, model):
        """
        Saves the hyperparameters 
        """
        mapping_file = os.path.join(self.mapping_location, self.id)        
        with open(mapping_file, "w") as map_file:
            print(self.delim, file=map_file)
            print(self.args, file=map_file)
            print(self.delim, file=map_file)
            print(self, file=map_file)
            print(self.delim, file=map_file)

    def checkpoint(self, epoch, metrics):
        checkpoint_file = os.path.join(self.checkpoint_dir, 
            f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_e{epoch}')
        self.dict_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }
        self.dict_checkpoint.update(metrics)
        torch.save(self.dict_checkpoint, checkpoint_file)

    def load_best_model(self):
        newest_checkpoint = get_last_checkpoint_by_date(os.path.join(self.model_dir, self.args["dirs"]["checkpoint"]))
        print(f"Loading best model from {newest_checkpoint}")
        self.load_checkpoint(newest_checkpoint)

    def load_checkpoint(self, newest_file_name):
        checkpoint_dir = os.path.join(self.model_dir, self.args["dirs"]["checkpoint"])           

        path = os.path.join(checkpoint_dir, newest_file_name)
        print(f"Loading checkpoint: {path}") 
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.metrics = {}
        for key in checkpoint.keys():
            if key not in ['epoch', 'model_state_dict', 'optimizer_state_dict']:
                self.metrics[key] = checkpoint[key]

    def save_results(self, metrics, file_suffix=""):
        metrics_path = os.path.join(self.model_dir, self.args["dirs"]["metrics"])
        results_file = os.path.join(metrics_path, f"results_{file_suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
        with open(results_file, "w") as f:
            f.write(str(metrics))
            f.write("\n\n")
            f.write("Loss, acc, prec, rec, F1, macroF1, microF1, weightedF1\n")
            metric_names = ['test_f_loss', 'test_f_acc', 'test_f_prec', 'test_f_rec', 'test_f_f1', 'test_f_macrof1', 'test_f_microf1', 'test_f_weightedf1']
            res = []
            for metric in metric_names:
                res += ["{0:.4f}".format(metrics.get(metric, -1))]
            f.write(" & ".join(res))

    def train_model(self, iterator):
        """
        metrics.keys(): [train_acc, train_loss, train_prec,
                        train_rec, train_f1, train_macrof1,
                        train_microf1, train_weightedf1]
        e.g. metrics={"train_acc": 90.0, "train_loss": 0.002}
        """
        e_loss = 0
        e_acc, e_prec, e_rec = 0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0

        self.train()

        for batch in iterator:
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
            logits = self.forward(text, text_lengths).squeeze()
            batch.label = batch.label.to(self.device)
            loss = self.criterion(logits, batch.label)

            y_pred = torch.round(torch.sigmoid(logits)).detach().cpu().numpy()
            y_true = batch.label.cpu().numpy()
            #metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            macrof1 = f1_score(y_true, y_pred, average='macro')
            microf1 = f1_score(y_true, y_pred, average='micro')
            wf1 = f1_score(y_true, y_pred, average='weighted')

            loss.backward()
            self.optimizer.step()

            e_loss += loss.item()
            e_acc += acc
            e_prec += prec
            e_rec += rec
            e_f1 += f1
            e_macrof1 += macrof1
            e_microf1 += microf1
            e_wf1 += wf1

        metrics ={}
        size = len(iterator)
        metrics["train_loss"] = e_loss/size
        metrics["train_acc"] = e_acc/size
        metrics["train_prec"] = e_prec/size
        metrics["train_rec"] = e_rec/size
        metrics["train_f1"] = e_f1/size
        metrics["train_macrof1"] = e_macrof1/size
        metrics["train_microf1"] = e_microf1/size
        metrics["train_weightedf1"] = e_wf1/size

        return metrics

    def evaluate(self, iterator, prefix="test"):
        """
            Return a metrics dict with the keys prefixed by prefix
            metrics = {}
            e.g. metrics={f"{prefix}_acc": 90.0, f"{prefix}_loss": 0.002}
        """
        self.eval()

        e_loss = 0
        e_acc, e_prec, e_rec = 0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0
        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text
                logits = self.forward(text, text_lengths, prefix).squeeze()
                batch.label = batch.label.to(self.device)
                loss = self.criterion(logits, batch.label)

                predictions = torch.round(torch.sigmoid(logits))

                y_pred = predictions.detach().cpu().numpy()
                y_true = batch.label.cpu().numpy()

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                macrof1 = f1_score(y_true, y_pred, average='macro')
                microf1 = f1_score(y_true, y_pred, average='micro')
                wf1 = f1_score(y_true, y_pred, average='weighted')

                e_loss += loss.item()
                e_acc += acc
                e_prec += prec
                e_rec += rec
                e_f1 += f1
                e_macrof1 += macrof1
                e_microf1 += microf1
                e_wf1 += wf1

        metrics ={}
        size = len(iterator)
        metrics[f"{prefix}_loss"] = e_loss/size
        metrics[f"{prefix}_acc"] = e_acc/size
        metrics[f"{prefix}_prec"] = e_prec/size
        metrics[f"{prefix}_rec"] = e_rec/size
        metrics[f"{prefix}_f1"] = e_f1/size
        metrics[f"{prefix}_macrof1"] = e_macrof1/size
        metrics[f"{prefix}_microf1"] = e_microf1/size
        metrics[f"{prefix}_weightedf1"] = e_wf1/size
        return metrics

"""## Vanilla"""

class VLSTM(AbstractModel):
    """
    Baseline - no generator model
    """
    def __init__(self, id, mapping_file_location, model_args):
        """
        id: Model id
        mapping_file_location: directory to store the file "model_id" 
                               that containes the hyperparameters values and 
                               the model summary
        logs_location: directory for the logs location of the model
        model_args: hyperparameters of the model
        """
        super().__init__(id, mapping_file_location, model_args)
        self.device = torch.device('cuda' if model_args["cuda"] else 'cpu')

        # UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        # PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        self.input_size = model_args["max_vocab_size"]
        self.embedding = nn.Embedding(self.input_size, model_args["emb_dim"])
        nn.init.uniform_(self.embedding.weight.data,-1,1)

        self.lstm = nn.LSTM(model_args["emb_dim"], 
                           model_args["hidden_dim"], 
                           num_layers=model_args["n_layers"], 
                           bidirectional=True, 
                           dropout=model_args["dropout"])
        self.lin = nn.Linear(2*model_args["hidden_dim"], model_args["output_dim"])
        self.dropout = nn.Dropout(model_args["dropout"])

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

        self = self.to(self.device)
        super().save_model_type(self)

    def forward(self, text, text_lengths, defaults=None):
        text = text.to(self.device)
        #text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))

        #embedded = [sent len, batch size, emb dim]

        return self.raw_forward(embedded, text_lengths)

    def raw_forward(self, embedded, text_lengths):
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors

        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        #hidden = [batch size, hid dim * num directions]

        return self.lin(hidden).to(self.device)

"""## MLP"""


class MLPGen(AbstractModel):
    """
    MLP generator - dictionary for all classes (mixed)
    """
    def __init__(self, id, mapping_file_location, model_args, dataset, explanations):
        """
        id: Model id
        mapping_file_location: directory to store the file "model_id" 
                               that containes the hyperparameters values and 
                               the model summary
        logs_location: directory for the logs location of the model
        model_args: hyperparameters of the model
        explanations: Dictionary of explanations [{phrase: {class:freq}}]
        """
        super().__init__(id, mapping_file_location, model_args)
        self.explanations_path = os.path.join(self.model_dir, model_args["dirs"]["explanations"], "e")

        self.vanilla = VLSTM("gen-van-lstm", mapping_file_location, model_args)
        self.TEXT = dataset.TEXT

        self.max_sent_len = dataset.max_sent_len
        UNK_IDX = dataset.TEXT.vocab.stoi[dataset.TEXT.unk_token]
        PAD_IDX = dataset.TEXT.vocab.stoi[dataset.TEXT.pad_token]
        self.input_size = len(dataset.TEXT.vocab)
        self.embedding = nn.Embedding(self.input_size, model_args["emb_dim"], padding_idx=PAD_IDX)
        
        nn.init.uniform_(self.embedding.weight.data,-1,1)

        self.emb_dim = model_args["emb_dim"]
        # self.gen = nn.LSTM(model_args["emb_dim"], 
        #                    model_args["hidden_dim"], 
        #                    num_layers=model_args["n_layers"], 
        #                    bidirectional=True,
        #                    dropout=model_args["dropout"])


        self.lin = nn.Linear(model_args["emb_dim"], model_args["hidden_dim"]).to(self.device)
        # self.lin12 = nn.Linear(model_args["emb_dim"], 2*model_args["hidden_dim"]).to(self.device)
        # self.tanhsh = nn.Tanhshrink()
        # self.lin21 = nn.Linear(2*model_args["hidden_dim"], model_args["hidden_dim"]).to(self.device)
        # self.selu = nn.SELU() 
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax()  
        # self.sigmoid = nn.Sigmoid()

        self.dictionaries = explanations.get_dict()
        self.lin_pos = nn.Linear(model_args["hidden_dim"], len(self.dictionaries["pos"])).to(self.device)
        self.lin_neg = nn.Linear(model_args["hidden_dim"], len(self.dictionaries["neg"])).to(self.device)
        self.aggregation_pos = nn.Linear(self.max_sent_len, 1).to(self.device)
        self.aggregation_neg = nn.Linear(self.max_sent_len, 1).to(self.device)


        self.explanations = []
        for class_label in self.dictionaries.keys():
            dictionary = self.dictionaries[class_label]
            stoi_expl = self.__pad([
                torch.tensor([self.TEXT.vocab.stoi[word] for word in phrase.split()]).to(self.device)
                for phrase in dictionary.keys()], explanations.max_words)
            self.explanations.append(stoi_expl)

        self.dropout = nn.Dropout(model_args["dropout"])

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

        self = self.to(self.device)
        super().save_model_type(self)

    def __pad(self, tensor_list, length):
        """
        0 pad to the right for a list of variable sized tensors
        e.g. [torch.tensor([1,2]), torch.tensor([1,2,3,4]),torch.tensor([1,2,3,4,5])], 5 ->
                [tensor([1, 2, 0, 0, 0]), tensor([1, 2, 3, 4, 0]), tensor([1, 2, 3, 4, 5])]
        """
        return torch.stack([torch.cat([tensor, tensor.new(length-tensor.size(0)).zero_()])
            for tensor in tensor_list]).to(self.device)


    def _decode_expl_distr(self, distr, dictionary, threshold_expl_score=0.5):
        """
        An expl distribution for a given dict
        dictionary - the dict corresponding to the class of the distribution
        """
        decoded = OrderedDict()
#         for distr in len(distributions):          
            # dict phrase:count
            # distribution for each dict/class
            # index sort - top predicted explanations
        top_rated_expl_index = torch.argsort(distr, 0, True).tolist()
        most_important_expl_idx = [idx for idx in top_rated_expl_index if distr[idx]>=threshold_expl_score]
        if not most_important_expl_idx:
            # empty, then take max only
            max_val = torch.max(distr)
            most_important_expl_idx = [idx for idx in top_rated_expl_index if distr[idx]==max_val]
        # top expl for each instance
        expl_text = np.array(list(dictionary.keys()))[most_important_expl_idx]
        #expl: (count in class, distr value)
        for i, text in enumerate(expl_text):
            decoded[text]= (dictionary[text], distr[most_important_expl_idx[i]].item())
#         batch_explanations.append(decoded)
        # list of 
        # ordered dict {expl:count} for a given dictionary/class
        return decoded

    def get_explanations(self, text, file_name=None):
        text = text.transpose(0,1)
#             start = datetime.now()
#             formated_date = start.strftime(DATE_FORMAT)
#             e_file = f"{self.explanations_path}_{file_name}_{formated_date}.txt"
#             with open(e_file, "w", encoding="utf-8") as f:
#                 print("Saving explanations at ", e_file)
        text_expl = OrderedDict() # text: [expl_c1, expl_c2]           
        for class_idx, class_batch_dict in enumerate(self.expl_distributions):
            #  tensor [batch, dict]
            label = list(self.dictionaries.keys())[class_idx]
            dictionary = self.dictionaries[label]
            for i in range(len(class_batch_dict)):
                nlp_expl_dict = self._decode_expl_distr(class_batch_dict[i], dictionary)
                nlp_text = " ".join([self.TEXT.vocab.itos[idx] for idx in (text[i])])
                val = text_expl.get(nlp_text,[])
                val.append(nlp_expl_dict)
                val.append(self.predictions[i])
                val.append(self.true_labels[i])
                text_expl[nlp_text] = val

            # header text,list of classes
#                 f.write("text, " + ", ".join(list(self.dictionaries.keys()))+"\n")
#                 f.write("\n".join([f"{review} ~ {text_expl[review]}" for review in text_expl.keys()]))
        return text_expl

    def gen(self, activ, batch_size):
        context_vector, final_dict, expl_distributions = [], [], []
        # [dict_size, max_words, emb_dim]
        # explanations[i] -> [dict_size, max_words, emb_dim]
        v_emb_pos = self.embedding(self.explanations[0])
        v_emb_neg = self.embedding(self.explanations[1])
        #[batch,dict_size, max_words, emd_dim
        vocab_emb_pos = v_emb_pos.repeat(batch_size,1,1,1)
        vocab_emb_neg = v_emb_neg.repeat(batch_size,1,1,1)

        #[batch,dict_size, max_words* emd_dim]
        vocab_emb_pos = vocab_emb_pos.reshape(vocab_emb_pos.size(0),vocab_emb_pos.size(1),-1)
        vocab_emb_neg = vocab_emb_neg.reshape(vocab_emb_neg.size(0),vocab_emb_neg.size(1),-1)

        # [sent, batch, dict_size]
        expl_activ_pos = self.lin_pos(activ)
        expl_activ_neg = self.lin_neg(activ)

        # [batch, sent, dict_size]
        expl_distribution_pos = torch.transpose(expl_activ_pos, 0, 1)
        expl_distribution_neg = torch.transpose(expl_activ_neg, 0, 1)
        
        # [batch, max_sent, dict_size] (pad right)
        size1, size2, size3 = expl_distribution_pos.shape[0], expl_distribution_pos.shape[1], expl_distribution_pos.shape[2]
        if self.max_sent_len>=size2:
            # 0-padding
            expl_distribution_pos = torch.cat([expl_distribution_pos, expl_distribution_pos.new(size1, self.max_sent_len-size2, size3).zero_()],1).to(self.device)
            expl_distribution_neg = torch.cat([expl_distribution_neg, expl_distribution_neg.new(size1, self.max_sent_len-size2, size3).zero_()],1).to(self.device)
        else:
            # trimming
            expl_distribution_pos = expl_distribution_pos[:,:self.max_sent_len,:]
            expl_distribution_neg = expl_distribution_neg[:,:self.max_sent_len,:]
        


        # [batch,dict_size, sent]
        e_pos = torch.transpose(expl_distribution_pos,1,2)
        e_neg = torch.transpose(expl_distribution_neg,1,2)
        # [batch, dict, 1]
        expl_distribution_pos = self.aggregation_pos(e_pos).squeeze()
        expl_distribution_neg = self.aggregation_neg(e_neg).squeeze()
        # expl_distribution = self.sigmoid(expl_distribution) # on dim 1
        # expl_distribution_pos = F.gumbel_softmax(expl_distribution_pos, hard=True)
        # expl_distribution_neg = F.gumbel_softmax(expl_distribution_neg, hard=True)

        expl_distribution_pos = self.softmax(expl_distribution_pos)
        expl_distribution_neg = self.softmax(expl_distribution_neg)
        #[batch, dict]
        expl_distributions.append(torch.squeeze(expl_distribution_pos))
        expl_distributions.append(torch.squeeze(expl_distribution_neg))

        # [batch,1, dict]
        e_dist_pos = torch.transpose(expl_distribution_pos.unsqueeze(-1),1,2)
        e_dist_neg = torch.transpose(expl_distribution_neg.unsqueeze(-1),1,2)
        # batch, 1, dict x batch, dict, emb (max_words*emb_dim)
        expl_pos = torch.bmm(e_dist_pos, vocab_emb_pos)
        expl_neg = torch.bmm(e_dist_neg, vocab_emb_neg)

        #[batch,max_words,emb_dim]
        context_vector.append(torch.max(expl_pos, dim=1).values.reshape(batch_size, v_emb_pos.size(1),-1))
        context_vector.append(torch.max(expl_neg, dim=1).values.reshape(batch_size, v_emb_neg.size(1),-1))


        sep = torch.zeros((batch_size,1,self.emb_dim), device=self.device)
        # [batch, 1+1, emb_dim]
        final_dict.append(torch.cat((sep, context_vector[0]), 1))
        final_dict.append(torch.cat((sep, context_vector[1]), 1))

        return final_dict, expl_distributions



    def forward(self, text, text_lengths, expl_file=None):

        batch_size = text.size()[1]

        #text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        #embedded = [sent len, batch size, emb dim]

        ##GEN
        # # [sent len, batch, 2*hidden_dim]
        # expl_activ, (_, _) = self.gen(embedded)
        # expl_activ = nn.Dropout(0.4)(expl_activ)

        #TODO batch, sent*emb
        # batch, sent*hidden
        # batch, dict


        # [sent, batch, hidden]
        expl_activ = self.lin(embedded)
        # expl_activ = self.lin21(embedded)
        expl_activ = self.relu(expl_activ)
        expl_activ = self.dropout(expl_activ)
        # expl_activ = self.lin2(expl_activ)
        # expl_activ = self.relu(expl_activ)
        # # expl_activ = nn.Dropout(0.2)(expl_activ)

        final_dict, expl_distributions = self.gen(expl_activ, batch_size)


        # new TODO
        # reshape
        # batch, hidden, sent -> batch, hidden, dict
        # batch, 1, dict
        final_expl = final_dict[0]
        for i in range(1, len(final_dict)):
            final_expl = torch.cat((final_expl, final_dict[i]), 1)

        #[batch, sent, emb]
        x = torch.transpose(embedded,0,1)

        # [batch, sent_len+2, emb_dim]
        concat_input = torch.cat((x,final_expl),1) 

        #[sent_len+1, batch, emb_dim]
        final_input = torch.transpose(concat_input,0,1)

        output = self.vanilla.raw_forward(final_input, text_lengths)


        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors

        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout

        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        #hidden = [batch size, hid dim * num directions]

        # (list) class, tensor[batch, dict_size]
        self.expl_distributions = expl_distributions 
#             if expl_file and "test" in expl_file and expl_file!="test":
#                 self.save_explanations(text, expl_file)
        return output


    def evaluate(self, iterator, prefix="test"):
        save = False # save explanations
        if prefix=="test_f":
            save = True
            expl = "text, " + ", ".join(list(self.dictionaries.keys())) + ", predictions, true label\n"
            e_list = []
            distr = [torch.tensor([]).to(self.device) for i in range(len(self.dictionaries.keys()))]

        self.eval()
        e_loss = 0
        e_acc, e_prec, e_rec = 0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0
        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text
                logits = self.forward(text, text_lengths, prefix).squeeze()
                batch.label = batch.label.to(self.device)
                loss = self.criterion(logits, batch.label)

                predictions = torch.round(torch.sigmoid(logits))

                y_pred = predictions.detach().cpu().numpy()
                y_true = batch.label.cpu().numpy()
                self.predictions = y_pred
                self.true_labels = y_true
                if save:
                    text_expl= self.get_explanations(text)
                    e_list.append("\n".join([f"{review} ~ {text_expl[review]}" for review in text_expl.keys()]))
                    for class_idx in range(len(distr)):
                        distr[class_idx] = torch.cat((distr[class_idx], self.expl_distributions[class_idx]))
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                macrof1 = f1_score(y_true, y_pred, average='macro')
                microf1 = f1_score(y_true, y_pred, average='micro')
                wf1 = f1_score(y_true, y_pred, average='weighted')

                e_loss += loss.item()
                e_acc += acc
                e_prec += prec
                e_rec += rec
                e_f1 += f1
                e_macrof1 += macrof1
                e_microf1 += microf1
                e_wf1 += wf1
        if save:
            start = datetime.now()
            formated_date = start.strftime(DATE_FORMAT)
            e_file = f"{self.explanations_path}_test-{self.epoch}_{formated_date}.txt"
            print(f"Saving explanations at {e_file}")
            with open(e_file, "w") as f:
                f.write(expl)
                f.write("".join(e_list))
            with open(f"{self.explanations_path}_distr.txt", "w") as f:
                f.write(str(distr))
                f.write("\nSUMs\n")
                f.write(str([torch.sum(torch.tensor(d), dim=1) for d in distr]))
                f.write("\nHard sums\n")
                f.write(str([torch.sum(torch.where(d>0.5, torch.ones(d.shape).to(self.device), torch.zeros(d.shape).to(self.device)), dim=1) for d in distr]))
                f.write("\nIndices\n")
                f.write(str([d.nonzero() for d in distr]))

        metrics ={}
        size = len(iterator)
        metrics[f"{prefix}_loss"] = e_loss/size
        metrics[f"{prefix}_acc"] = e_acc/size
        metrics[f"{prefix}_prec"] = e_prec/size
        metrics[f"{prefix}_rec"] = e_rec/size
        metrics[f"{prefix}_f1"] = e_f1/size
        metrics[f"{prefix}_macrof1"] = e_macrof1/size
        metrics[f"{prefix}_microf1"] = e_microf1/size
        metrics[f"{prefix}_weightedf1"] = e_wf1/size
        return metrics

"""# Main"""


start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)
experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 300, 
    "cuda": True,
    "restore_checkpoint" : False,
    "train": True,
    # "epochs": 1,
    # "max_words_dict": 2,
    # "toy_data": True
})

"""## Data"""

# pip install ipdb

# dataset.get_training_corpus()

# dataset = UCIDataset(experiment.config)

dataset = IMDBDataset(experiment.config)

"""## Dict"""

# pip install torch==1.2.0

# explanations = RakeMaxWordsPerInstanceExplanations(f"imdb-rake-max-words-instance-300-{experiment.config['max_words_dict']}", dataset, experiment.config)

# explanations = RakeMaxWordsExplanations(f"rake-dot-300-3", dataset, experiment.config)

# explanations = RakeMaxWordsExplanations(f"rake-corpus-max-{experiment.config['max_words_dict']}")

# explanations = TextRank(f"textrank-300-5", dataset, experiment.config)

# explanations = TFIDF("tfidf", dataset, experiment.config)

# explanations = DefaultYAKE("default-yake", dataset, experiment.config)

"""## Experim"""

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 300, 
    "cuda": True,
    "restore_checkpoint" : False,
    "train": True,
    # "max_words_dict": i,
})
print(experiment.config)

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)
model = VLSTM("v-lstm", MODEL_MAPPING, experiment.config)
experiment.with_data(dataset).with_model(model).run()

print(f"Time: {str(datetime.now()-start)}")

# """### Yake"""

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 300, 
    "cuda": True,
    "restore_checkpoint" : False,
    "train": True,
    # "epochs": 1,
    # "max_words_dict": i,
    # "toy_data": True
})
print(experiment.config)
explanations = DefaultYAKE("default-yake", dataset, experiment.config)
print(f"Time yake expl: {str(datetime.now()-start)}")

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)
#-{experiment.config['max_words_dict']}
model = MLPGen(f"{MODEL_NAME}-yake", MODEL_MAPPING, experiment.config, dataset, explanations)

experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

print(f"Time yake trian: {str(datetime.now()-start)}")

"""###TextRank"""

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 300, 
    "cuda": True,
    "restore_checkpoint" : False,
    "train": True,
    # "epochs": 1,
    # "max_words_dict": i,
    # "toy_data": True
})
print(experiment.config)
explanations = TextRank(f"textrank-300-5", dataset, experiment.config)
print(f"Time textrank expl: {str(datetime.now()-start)}")

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)
#-{experiment.config['max_words_dict']}
model = MLPGen(f"{MODEL_NAME}-textrank", MODEL_MAPPING, experiment.config, dataset, explanations)

experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

print(f"Time TextRank train: {str(datetime.now()-start)}")

"""### TF-IDF"""

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 300, 
    "cuda": True,
    "restore_checkpoint" : False,
    "train": True,
    # "epochs": 1,
    # "max_words_dict": i,
    # "toy_data": True
})
print(experiment.config)
explanations = TFIDF("tfidf", dataset, experiment.config)
print(f"Time TFIDF expl: {str(datetime.now()-start)}")

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)
#-{experiment.config['max_words_dict']}
model = MLPGen(f"{MODEL_NAME}-tfidf", MODEL_MAPPING, experiment.config, dataset, explanations)

experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

print(f"Time TFIDF train: {str(datetime.now()-start)}")

"""### Rake instance"""

for i in range(1,6):
    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
      "hidden_dim": 256,
      "n_layers": 2,
      "max_dict": 300, 
      "cuda": True,
      "restore_checkpoint" : False,
      "train": True,
      # "epochs": 1,
      "max_words_dict": i,
      # "toy_data": True
    })
    print(experiment.config)
    explanations = RakeMaxWordsPerInstanceExplanations(f"rake-max-words-instance-300-{experiment.config['max_words_dict']}", dataset, experiment.config)
    print(f"Time rake explanations: {str(datetime.now()-start)}")

    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)

    model = MLPGen(f"{MODEL_NAME}-inst-rake-inst-max-{experiment.config['max_words_dict']}", MODEL_MAPPING, experiment.config, dataset, explanations)

    experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

    print(f"Time rake train: {str(datetime.now()-start)}")

"""### Rake"""

for i in range(1,6):
    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
      "hidden_dim": 256,
      "n_layers": 2,
      "max_dict": 300, 
      "cuda": True,
      "restore_checkpoint" : False,
      "train": True,
      # "epochs": 1,
      "max_words_dict": i,
      # "toy_data": True
    })
    print(experiment.config)
    explanations = RakeMaxWordsExplanations(f"rake-max-words-corpus-300-{experiment.config['max_words_dict']}", dataset, experiment.config)
    print(f"Time rake corpus explanations: {str(datetime.now()-start)}")

    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)

    model = MLPGen(f"{MODEL_NAME}-inst-rake-corpus-max-{experiment.config['max_words_dict']}", MODEL_MAPPING, experiment.config, dataset, explanations)

    experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

    print(f"Time rake corpus train: {str(datetime.now()-start)}")



"""## Train

### main
"""



start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

model = MLPGen(f"{MODEL_NAME}-inst-rake-inst-max-{experiment.config['max_words_dict']}", MODEL_MAPPING, experiment.config, dataset, explanations)

experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

print(f"Time: {str(datetime.now()-start)}")

# %% [code]
# start = datetime.now()

# model = VLSTM("v-lstm", MODEL_MAPPING, experiment.config, dataset.TEXT)

# experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

# print(f"Time: {str(datetime.now()-start)}")


start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 300, 
    "cuda": True,
    "restore_checkpoint" : False,
    "train": True,
    # "epochs": 1,
    "max_words_dict": 3,
    # "toy_data": True
})
print(experiment.config)
explanations = RakeMaxWordsPerInstanceExplanations(f"imdb-rake-max-words-instance-300-{experiment.config['max_words_dict']}", dataset, experiment.config)
print(f"Time explanations: {str(datetime.now()-start)}")

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

model = MLPGen(f"{MODEL_NAME}-inst-rake-inst-max-{experiment.config['max_words_dict']}", MODEL_MAPPING, experiment.config, dataset, explanations)

experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

print(f"Time: {str(datetime.now()-start)}")


start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 300, 
    "cuda": True,
    "restore_checkpoint" : False,
    "train": True,
    # "epochs": 1,
    "max_words_dict": 4,
    # "toy_data": True
})
print(experiment.config)
explanations = RakeMaxWordsPerInstanceExplanations(f"imdb-rake-max-words-instance-300-{experiment.config['max_words_dict']}", dataset, experiment.config)
print(f"Time explanations: {str(datetime.now()-start)}")

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

model = MLPGen(f"{MODEL_NAME}-inst-rake-inst-max-{experiment.config['max_words_dict']}", MODEL_MAPPING, experiment.config, dataset, explanations)

experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

print(f"Time: {str(datetime.now()-start)}")


start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 300, 
    "cuda": True,
    "restore_checkpoint" : False,
    "train": True,
    # "epochs": 1,
    "max_words_dict": 5,
    # "toy_data": True
})
print(experiment.config)
explanations = RakeMaxWordsPerInstanceExplanations(f"imdb-rake-max-words-instance-300-{experiment.config['max_words_dict']}", dataset, experiment.config)
print(f"Time explanations: {str(datetime.now()-start)}")

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

model = MLPGen(f"{MODEL_NAME}-inst-rake-inst-max-{experiment.config['max_words_dict']}", MODEL_MAPPING, experiment.config, dataset, explanations)

experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

print(f"Time: {str(datetime.now()-start)}")