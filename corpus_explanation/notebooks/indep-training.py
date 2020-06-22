# -*- coding: utf-8 -*-

"""# Constants"""

# %% [code]
# -*- coding: utf-8 -*-
from abc import ABC
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
#PREFIX_DIR = "experiments/independent"
#MODEL_MAPPING = "experiments/model_mappings/independent"
PREFIX_DIR = "experiments/soa-dicts"
MODEL_MAPPING = "experiments/soa-dicts/model_mapping"

MODEL_NAME = "mlp+frozen_bilstm_gumb-emb-one-dict-long"

CONFIG = {
    "toy_data": False, # load only a small subset

    "cuda": True,

    "embedding": "glove",

    "restore_checkpoint" : False,
    "checkpoint_file": None, #"experiments/independent/bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.0-c-e25-2020-05-25_19-22-13/snapshot/2020-05-25_e20",
    "train": True,

    "dropout": 0.05,
    "weight_decay": 5e-06,

    "alpha": 0.5,

    "patience": 10,

    "epochs": 200,

    "objective": "cross_entropy",
    "lr": 0.001,
    "l2_wd": 0,

    "gumbel_decay": 1e-5,


    "max_words_dict": 5,


    "prefix_dir" : PREFIX_DIR,

    "dirs": {
        "metrics": "metrics",
        "checkpoint": "snapshot",
        "dictionary": "dictionaries",
        "explanations": "explanations",
        "plots": "plots"
        },

    "aspect": "palate", # aroma, palate, smell, all
    "max_vocab_size": 100000,
    "emb_dim": 300,
    "batch_size": 32,
    "output_dim": 1,
    "load_dictionary": False
}


# %% [code]
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
DATE_REGEXP = '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}'

"""# Helpers"""

def plot_training_metrics(plot_path, train_loss, valid_loss, training_acc, valid_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(train_loss, 'r', label='Train')
    ax1.plot(valid_loss, 'b', label="Validation")
    ax1.set_title('Loss curve')
    ax1.legend()

    ax2.plot(training_acc, 'r', label='Train')
    ax2.plot(valid_acc, 'b', label="Valid")
    ax2.set_title('Accuracy curve')
    ax2.legend()

    fig.savefig(plot_path)
        
def plot_contributions(plot_path, train_c, valid_c): 
    plt.plot(train_c, 'r', label="Train")
    plt.plot(valid_c, 'b', label="Valid")
    plt.title('Epochs avg contributions')
    plt.legend()

    plt.savefig(plot_path)       

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

        training_losses, training_acc, training_raw_acc = [], [], []
        v_losses, v_acc, v_raw_acc = [], [], []
        training_contrib, v_contrib = [], []

        best_valid_loss = float('inf')
        n_epochs = self.config["epochs"]
        patience, prev_loss = 0, 100
        for epoch in range(n_epochs):
            self.model.epoch = epoch
            start_time = datetime.now()

            train_metrics = self.model.train_model(self.train_iterator, epoch+1)
            valid_metrics = self.model.evaluate(self.valid_iterator, "valid")

            end_time = datetime.now()

            training_losses.append(train_metrics["train_loss"])
            training_acc.append(train_metrics["train_acc"])
            v_losses.append(valid_metrics["valid_loss"])
            v_acc.append(valid_metrics["valid_acc"])
            
            if self.config["id"] != "bilstm":
                training_raw_acc.append(train_metrics["train_raw_acc"])
                training_contrib.append(train_metrics["train_avg_contributions"])
            
                v_raw_acc.append(valid_metrics["valid_raw_acc"])
                v_contrib.append(valid_metrics["valid_avg_contributions"])

            if round(valid_metrics["valid_loss"],3) <= round(best_valid_loss,3):
                best_valid_loss = valid_metrics["valid_loss"]
                print(f"Best valid at epoch {epoch+1}: {best_valid_loss}")
                metrics = train_metrics
                metrics.update(valid_metrics)
                self.model.checkpoint(epoch+1, metrics)
            if prev_loss < valid_metrics["valid_loss"]:
                patience += 1
                if patience == self.config["patience"]:
                    print(f"Patience {patience} break, epoch {epoch+1}")
                    break
            prev_loss = valid_metrics["valid_loss"]


            print(f'Epoch: {epoch+1:02} | Epoch Time: {str(end_time-start_time)}')
            print(f'\tTrain Loss: {train_metrics["train_loss"]:.3f} | Train Acc: {train_metrics["train_acc"]*100:.2f}%')
            print(f'\t Val. Loss: {valid_metrics["valid_loss"]:.3f} |  Val. Acc: {valid_metrics["valid_acc"]*100:.2f}%')
            if self.config["id"] != "bilstm":
                print(f'\tTrain avgC: {train_metrics["train_avg_contributions"]} |  Val. avgC: {valid_metrics["valid_avg_contributions"]}')
                print(f'\tTrain raw_acc: {train_metrics["train_raw_acc"]*100:.2f} | Val. Raw_acc: {valid_metrics["valid_raw_acc"]*100:.2f}%')


        print(f'Training Time: {str(datetime.now()-training_start_time)}')
        print(f'Training losses: {training_losses}')
        print(f'Training acc: {training_acc}')
        print(f'Valid losses: {v_losses}')
        print(f'Valid acc: {v_acc}')
        metrics = {
            "training_time":  str(datetime.now()-training_start_time),
            "training_loss": training_losses,
            "training_acc": training_acc,
            "training_raw_acc": training_raw_acc,
            "valid_loss": v_losses,
            "valid_acc": v_acc,
            "valid_raw_acc": v_raw_acc
         }
        plot_path = self.model.get_plot_path("train_plot")
        print(f"Plotting at {plot_path}")
        plot_training_metrics(plot_path, training_losses, v_losses, training_acc, v_acc)
        if self.config["id"] != "bilstm":
            plot_path = self.model.get_plot_path("contributions_plot")
            print(f"Plotting contributions at {plot_path}")
            plot_contributions(plot_path, training_contrib, v_contrib)

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
        print("Test metrics:")
        print(metrics)
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
    """
    Returns a dictionary: {class: list of instances for that class}
    """
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
    if args["load_dictionary"]:
        print("Loading dictionary...")
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
        # phrases = list(set(phrases))
        phrases.sort(reverse=True)
        with open(os.path.join(self.path, f"phrases-{text_class}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in phrases]))
        if max_per_class:
            phrases = phrases[:max_per_class]
        dictionary[text_class] = OrderedDict(ChainMap(*[{ph[1]:" ".join(corpus[text_class]).count(ph[1])} for ph in phrases]))
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
        dictionary[text_class] = OrderedDict(ChainMap(*result)) # len(re.findall(".*".join(phrase.split()), class_corpus))

    return dictionary


class RakeCorpusPolarityFiltered(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args.get("max_dict", None)
    self.max_words = args["max_words_dict"]
    # self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    if args["load_dict"]:
        print(f"Loading RakeCorpusPolarityFiltered from: {args['dict_checkpoint']}")
        self.dictionary = self.load_dict(args["dict_checkpoint"])
        print(f"Loaded dict keys: {[f'{key}:{len(self.dictionary[key].keys())}' for key in self.dictionary.keys()]}")
    else:
        self.dictionary = self.get_dict()
        self._save_dict()
    self.tokenizer = spacy.load("en")

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
    
    sentiment = SentimentIntensityAnalyzer()
    
    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
        dictionary[text_class] = OrderedDict()
        class_corpus = ".\n".join(corpus[text_class])
        phrases = []
        for i in range(1, self.max_words+1):
            rake = Rake(max_length=self.max_words)
            rake.extract_keywords_from_sentences(corpus[text_class])
            phrases += rake.get_ranked_phrases()
#         with open(os.path.join(self.path, f"raw-phrases-{text_class}.txt"), "w", encoding="utf-8") as f:
#             f.write("\n".join(phrases))
        # extract only phrases with a night polarity degree
        ph_polarity = [(phrase, abs(sentiment.polarity_scores(phrase)['compound'])) for phrase in phrases if abs(sentiment.polarity_scores(phrase)['compound'])>0.5]
        ph_polarity.sort(reverse=True, key=lambda x: x[1])
        # rank based on ferquency and eliminate freq 0
        if not max_per_class:
            max_per_class = len(ph_polarity)
        result = [{phrase[0]: class_corpus.count(phrase[0])} for phrase in ph_polarity[:max_per_class] if class_corpus.count(phrase[0])>0]
        
        # tok_words = self.tokenizer(class_corpus)
        # word_freq = Counter([token.text for token in tok_words if not token.is_punct])
        dictionary[text_class] = OrderedDict(ChainMap(*result)) # len(re.findall(".*".join(phrase.split()), class_corpus))

    return dictionary

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
    if hasattr(self, 'dictionary') and self.dictionary:
        return self.dictionary
    print("Generating new dict rake-inst")
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
            print("Filtering by polarity...")
            phrases = self.filter_by_sentiment_polarity(phrases)
        with open(os.path.join(self.path, f"phrases-{text_class}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in phrases]))
        if max_per_class:
            phrases = phrases[:max_per_class]
        dictionary[text_class] = OrderedDict(ChainMap(*[{ph[1]:" ".join(corpus[text_class]).count(ph[1])} for ph in phrases]))
    return dictionary


"""## TextRank"""

# pip install summa

class TextRank(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
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
        rev_phrases = [(score, phrase) for phrase, score in phrases]
        if self.args["filterpolarity"]:
            print(f"Filtering by polarity {text_class}...")
            rev_phrases = self.filter_by_sentiment_polarity(rev_phrases)
        with open(os.path.join(self.path, f"raw-phrases-{text_class}-{formated_date}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([str(ph) for ph in rev_phrases]))
        phrases = list(set([" ".join(ph[1].split()[:self.max_words]) for ph in rev_phrases]))
        dictionary[text_class] = OrderedDict(ChainMap(*[{phrases[i]:" ".join(corpus[text_class]).count(phrases[i])} for i in range(min(max_per_class,len(phrases)))]))
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
            f.write("\n".join([f"{kw}: {word2tfidf.get(kw)}" for kw in sorted_keywords]))
        # phrases = list(set(sorted_keywords))
        # dictionary[text_class] = [phrases[i] for i in range(min(max_per_class,len(phrases)))]
        dictionary[text_class] = OrderedDict(ChainMap(*[{sorted_keywords[i]:" ".join(corpus[text_class]).count(sorted_keywords[i])} for i in range(min(max_per_class,len(sorted_keywords)))]))
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
        dictionary[text_class] = OrderedDict(ChainMap(*[{phrases[i]:" ".join(corpus[text_class]).count(phrases[i])} for i in range(min(max_per_class,len(phrases)))]))
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
        mapping_file = os.path.join(self.mapping_location, f"{self.id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")        
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
        checkpoint_dir = os.path.join(self.model_dir, self.args["dirs"]["checkpoint"])           
        newest_checkpoint = os.path.join(checkpoint_dir, newest_checkpoint)
        print(f"Loading best model from {newest_checkpoint}")
        self.load_checkpoint(newest_checkpoint)

    def load_checkpoint(self, path):
        print(f"Loading checkpoint: {path}") 
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.metrics = {}
        for key in checkpoint.keys():
            if key not in ['epoch', 'model_state_dict', 'optimizer_state_dict']:
                self.metrics[key] = checkpoint[key]

    def get_plot_path(self, file_suffix ):
        dir_path = os.path.join(self.model_dir, self.args["dirs"]["plots"])
        return os.path.join(dir_path, f"{file_suffix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

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

    def train_model(self, iterator, args=None):
        """
        metrics.keys(): [train_acc, train_loss, train_prec,
                        train_rec, train_f1, train_macrof1,
                        train_microf1, train_weightedf1]
        e.g. metrics={"train_acc": 90.0, "train_loss": 0.002}
        """
        e_loss = 0
        e_acc, e_prec_neg, e_prec_pos, e_rec_neg, e_rec_pos = 0,0,0,0,0
        e_f1_neg, e_f1_pos, e_macrof1, e_microf1, e_wf1 = 0,0,0,0,0

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
            prec_neg = precision_score(y_true, y_pred, pos_label=1)
            prec_pos = precision_score(y_true, y_pred, pos_label=0)
            rec_neg = recall_score(y_true, y_pred, pos_label=1)
            rec_pos = recall_score(y_true, y_pred, pos_label=0)
            f1_neg = f1_score(y_true, y_pred, pos_label=1)
            f1_pos = f1_score(y_true, y_pred, pos_label=0)
            macrof1 = f1_score(y_true, y_pred, average='macro')
            microf1 = f1_score(y_true, y_pred, average='micro')
            wf1 = f1_score(y_true, y_pred, average='weighted')

            loss.backward()
            self.optimizer.step()

            e_loss += loss.item()
            e_acc += acc
            e_prec_neg += prec_neg
            e_prec_pos += prec_pos
            e_rec_neg += rec_neg
            e_rec_pos += rec_pos
            e_f1_neg += f1_neg
            e_f1_pos += f1_pos
            e_macrof1 += macrof1
            e_microf1 += microf1
            e_wf1 += wf1

        metrics ={}
        size = len(iterator)
        metrics["train_loss"] = e_loss/size
        metrics["train_acc"] = e_acc/size
        metrics["train_prec_neg"] = e_prec_neg/size
        metrics["train_prec_pos"] = e_prec_pos/size
        metrics["train_rec_neg"] = e_rec_neg/size
        metrics["train_rec_pos"] = e_rec_pos/size
        metrics["train_f1_neg"] = e_f1_neg/size
        metrics["train_f1_pos"] = e_f1_pos/size
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
        e_acc, e_prec_neg, e_prec_pos, e_rec_neg, e_rec_pos = 0,0,0,0,0
        e_f1_neg, e_f1_pos, e_macrof1, e_microf1, e_wf1 = 0,0,0,0,0
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
                prec_neg = precision_score(y_true, y_pred, pos_label=1)
                prec_pos = precision_score(y_true, y_pred, pos_label=0)
                rec_neg = recall_score(y_true, y_pred, pos_label=1)
                rec_pos = recall_score(y_true, y_pred, pos_label=0)
                f1_neg = f1_score(y_true, y_pred, pos_label=1)
                f1_pos = f1_score(y_true, y_pred, pos_label=0)
                macrof1 = f1_score(y_true, y_pred, average='macro')
                microf1 = f1_score(y_true, y_pred, average='micro')
                wf1 = f1_score(y_true, y_pred, average='weighted')

                e_loss += loss.item()
                e_acc += acc
                e_prec_neg += prec_neg
                e_prec_pos += prec_pos
                e_rec_neg += rec_neg
                e_rec_pos += rec_pos
                e_f1_neg += f1_neg
                e_f1_pos += f1_pos
                e_macrof1 += macrof1
                e_microf1 += microf1
                e_wf1 += wf1

        metrics ={}
        size = len(iterator)
        metrics[f"{prefix}_loss"] = e_loss/size
        metrics[f"{prefix}_acc"] = e_acc/size
        metrics[f"{prefix}_prec_neg"] = e_prec_neg/size
        metrics[f"{prefix}_prec_pos"] = e_prec_pos/size
        metrics[f"{prefix}_rec_neg"] = e_rec_neg/size
        metrics[f"{prefix}_rec_pos"] = e_rec_pos/size
        metrics[f"{prefix}_f1_neg"] = e_f1_neg/size
        metrics[f"{prefix}_f1_pos"] = e_f1_pos/size
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
        # self.softmax = nn.Softmax()  
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
                val.append(self.raw_predictions[i])
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
        expl_distribution_pos = F.gumbel_softmax(expl_distribution_pos, hard=True)
        expl_distribution_neg = F.gumbel_softmax(expl_distribution_neg, hard=True)

        # expl_distribution_pos = self.softmax(expl_distribution_pos)
        # expl_distribution_neg = self.softmax(expl_distribution_neg)
        #[batch, dict]
        expl_distributions.append(torch.squeeze(expl_distribution_pos))
        expl_distributions.append(torch.squeeze(expl_distribution_neg))

        # [batch,1, dict]
        e_dist_pos = torch.transpose(expl_distribution_pos.unsqueeze(-1),1,2)
        e_dist_neg = torch.transpose(expl_distribution_neg.unsqueeze(-1),1,2)
        # batch, 1, dict x batch, dict, emb (max_words*emb_dim)
        expl_pos = torch.bmm(e_dist_pos, vocab_emb_pos)
        expl_neg = torch.bmm(e_dist_neg, vocab_emb_neg)

        # #[batch,max_words,emb_dim]
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

        # [batch, sent_len+2*len(final_dict), emb_dim]
        concat_input = torch.cat((x,final_expl),1) 

        #[sent_len+len(final_dict), batch, emb_dim]
        final_input = torch.transpose(concat_input,0,1)

        output = self.vanilla.raw_forward(final_input, text_lengths+2*len(final_dict))


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
                f.write("\n")
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
##########################################################################################################
############################################Independent training #########################################
##########################################################################################################

class FrozenVLSTM(AbstractModel):
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

        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

        self = self.to(self.device)
        super().save_model_type(self)

    def forward(self, text, text_lengths, defaults=None):
        text = text.to(self.device)
        #text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))

        #embedded = [sent len, batch size, emb dim]

        return self.raw_forward(embedded, text_lengths)[0]

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

        return self.lin(hidden).to(self.device), hidden


    def train_model(self, iterator, args=None):
        """
        metrics.keys(): [train_acc, train_loss, train_prec,
                        train_rec, train_f1, train_macrof1,
                        train_microf1, train_weightedf1]
        e.g. metrics={"train_acc": 90.0, "train_loss": 0.002}
        """
        e_loss = 0
        e_acc, e_prec_neg, e_prec_pos, e_rec_neg, e_rec_pos = 0,0,0,0,0
        e_f1_neg, e_f1_pos, e_macrof1, e_microf1, e_wf1 = 0,0,0,0,0

        self.train()

        for batch in iterator:
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
            #logits = self.forward(text, text_lengths)[0].squeeze()
            logits = self.forward(text, text_lengths).squeeze()


            batch.label = batch.label.to(self.device)
            loss = self.criterion(logits, batch.label)       
            y_pred = torch.round(torch.sigmoid(logits)).detach().cpu().numpy()
            y_true = batch.label.cpu().numpy()
            #metrics
            acc = accuracy_score(y_true, y_pred)
            prec_neg = precision_score(y_true, y_pred, pos_label=1)
            prec_pos = precision_score(y_true, y_pred, pos_label=0)
            rec_neg = recall_score(y_true, y_pred, pos_label=1)
            rec_pos = recall_score(y_true, y_pred, pos_label=0)
            f1_neg = f1_score(y_true, y_pred, pos_label=1)
            f1_pos = f1_score(y_true, y_pred, pos_label=0)
            macrof1 = f1_score(y_true, y_pred, average='macro')
            microf1 = f1_score(y_true, y_pred, average='micro')
            wf1 = f1_score(y_true, y_pred, average='weighted')

            loss.backward()
            self.optimizer.step()

            e_loss += loss.item()
            e_acc += acc
            e_prec_neg += prec_neg
            e_prec_pos += prec_pos
            e_rec_neg += rec_neg
            e_rec_pos += rec_pos
            e_f1_neg += f1_neg
            e_f1_pos += f1_pos
            e_macrof1 += macrof1
            e_microf1 += microf1
            e_wf1 += wf1

        metrics ={}
        size = len(iterator)
        metrics["train_loss"] = e_loss/size
        metrics["train_acc"] = e_acc/size
        metrics["train_prec_neg"] = e_prec_neg/size
        metrics["train_prec_pos"] = e_prec_pos/size
        metrics["train_rec_neg"] = e_rec_neg/size
        metrics["train_rec_pos"] = e_rec_pos/size
        metrics["train_f1_neg"] = e_f1_neg/size
        metrics["train_f1_pos"] = e_f1_pos/size
        metrics["train_macrof1"] = e_macrof1/size
        metrics["train_microf1"] = e_microf1/size
        metrics["train_weightedf1"] = e_wf1/size

        return metrics




class MLPIndependentOneDict(AbstractModel):
    """
    pretrained bi-LSTM + MLP
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


        if model_args["restore_v_checkpoint"]:
            vanilla_args = copy.deepcopy(model_args)
            vanilla_args["restore_checkpoint"] = True
            self.vanilla = FrozenVLSTM("frozen-bi-lstm", mapping_file_location, vanilla_args)
            print(model_args["checkpoint_v_file"])
            self.vanilla.load_checkpoint(model_args["checkpoint_v_file"])
            for param in self.vanilla.parameters():
                param.requires_grad=False
        else:
            self.vanilla = FrozenVLSTM("bi-lstm", mapping_file_location, model_args)


        self.TEXT = dataset.TEXT

        self.max_sent_len = dataset.max_sent_len
        # UNK_IDX = dataset.TEXT.vocab.stoi[dataset.TEXT.unk_token]
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
        # self.softmax = nn.Softmax()  
        # self.sigmoid = nn.Sigmoid()

        dictionaries = explanations.get_dict()
        self.dictionary = copy.deepcopy(dictionaries['pos'])
        self.dictionary.update(dictionaries['neg'])
        print("Dict size", len(self.dictionary.keys()))
        self.lin_pos = nn.Linear(2*model_args["hidden_dim"], len(self.dictionary.keys())).to(self.device)
        # self.lin_neg = nn.Linear(model_args["hidden_dim"], len(self.dictionaries["neg"])).to(self.device)
        self.aggregation_pos = nn.Linear(self.max_sent_len, 1).to(self.device)
        # self.aggregation_neg = nn.Linear(self.max_sent_len, 1).to(self.device)


        self.explanations = self.__pad([
                torch.tensor([self.TEXT.vocab.stoi[word] for word in phrase.split()]).to(self.device)
                for phrase in self.dictionary.keys()], explanations.max_words)

        self.dropout = nn.Dropout(model_args["dropout"])

        self.optimizer = optim.Adam(list(set(self.parameters()) - set(self.vanilla.parameters())))
        #self.criterion = nn.BCEWithLogitsLoss().to(self.device)

        self = self.to(self.device)
        super().save_model_type(self)

    def __pad(self, tensor_list, length):
        """
        0 pad to the right for a list of variable sized tensors
        e.g. [torch.tensor([1,2]), torch.tensor([1,2,3,4]),torch.tensor([1,2,3,4,5])], 5 ->
                [tensor([1, 2, 0, 0, 0]), tensor([1, 2, 3, 4, 0]), tensor([1, 2, 3, 4, 5])]
        """
        return torch.stack([torch.cat([tensor.data, tensor.new(length-tensor.size(0)).zero_()])
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
        # for class_idx, class_batch_dict in enumerate(self.expl_distributions):
        #     #  tensor [batch, dict]
        # label = list(self.dictionary.keys())
        # dictionary = self.dictionaries[label]
        for i in range(len(self.expl_distributions)):
            nlp_expl_dict = self._decode_expl_distr(self.expl_distributions[i], self.dictionary)
            nlp_text = " ".join([self.TEXT.vocab.itos[idx] for idx in (text[i])])
            val = text_expl.get(nlp_text,[])
            val.append(nlp_expl_dict)
            val.append(self.predictions[i])
            val.append(self.true_labels[i])
            val.append(self.raw_predictions[i])
            text_expl[nlp_text] = val

            # header text,list of classes
#                 f.write("text, " + ", ".join(list(self.dictionaries.keys()))+"\n")
#                 f.write("\n".join([f"{review} ~ {text_expl[review]}" for review in text_expl.keys()]))
        return text_expl

    def gen(self, activ, batch_size):
        context_vector, final_dict, expl_distributions = [], [], []
        # [dict_size, max_words, emb_dim]
        # explanations[i] -> [dict_size, max_words, emb_dim]
        v_emb_pos = self.embedding(self.explanations)
        # v_emb_neg = self.embedding(self.explanations[1])
        #[batch,dict_size, max_words, emd_dim
        vocab_emb_pos = v_emb_pos.repeat(batch_size,1,1,1)
        # vocab_emb_neg = v_emb_neg.repeat(batch_size,1,1,1)

        #[batch,dict_size, max_words* emd_dim]
        vocab_emb_pos = vocab_emb_pos.reshape(vocab_emb_pos.size(0),vocab_emb_pos.size(1),-1)
        # vocab_emb_neg = vocab_emb_neg.reshape(vocab_emb_neg.size(0),vocab_emb_neg.size(1),-1)

        # [sent, batch, dict_size]
        expl_distribution_pos = self.lin_pos(activ)
        # expl_activ_neg = self.lin_neg(activ)

        # [batch, dict_size]
        # expl_distribution_pos = torch.transpose(expl_activ_pos, 0, 1)
        # expl_distribution_neg = torch.transpose(expl_activ_neg, 0, 1)

        # [batch, max_sent, dict_size] (pad right)
        # size1, size2, size3 = expl_distribution_pos.shape[0], expl_distribution_pos.shape[1], expl_distribution_pos.shape[2]
        # if self.max_sent_len>=size2:
        #     # 0-padding
        #     expl_distribution_pos = torch.cat([expl_distribution_pos, expl_distribution_pos.new(size1, self.max_sent_len-size2, size3).zero_()],1).to(self.device)
        #     # expl_distribution_neg = torch.cat([expl_distribution_neg, expl_distribution_neg.new(size1, self.max_sent_len-size2, size3).zero_()],1).to(self.device)
        # else:
        #     # trimming
        #     expl_distribution_pos = expl_distribution_pos[:,:self.max_sent_len,:]
        #     # expl_distribution_neg = expl_distribution_neg[:,:self.max_sent_len,:]
        


        # [batch,dict_size, sent]
        # e_pos = torch.transpose(expl_distribution_pos,1,2)
        # e_neg = torch.transpose(expl_distribution_neg,1,2)
        # [batch, dict, 1]
        # expl_distribution_pos = self.aggregation_pos(e_pos).squeeze()
        # expl_distribution_neg = self.aggregation_neg(e_neg).squeeze()
        # expl_distribution = self.sigmoid(expl_distribution) # on dim 1
        
        # batch, dict
        expl_distribution_pos = F.gumbel_softmax(expl_distribution_pos, hard=True)
        # expl_distribution_neg = F.gumbel_softmax(expl_distribution_neg, hard=True)

        # expl_distribution_pos = self.softmax(expl_distribution_pos)
        # expl_distribution_neg = self.softmax(expl_distribution_neg)
        #[batch, dict]
        # expl_distributions.append(torch.squeeze(expl_distribution_pos))
        # expl_distributions.append(torch.squeeze(expl_distribution_neg))

        # [batch,1, dict]
        e_dist_pos = torch.transpose(expl_distribution_pos.unsqueeze(-1),1,2)
        # e_dist_neg = torch.transpose(expl_distribution_neg.unsqueeze(-1),1,2)
        # batch, 1, dict x batch, dict, emb (max_words*emb_dim)
        
        # ipdb.set_trace(context=10)
        expl_pos = torch.bmm(e_dist_pos, vocab_emb_pos)
        # expl_neg = torch.bmm(e_dist_neg, vocab_emb_neg)
        # import ipdb
        # ipdb.set_trace(context=10)
        #[batch,max_words,emb_dim]
        context_vector.append(expl_pos.reshape(batch_size, v_emb_pos.size(1),-1))
        # context_vector.append(torch.max(expl_neg, dim=1).values.reshape(batch_size, v_emb_neg.size(1),-1))


        sep = torch.zeros((batch_size,1,self.emb_dim), device=self.device)
        # [batch, 1+1, emb_dim]
        final_dict.append(torch.cat((sep, context_vector[0]), 1))
        # final_dict.append(torch.cat((sep, context_vector[1]), 1))

        return final_dict, expl_distribution_pos



    def forward(self, text, text_lengths, expl_file=None):

        batch_size = text.size()[1]

        #text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        #embedded = [sent len, batch size, emb dim]

        output, hidden = self.vanilla.raw_forward(embedded, text_lengths)
        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors

        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        final_dict, expl_distributions = self.gen(hidden, batch_size)
        self.expl_distributions = expl_distributions

        final_expl = final_dict[0]

        #[batch, sent, emb]
        x = torch.transpose(embedded,0,1)

        # [batch, sent_len+2, emb_dim]
        concat_input = torch.cat((x,final_expl),1) 

        #[sent_len+1, batch, emb_dim]
        final_input = torch.transpose(concat_input,0,1)
        output, hidden = self.vanilla.raw_forward(final_input, text_lengths + 2)
        return output


    def evaluate(self, iterator, prefix="test"):
        save = False # save explanations
        if prefix=="test_f":
            save = True
            expl = "text, explanation, prediction, true label\n"
            e_list = []
            # distr = [torch.tensor([]).to(self.device) for i in range(len(self.dictionaries.keys()))]
            distr = torch.tensor([]).to(self.device)
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
                    # for class_idx in range(len(distr)):
                    #     distr[class_idx] = torch.cat((distr[class_idx], self.expl_distributions[class_idx]))
                    distr = torch.cat((distr, self.expl_distributions))
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
                f.write("\n")
            with open(f"{self.explanations_path}_distr.txt", "w") as f:
                f.write(f"{torch.tensor(distr).shape}\n")
                f.write(str(distr))
                f.write("\nSUMs\n")
                f.write(str(torch.sum(torch.tensor(distr), dim=1)))
                f.write("\nHard sums\n")
                f.write(str([torch.sum(torch.where(d>0.5, torch.ones(d.shape).to(self.device), torch.zeros(d.shape).to(self.device))) for d in distr]))
                f.write("\nIndices\n")
                f.write(str(distr.nonzero().data[:,1]))

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



##########################################################################################################
############################################End of Independent training ##################################
##########################################################################################################

##########################################################################################################
############################################ MLP before frozen bi-LSTM ##################################
##########################################################################################################
class MLPBefore(MLPIndependentOneDict):
    """
    MLP + pretrained bi-LSTM
    """
    def __init__(self, id, mapping_file_location, model_args, dataset, explanations):
        super().__init__(id, mapping_file_location, model_args, dataset, explanations)
        self.relu = nn.ReLU() 
        self.lin1s = [nn.Linear(2*model_args["hidden_dim"], 2*model_args["hidden_dim"]).to(self.device) for i in range(model_args["n1"])]
  
        self.lin2 = nn.Linear(2*model_args["hidden_dim"], model_args["hidden_dim"]).to(self.device)
        self.lin3s = [nn.Linear(model_args["hidden_dim"], model_args["hidden_dim"]).to(self.device) for i in range(model_args["n3"])]
        self.lin4 = nn.Linear(model_args["hidden_dim"], len(self.dictionary.keys())).to(self.device)
        self.max_words_dict = explanations.max_words
        self.alpha = model_args['alpha']
        self.decay = model_args['alpha_decay']
        self.criterion = self.loss

        self.lin = nn.Linear(self.emb_dim, 2*model_args["hidden_dim"]).to(self.device)


    def forward(self, text, text_lengths, expl_file=None):

        batch_size = text.size()[1]

        #text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))
        #embedded = [sent len, batch size, emb dim]
        expl_activ = self.lin(embedded)
        # # expl_activ = self.lin21(embedded)
        expl_activ = self.relu(expl_activ)
        expl_activ = self.dropout(expl_activ)
        # expl_activ = self.lin2(expl_activ)
        # expl_activ = self.relu(expl_activ)
        # # expl_activ = nn.Dropout(0.2)(expl_activ)

        expl, expl_distributions = self.gen(expl_activ, batch_size)
        # final_expl = final_dict[0]

        #embedded = self.lin(embedded)
        #embedded = self.relu(embedded)
        #embedded = self.dropout(embedded)


        x = torch.transpose(embedded,0,1)
        sep = torch.zeros((batch_size,1,self.emb_dim), device=self.device)
        concat_input = torch.cat((x,torch.cat((sep, expl), 1)),1) 
        
        # [batch, sent_len+2*len(final_dict), emb_dim]
        # concat_input = torch.cat((x,final_dict),1)
        #[sent_len+len(final_dict), batch, emb_dim]
        final_input = torch.transpose(concat_input,0,1)


        output, hidden = self.vanilla.raw_forward(final_input, text_lengths+1+self.max_words_dict)
        
        self.raw_predictions = torch.sigmoid(output).squeeze().detach()
        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors

        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        self.expl_distributions = expl_distributions

        return output, (expl, x)

    def gen(self, activ, batch_size):
        context_vector, final_dict, expl_distributions = [], [], []
        # [dict_size, max_words, emb_dim]
        # explanations[i] -> [dict_size, max_words, emb_dim]
        v_emb_pos = self.embedding(self.explanations)
        #[batch,dict_size, max_words, emd_dim
        vocab_emb_pos = v_emb_pos.repeat(batch_size,1,1,1)

        #[batch,dict_size, max_words* emd_dim]
        vocab_emb_pos = vocab_emb_pos.reshape(vocab_emb_pos.size(0),vocab_emb_pos.size(1),-1)

        # activ [max_sent, batch, hid]

        for lin in self.lin1s:
            activ = lin(activ)
            activ = self.relu(activ)
            activ = self.dropout(activ)
        activ = self.lin2(activ)
        activ = self.relu(activ)
        activ = self.dropout(activ)
        for lin in self.lin3s:
            activ = lin(activ)
            activ = self.relu(activ)
            activ = self.dropout(activ)

        #maxsent, batch,dict
        expl_distribution_pos = self.lin4(activ)

        # expl_dist_pos -  sent, batch,dict -> dict batch sent
        expl_dist_pos = torch.transpose(expl_distribution_pos, 0,2)

        # dict, batch, sent
        expl_dist_pos = F.pad(expl_dist_pos, (0, self.max_sent_len-expl_distribution_pos.shape[2]))

        #  sent, batch, dict 
        expl_dist_pos = torch.transpose(expl_dist_pos, 0, 2)
        #  batch, sent, dict 
        expl_distribution_pos = torch.transpose(expl_dist_pos, 0, 1)

        # [batch, max_sent, dict_size] (pad right)
        size1, size2, size3 = expl_distribution_pos.shape[0], expl_distribution_pos.shape[1], expl_distribution_pos.shape[2]
        if self.max_sent_len>=size2:
            # 0-padding
            expl_distribution_pos = torch.cat([expl_distribution_pos, expl_distribution_pos.new(size1, self.max_sent_len-size2, size3).zero_()],1).to(self.device)
        else:
            # trimming
            expl_distribution_pos = expl_distribution_pos[:,:self.max_sent_len,:]
        #batch, max_sent, dict

        expl_distribution_pos = torch.transpose(expl_distribution_pos, 1, 2)

        #batch dict 1
        expl_distribution_pos = self.aggregation_pos(expl_distribution_pos).squeeze()

        expl_distribution_pos = F.gumbel_softmax(expl_distribution_pos, hard=True)

        # batch, 1, dict x batch, dict, emb (max_words*emb_dim)

        expl_distribution_pos = torch.unsqueeze(expl_distribution_pos, 1)
        expl_pos = torch.bmm(expl_distribution_pos, vocab_emb_pos)

        # #[batch,max_words,emb_dim]
        # context_vector.append(torch.max(expl_pos, dim=1).values.reshape(batch_size, v_emb_pos.size(1),-1))

        expl = expl_pos.reshape(batch_size, v_emb_pos.size(1),-1)

        return expl, expl_distribution_pos.squeeze()


    def loss(self, output, target, sth, sthelse, alpha=None, epoch=0):
        bce = nn.BCEWithLogitsLoss().to(self.device)
        simple_bce = nn.BCELoss().to(self.device)
        if not alpha:
            alpha = self.alpha
        # if epoch == 10:
        #     alpha = 0.25
        # elif epoch == 15:
        #     alpha = 0

        # output = torch.sigmoid(output)
        min_contributions = 1 - torch.sign(target - 0.5)*(torch.sigmoid(output)-self.raw_predictions)
        # min_contributions = abs(output-self.raw_predictions)
        # print(f"Raw BCELoss in Epoch {epoch}: {simple_bce(output, self.raw_predictions)}")
        return alpha*bce(output, target) + (1-alpha)*(torch.mean(min_contributions))


    def evaluate(self, iterator, prefix="test"):
        save = False # save explanations
        if prefix=="test_f":
            save = True
            expl = "text, explanation (freq:confidence), prediction, true label\n"
            e_list = []
            # distr = [torch.tensor([]).to(self.device) for i in range(len(self.dictionaries.keys()))]
            distr = torch.tensor([]).to(self.device)
        self.eval()
        e_loss = 0
        e_acc, e_raw_acc, e_prec, e_rec = 0,0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0
        e_contributions, e_len = 0,0
        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text
                logits, (expl_emb, text_emb) = self.forward(text, text_lengths, prefix)
                logits = logits.squeeze()
                predictions = torch.sigmoid(logits)

                e_len += len(text)
                contributions = torch.sign(batch.label - 0.5)*(predictions-self.raw_predictions)
                e_contributions += sum(contributions)


                batch.label = batch.label.to(self.device)


                loss = self.criterion(logits, batch.label, expl_emb, text_emb)


                predictions = torch.round(predictions)
                y_pred = predictions.detach().cpu().numpy()
                y_true = batch.label.cpu().numpy()
                self.predictions = y_pred
                self.true_labels = y_true
                if save:
                    text_expl= self.get_explanations(text)
                    e_list.append("\n".join([f"{review} ~ {text_expl[review]} ~ C: {contributions[i].data}" for i, review in enumerate(text_expl.keys())]))
                    # for class_idx in range(len(distr)):
                    #     distr[class_idx] = torch.cat((distr[class_idx], self.expl_distributions[class_idx]))
                    distr = torch.cat((distr, self.expl_distributions))
                acc = accuracy_score(y_true, y_pred)
                raw_acc = accuracy_score(y_true, torch.round(self.raw_predictions).cpu().numpy())
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                macrof1 = f1_score(y_true, y_pred, average='macro')
                microf1 = f1_score(y_true, y_pred, average='micro')
                wf1 = f1_score(y_true, y_pred, average='weighted')

                e_loss += loss.item()
                e_acc += acc
                e_raw_acc += raw_acc
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
                f.write("\n")
            with open(f"{self.explanations_path}_distr.txt", "w") as f:
                f.write(f"{torch.tensor(distr).shape}\n")
                f.write(str(distr))
                f.write("\nSUMs\n")
                f.write(str(torch.sum(torch.tensor(distr), dim=1)))
                f.write("\nHard sums\n")
                f.write(str([torch.sum(torch.where(d>0.5, torch.ones(d.shape).to(self.device), torch.zeros(d.shape).to(self.device))).item() for d in distr]))
                f.write("\nIndices\n")
                f.write(str(distr.nonzero().data[:,1]))


        metrics ={}
        size = len(iterator)
        metrics[f"{prefix}_loss"] = e_loss/size
        metrics[f"{prefix}_acc"] = e_acc/size
        metrics[f"{prefix}_raw_acc"] = e_raw_acc/size
        metrics[f"{prefix}_prec"] = e_prec/size
        metrics[f"{prefix}_rec"] = e_rec/size
        metrics[f"{prefix}_f1"] = e_f1/size
        metrics[f"{prefix}_macrof1"] = e_macrof1/size
        metrics[f"{prefix}_microf1"] = e_microf1/size
        metrics[f"{prefix}_weightedf1"] = e_wf1/size
        metrics[f"{prefix}_avg_contributions"] = (e_contributions/e_len).item()
        return metrics

    def train_model(self, iterator, epoch):
        """
        metrics.keys(): [train_acc, train_loss, train_prec,
                        train_rec, train_f1, train_macrof1,
                        train_microf1, train_weightedf1]
        e.g. metrics={"train_acc": 90.0, "train_loss": 0.002}
        """
        e_loss = 0
        e_acc, e_raw_acc, e_prec, e_rec = 0,0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0
        e_contributions, e_len = 0, 0 

        count = 0
        batch_raw_accs = []
        self.train()
        for batch in iterator:
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
            logits, (expl, emb_text) = self.forward(text, text_lengths)
            logits=logits.squeeze()

            if count < 3 and epoch<3:
               with open(f"debug/batch-{count}-e{epoch}", "w") as f:
                    f.write(str(text))
                    f.write("\n~\n")
                    f.write(str(self.raw_predictions))
                    f.write("\n\n**\n\n")
            count += 1
            
            batch.label = batch.label.to(self.device)
            loss = self.criterion(logits, batch.label, expl, emb_text, self.alpha - self.decay * epoch, epoch)
            y_pred = torch.round(torch.sigmoid(logits)).detach().cpu().numpy()
            y_true = batch.label.cpu().numpy()

            e_len += len(text)
            e_contributions += sum(torch.sign(batch.label - 0.5)*(torch.sigmoid(logits)-self.raw_predictions))

            #metrics
            raw_acc = accuracy_score(y_true, torch.round(self.raw_predictions).cpu().numpy())
            
            batch_raw_accs.append(raw_acc)
            
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
            e_raw_acc += raw_acc
            e_prec += prec
            e_rec += rec
            e_f1 += f1
            e_macrof1 += macrof1
            e_microf1 += microf1
            e_wf1 += wf1

        with open(f"debug/train-raw-accs-{epoch}.txt", "w") as f:
            f.write(str(batch_raw_accs))
            f.write("\n\n**\n\n")
        metrics ={}
        size = len(iterator)
        metrics["train_loss"] = e_loss/size
        metrics["train_acc"] = e_acc/size
        metrics["train_raw_acc"] = e_raw_acc/size
        metrics["train_prec"] = e_prec/size
        metrics["train_rec"] = e_rec/size
        metrics["train_f1"] = e_f1/size
        metrics["train_macrof1"] = e_macrof1/size
        metrics["train_microf1"] = e_microf1/size
        metrics["train_weightedf1"] = e_wf1/size
        metrics["train_avg_contributions"] = (e_contributions/e_len).item()

        return metrics



##########################################################################################################
############################################biLSTM  + MLP  + similarity ##################################
##########################################################################################################

from math import log

class MLPAfterIndependentOneDictSimilarity(AbstractModel):
    """
    pretrained bi-LSTM + MLP
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
        self.alpha = model_args['alpha']
        self.decay = model_args['alpha_decay']
        self.explanations_path = os.path.join(self.model_dir, model_args["dirs"]["explanations"], "e")

        if model_args["restore_v_checkpoint"]:
            vanilla_args = copy.deepcopy(model_args)
            vanilla_args["restore_checkpoint"] = True
            self.vanilla = FrozenVLSTM("frozen-bi-lstm", mapping_file_location, vanilla_args)
            print(model_args["checkpoint_v_file"])
            self.vanilla.load_checkpoint(model_args["checkpoint_v_file"])
            print(f"Vanilla frozen, params {len(list(self.vanilla.parameters()))}: {[name for name, param in self.vanilla.named_parameters()]}")
            for param in self.vanilla.parameters():
                param.requires_grad=False
            self.vanilla.eval()
        else:
            self.vanilla = FrozenVLSTM("bi-lstm", mapping_file_location, model_args)


        self.TEXT = dataset.TEXT

        self.max_sent_len = dataset.max_sent_len
        # UNK_IDX = dataset.TEXT.vocab.stoi[dataset.TEXT.unk_token]
        PAD_IDX = dataset.TEXT.vocab.stoi[dataset.TEXT.pad_token]
        self.input_size = len(dataset.TEXT.vocab)
        self.embedding = nn.Embedding(self.input_size, model_args["emb_dim"], padding_idx=PAD_IDX)
        
        nn.init.uniform_(self.embedding.weight.data,-1,1)

        self.emb_dim = model_args["emb_dim"]


        dictionaries = explanations.get_dict()
        start = datetime.now()
        formated_date = start.strftime(DATE_FORMAT)
        self.dictionary = copy.deepcopy(dictionaries['pos'])
        self.dictionary.update(dictionaries['neg'])
        with open(f"dict-{formated_date}", "w") as f:
            f.write(f"{dictionaries}")
        print("Dict size", len(self.dictionary.keys()))
        self.lin1s = [nn.Linear(2*model_args["hidden_dim"], 2*model_args["hidden_dim"]).to(self.device) for i in range(model_args["n1"])]
        self.relu = nn.ReLU() 
        self.lin2 = nn.Linear(2*model_args["hidden_dim"], model_args["hidden_dim"]).to(self.device)
        self.lin3s = [nn.Linear(model_args["hidden_dim"], model_args["hidden_dim"]).to(self.device) for i in range(model_args["n3"])]
        self.lin4 = nn.Linear(model_args["hidden_dim"], len(self.dictionary.keys())).to(self.device)

        self.explanations = self.__pad([
                torch.tensor([self.TEXT.vocab.stoi[word] for word in phrase.split()]).to(self.device)
                for phrase in self.dictionary.keys()], explanations.max_words)

        start = datetime.now()
        formated_date = start.strftime(DATE_FORMAT)
        with open(f"explanations_dict-MLPAfterIndependentOneDictSimilarity-{formated_date}", "w") as f:
            f.write(str(self.dictionary.keys()))
            f.write("\n\n\n**\n\n\n")
            print("explanations file")

        self.dropout = nn.Dropout(model_args["dropout"])

        self.optimizer = optim.AdamW(list(set(self.parameters()) - set(self.vanilla.parameters())), lr=model_args["lr"], weight_decay=model_args["l2_wd"])
        # self.optimizer = optim.Adam(list(set(self.parameters()) - set(self.vanilla.parameters())))
        self.criterion = self.loss

        self = self.to(self.device)
        super().save_model_type(self)

    def __pad(self, tensor_list, length):
        """
        0 pad to the right for a list of variable sized tensors
        e.g. [torch.tensor([1,2]), torch.tensor([1,2,3,4]),torch.tensor([1,2,3,4,5])], 5 ->
                [tensor([1, 2, 0, 0, 0]), tensor([1, 2, 3, 4, 0]), tensor([1, 2, 3, 4, 5])]
        """
        return torch.stack([torch.cat([tensor.data, tensor.new(length-tensor.size(0)).zero_()])
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
        # for class_idx, class_batch_dict in enumerate(self.expl_distributions):
        #     #  tensor [batch, dict]
        # label = list(self.dictionary.keys())
        # dictionary = self.dictionaries[label]
        for i in range(len(self.expl_distributions)):
            nlp_expl_dict = self._decode_expl_distr(self.expl_distributions[i], self.dictionary)
            nlp_text = " ".join([self.TEXT.vocab.itos[idx] for idx in (text[i])])
            val = text_expl.get(nlp_text,[])
            val.append(nlp_expl_dict)
            val.append(self.predictions[i])
            val.append(self.true_labels[i])
            val.append(self.raw_predictions[i])
            text_expl[nlp_text] = val

            # header text,list of classes
#                 f.write("text, " + ", ".join(list(self.dictionaries.keys()))+"\n")
#                 f.write("\n".join([f"{review} ~ {text_expl[review]}" for review in text_expl.keys()]))
        return text_expl

    def gen(self, activ, batch_size):
        context_vector, final_dict, expl_distributions = [], [], []
        # [dict_size, max_words, emb_dim]
        # explanations[i] -> [dict_size, max_words, emb_dim]
        v_emb_pos = self.embedding(self.explanations)
        # v_emb_neg = self.embedding(self.explanations[1])
        #[batch,dict_size, max_words, emd_dim
        vocab_emb_pos = v_emb_pos.repeat(batch_size,1,1,1)
        # vocab_emb_neg = v_emb_neg.repeat(batch_size,1,1,1)

        #[batch,dict_size, max_words* emd_dim]
        vocab_emb_pos = vocab_emb_pos.reshape(vocab_emb_pos.size(0),vocab_emb_pos.size(1),-1)
        # vocab_emb_neg = vocab_emb_neg.reshape(vocab_emb_neg.size(0),vocab_emb_neg.size(1),-1)

        # [sent, batch, dict_size]
        # activ = self.dropout(activ)
        for lin in self.lin1s:
            activ = lin(activ)
            activ = self.relu(activ)
            activ = self.dropout(activ)
        activ = self.lin2(activ)
        activ = self.relu(activ)
        activ = self.dropout(activ)
        for lin in self.lin3s:
            activ = lin(activ)
            activ = self.relu(activ)
            activ = self.dropout(activ)

        expl_distribution_pos = self.lin4(activ)

        
        # expl_activ_neg = self.lin_neg(activ)

        # [batch,dict_size, sent]
        # e_pos = torch.transpose(expl_distribution_pos,1,2)
        # e_neg = torch.transpose(expl_distribution_neg,1,2)
        # [batch, dict, 1]
        # expl_distribution_pos = self.aggregation_pos(e_pos).squeeze()
        # expl_distribution_neg = self.aggregation_neg(e_neg).squeeze()
        # expl_distribution = self.sigmoid(expl_distribution) # on dim 1
        
        # batch, dict
        expl_distribution_pos = F.gumbel_softmax(expl_distribution_pos, hard=True)
        # expl_distribution_neg = F.gumbel_softmax(expl_distribution_neg, hard=True)

        # expl_distribution_pos = self.softmax(expl_distribution_pos)
        # expl_distribution_neg = self.softmax(expl_distribution_neg)
        #[batch, dict]
        # expl_distributions.append(torch.squeeze(expl_distribution_pos))
        # expl_distributions.append(torch.squeeze(expl_distribution_neg))

        # [batch,1, dict]
        e_dist = torch.transpose(expl_distribution_pos.unsqueeze(-1),1,2)
        # e_dist_neg = torch.transpose(expl_distribution_neg.unsqueeze(-1),1,2)
        # batch, 1, dict x batch, dict, emb (max_words*emb_dim)
        
        # ipdb.set_trace(context=10)
        expl_pos = torch.bmm(e_dist, vocab_emb_pos)
        # expl_neg = torch.bmm(e_dist_neg, vocab_emb_neg)
        # import ipdb
        # ipdb.set_trace(context=10)
        #[batch,max_words,emb_dim]
        expl = expl_pos.reshape(batch_size, v_emb_pos.size(1),-1)
        # context_vector.append(torch.max(expl_neg, dim=1).values.reshape(batch_size, v_emb_neg.size(1),-1))


        # [batch, 1+1, emb_dim]
        # final_dict.append(torch.cat((sep, context_vector[0]), 1))
        # final_dict.append(torch.cat((sep, context_vector[1]), 1))

        return expl, expl_distribution_pos



    def forward(self, text, text_lengths, expl_file=None):

        batch_size = text.size()[1]

        #text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        #embedded = [sent len, batch size, emb dim]
        self.vanilla.eval()
        output, hidden = self.vanilla.raw_forward(embedded, text_lengths)
        self.raw_predictions = torch.sigmoid(output).squeeze().detach()
        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors

        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        expl_emb, expl_distributions = self.gen(hidden, batch_size)
        self.expl_distributions = expl_distributions

        #[batch, sent, emb]
        x = torch.transpose(embedded,0,1)

        sep = torch.zeros((batch_size,1,self.emb_dim), device=self.device)

        # [batch, sent_len+2, emb_dim]
        concat_input = torch.cat((x,torch.cat((sep, expl_emb), 1)),1) 

        #[sent_len+1, batch, emb_dim]
        final_input = torch.transpose(concat_input,0,1)
        output, hidden = self.vanilla.raw_forward(final_input, text_lengths + 2)


        return output, (expl_emb, x) # batch, words, emb


    def evaluate(self, iterator, prefix="test"):
        save = False # save explanations
        if prefix=="test_f":
            save = True
            expl = "text, explanation (freq:confidence), prediction, true label, raw prediction\n"
            e_list = []
            # distr = [torch.tensor([]).to(self.device) for i in range(len(self.dictionaries.keys()))]
            distr = torch.tensor([]).to(self.device)
        self.eval()
        self.vanilla.eval()
        e_loss = 0
        e_acc, e_raw_acc, e_prec, e_rec = 0,0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0
        e_contributions, e_len = 0,0
        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text
                logits, (expl_emb, text_emb) = self.forward(text, text_lengths, prefix)
                logits = logits.squeeze()
                predictions = torch.sigmoid(logits)

                e_len += len(text)
                contributions = torch.sign(batch.label - 0.5)*(predictions-self.raw_predictions)
                e_contributions += sum(contributions)


                batch.label = batch.label.to(self.device)


                loss = self.criterion(logits, batch.label, expl_emb, text_emb)


                predictions = torch.round(predictions)
                y_pred = predictions.detach().cpu().numpy()
                y_true = batch.label.cpu().numpy()
                self.predictions = y_pred
                self.true_labels = y_true
                if save:
                    text_expl= self.get_explanations(text)
                    e_list.append("\n".join([f"{review} ~ {text_expl[review]} ~ C: {contributions[i].data}" for i, review in enumerate(text_expl.keys())]))
                    # for class_idx in range(len(distr)):
                    #     distr[class_idx] = torch.cat((distr[class_idx], self.expl_distributions[class_idx]))
                    distr = torch.cat((distr, self.expl_distributions))
                acc = accuracy_score(y_true, y_pred)
                raw_acc = accuracy_score(y_true, torch.round(self.raw_predictions).cpu().numpy())
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                macrof1 = f1_score(y_true, y_pred, average='macro')
                microf1 = f1_score(y_true, y_pred, average='micro')
                wf1 = f1_score(y_true, y_pred, average='weighted')

                e_loss += loss.item()
                e_acc += acc
                e_raw_acc += raw_acc
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
                f.write("\n")
            with open(f"{self.explanations_path}_distr.txt", "w") as f:
                f.write(f"{torch.tensor(distr).shape}\n")
                f.write(str(distr))
                f.write("\nSUMs\n")
                f.write(str(torch.sum(torch.tensor(distr), dim=1)))
                f.write("\nHard sums\n")
                f.write(str([torch.sum(torch.where(d>0.5, torch.ones(d.shape).to(self.device), torch.zeros(d.shape).to(self.device))).item() for d in distr]))
                f.write("\nIndices\n")
                f.write(str(distr.nonzero().data[:,1]))


        metrics ={}
        size = len(iterator)
        metrics[f"{prefix}_loss"] = e_loss/size
        metrics[f"{prefix}_acc"] = e_acc/size
        metrics[f"{prefix}_raw_acc"] = e_raw_acc/size
        metrics[f"{prefix}_prec"] = e_prec/size
        metrics[f"{prefix}_rec"] = e_rec/size
        metrics[f"{prefix}_f1"] = e_f1/size
        metrics[f"{prefix}_macrof1"] = e_macrof1/size
        metrics[f"{prefix}_microf1"] = e_microf1/size
        metrics[f"{prefix}_weightedf1"] = e_wf1/size
        metrics[f"{prefix}_avg_contributions"] = (e_contributions/e_len).item()
        return metrics


    def train_model(self, iterator, epoch):
        """
        metrics.keys(): [train_acc, train_loss, train_prec,
                        train_rec, train_f1, train_macrof1,
                        train_microf1, train_weightedf1]
        e.g. metrics={"train_acc": 90.0, "train_loss": 0.002}
        """
        self.vanilla.eval()
        e_loss = 0
        e_acc, e_raw_acc, e_prec, e_rec = 0,0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0
        e_contributions, e_len = 0, 0 

        batch_raw_accs = []
        self.train()
        for batch in iterator:
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
            logits, (expl, emb_text) = self.forward(text, text_lengths)
            logits=logits.squeeze()

            
            batch.label = batch.label.to(self.device)
            loss = self.criterion(logits, batch.label, expl, emb_text, self.alpha - self.decay * epoch, epoch)
            y_pred = torch.round(torch.sigmoid(logits)).detach().cpu().numpy()
            y_true = batch.label.cpu().numpy()

            e_len += len(text)
            e_contributions += sum(torch.sign(batch.label - 0.5)*(torch.sigmoid(logits)-self.raw_predictions))

            #metrics
            raw_acc = accuracy_score(y_true, torch.round(self.raw_predictions).cpu().numpy())
            
            batch_raw_accs.append(raw_acc)
            
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
            e_raw_acc += raw_acc
            e_prec += prec
            e_rec += rec
            e_f1 += f1
            e_macrof1 += macrof1
            e_microf1 += microf1
            e_wf1 += wf1

        with open(f"debug/train-raw-accs-{epoch}.txt", "w") as f:
            f.write(str(batch_raw_accs))
            f.write("\n\n**\n\n")
        metrics ={}
        size = len(iterator)
        metrics["train_loss"] = e_loss/size
        metrics["train_acc"] = e_acc/size
        metrics["train_raw_acc"] = e_raw_acc/size
        metrics["train_prec"] = e_prec/size
        metrics["train_rec"] = e_rec/size
        metrics["train_f1"] = e_f1/size
        metrics["train_macrof1"] = e_macrof1/size
        metrics["train_microf1"] = e_microf1/size
        metrics["train_weightedf1"] = e_wf1/size
        metrics["train_avg_contributions"] = (e_contributions/e_len).item()

        return metrics

    def loss(self, output, target, explanation, x_emb, alpha=None, epoch=0):
      bce = nn.BCEWithLogitsLoss().to(self.device)
      if not alpha:
        alpha = self.alpha
      text = torch.mean(x_emb, dim=1).squeeze()
      expl = torch.mean(explanation, dim=1).squeeze()
      cos = nn.CosineSimilarity(dim=1)
      semantic_cost = 1-cos(text, expl)
      # loss = alpha*bce(output, target) + (1-alpha)*torch.mean(semantic_cost)
      loss = bce(output, target) + torch.mean(semantic_cost)
      return loss

class MLPAfterIndependentOneDictImprove(MLPAfterIndependentOneDictSimilarity):

    def loss(self, output, target, sth, sthelse, alpha=None, epoch=0):
        bce = nn.BCEWithLogitsLoss().to(self.device)
        simple_bce = nn.BCELoss().to(self.device)
        if not alpha:
            alpha = self.alpha
        # if epoch == 10:
        #     alpha = 0.25
        # elif epoch == 15:
        #     alpha = 0

        # output = torch.sigmoid(output)
        min_contributions = 1 - torch.sign(target - 0.5)*(torch.sigmoid(output)-self.raw_predictions)
        # min_contributions = abs(output-self.raw_predictions)
        # print(f"Raw BCELoss in Epoch {epoch}: {simple_bce(output, self.raw_predictions)}")
        return alpha*bce(output, target) + (1-alpha)*(torch.mean(min_contributions))


class LSTMAfterIndependentOneDictImprove(MLPAfterIndependentOneDictSimilarity):

    def loss(self, output, target, sth, sthelse, alpha=None, epoch=0):
        bce = nn.BCEWithLogitsLoss().to(self.device)
        if not alpha:
            alpha = self.alpha
        # if epoch == 10:
        #     alpha = 0.25
        # elif epoch == 15:
        #     alpha = 0

        # output = torch.sigmoid(output)
        min_contributions = 1 - torch.sign(target - 0.5)*(torch.sigmoid(output)-self.raw_predictions)
        # min_contributions = abs(output-self.raw_predictions)
        return alpha*bce(output, target) + (1-alpha)*(mean(min_contributions))
        # return alpha*bce(output, target) + (1-alpha)*(sum(min_contributions)/10)

##########################################################################################################
############################################End of MLP before training ##################################
##########################################################################################################



import argparse
from datetime import datetime

try:
    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)


    parser = argparse.ArgumentParser(description='Config params.')
    parser.add_argument('-p', metavar='max_words_dict', type=int, default=CONFIG["max_words_dict"],
                        help='Max number of words per phrase in explanations dictionary')

    parser.add_argument('-n1', metavar='mlp_depth', type=int, default=1,
                        help='Number of deep layers of the DNN generator - 2*hid->2*hid')

    parser.add_argument('-n2', metavar='mlp_depth', type=int, default=1,
                        help='Number of deep layers of the DNN generator - 2*hid->1*hid')

    parser.add_argument('-n3', metavar='mlp_depth', type=int, default=1,
                        help='Number of deep layers of the DNN generator - 1*hid->1*hid')



    parser.add_argument('-dr', metavar='dropout', type=float, default=CONFIG["dropout"],
                        help='Dropout value')

    parser.add_argument('-a', metavar='alpha', type=float, default=CONFIG["alpha"],
                        help='Similarity cost hyperparameter')
    

    parser.add_argument('-hd', metavar='hidden_dim', type=int, default=256,
                        help='LSTM hidden dim')

    parser.add_argument('-nl', metavar='num_layers', type=int, default=2,
                        help='LSTM num_layers')


    parser.add_argument('-decay', metavar='decay', type=float, default=0, help='alpha decay')

    parser.add_argument('-d', metavar='dictionary_type', type=str, default=None,
                        help='Dictionary type: tfidf, rake-inst, rake-corpus, textrank, yake')

    parser.add_argument('-m', metavar='model_type', type=str,
                        help='frozen_mlp_bilstm, frozen_bilstm_mlp, bilstm_mlp_similarity')

    parser.add_argument('-e', metavar='epochs', type=int, default=CONFIG["epochs"],
                        help='Number of epochs')

    parser.add_argument('-lr', metavar='learning_rate', type=float, default=CONFIG["lr"], help='Optimizer\'s lr')

    parser.add_argument('-l2', metavar='L2 weight decay', type=float, default=CONFIG["l2_wd"], help='L2 weight decay optimizer')

    parser.add_argument('--td',  action='store_true',
                        help='Toy data (load just a small data subset)')

    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--train', dest='train', action='store_true')
    # parser.add_argument('--no_train', dest='train', action='store_false')
    # parser.set_defaults(train=CONFIG["train"])

    # parser.add_argument('--restore', dest='restore', action='store_true')
    # parser.set_defaults(restore=CONFIG["restore_checkpoint"])

    # parser.add_argument('--cuda', type=bool, default=CONFIG["cuda"])
    args = parser.parse_args()


    """# Main"""


    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
        "hidden_dim": args.hd,
        "n_layers": args.nl,
        "max_dict": 1000, 
        "cuda": True,
        "restore_v_checkpoint" : True,
        # "checkpoint_v_file": "experiments/gumbel-seed-true/v-lstm/snapshot/2020-04-10_15-04-57_e2",
        "checkpoint_v_file" :"experiments/soa-dicts/vanilla-lstm-n2-h256-dr0.5/snapshot/2020-06-16_22-06-00_e5",
        # "checkpoint_v_file": "experiments/soa-dicts/vanilla-lstm-n1-h64-dr0.05/snapshot/2020-06-16_19-33-50_e4",
        "train": True,
        "max_words_dict": args.p,
        "patience":20,
        "epochs": args.e,
        'alpha': args.a,
        "n1": args.n1,
        "n2": args.n2,
        "n3": args.n3,
        "alpha_decay": args.decay,
        "dropout": args.dr, 
        "load_dictionary":True,
        #"dict_checkpoint": "experiments/independent/dictionaries/rake-polarity/dictionary.h5",
        # "dict_checkpoint": "experiments/dict_acquisition/dictionaries/rake-max-words-instance-300-4/dictionary-2020-06-02_16-00-44.h5",
        #"dict_checkpoint":"experiments/dict_acquisition/dictionaries/textrank-filtered_True-p5-d300/dictionary-2020-06-05_14-56-57.h5",
        "dict_checkpoint": "experiments/dict_acquisition/dictionaries/rake-instance-600-4-filteredTrue/dictionary-2020-06-07_23-18-09.h5",
        "toy_data": args.td,
        "lr": args.lr,
        "l2_wd": args.l2, 
        "filterpolarity": True,
        "phrase_len":4,
        "id":args.m,
        "train": not args.eval,
        "restore_checkpoint" : args.eval,
        "checkpoint_file": "experiments/soa-dicts/bilstm_mlp_improve_15-25_l20.1_dr0.5_soa_vlstm2-256-0.5_pretrained_rake-4-600-dnn15-1-25-decay0.0-L2-dr0.5-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e30-2020-06-17_14-52-49/snapshot/2020-06-17_16-02-07_e6"
    })
    print(experiment.config)

    start = datetime.now()
    dataset = IMDBDataset(experiment.config)
    print(f"Time data load: {str(datetime.now()-start)}")

    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    if args.d=="tfidf":
        explanations = TFIDF("tfidf", dataset, experiment.config)
    elif args.d=="yake":
        explanations = DefaultYAKE("default-yake", dataset, experiment.config)
    elif args.d=="textrank":
        explanations = TextRank(f"textrank-filtered", dataset, experiment.config)
    elif args.d == "rake-inst":
        explanations = RakeInstanceExplanations(f"rake-max-words-instance-{CONFIG['max_words_dict']}-{args.p}-filtered", dataset, experiment.config)
        # explanations = RakeMaxWordsPerInstanceExplanations(f"rake-max-words-instance-300-{args.p}", dataset, experiment.config)
    elif args.d == "rake-corpus":
        explanations = RakeMaxWordsExplanations(f"rake-max-words-corpus-300-{args.p}", dataset, experiment.config)
    elif args.d == "rake-polarity":
        explanations = RakeCorpusPolarityFiltered(f"rake-polarity", dataset, experiment.config)
    elif args.d == None:
        explanations = None
    print(f"Dict {args.d}")
    if explanations:
        d = explanations.get_dict()
        print(str(d.keys()))
        print(str(d.items()))
    
        with open(f"dict/{args.d}-{args.p}-{formated_date}", "w") as f:
            for key in d.keys():
                f.write(f"{key}\n**\n")
                f.write("\n".join([f"{e} ~ {f}" for e,f in d[key].items()]))
                f.write(f"\n\n------------\n\n")
        print(f"Time expl dictionary {args.d} - max-phrase {args.p}: {str(datetime.now()-start)}")

    start = datetime.now()
    formated_date = start.strftime(DATE_FORMAT)
    if "mlp_bilstm" in args.m:
        model = MLPBefore(f"{args.m}-d{args.d}-p{args.p}dnn{args.n1}-{args.n3}-decay{args.decay}-L2{args.l2}-lr{args.lr}-dr{args.dr}-improveloss_mean-alpha{args.a}-e{args.e}-{formated_date}", MODEL_MAPPING, experiment.config, dataset, explanations)
    elif args.m =="frozen_bilstm_mlp":
        model = MLPIndependentOneDict(f"{args.m}-super-patient-{args.d}-{args.p}", MODEL_MAPPING, experiment.config, dataset, explanations)
    elif args.m =="bilstm_mlp_similarity":
        model = MLPAfterIndependentOneDictSimilarity(f"{args.m}-dnn{args.n1}-{args.n2}-{args.n3}-decay{args.decay}-L2-dr{args.dr}-eval1-{args.d}-sumloss-c", MODEL_MAPPING, experiment.config, dataset, explanations)
    elif "bilstm_mlp_improve" in args.m:
        model = MLPAfterIndependentOneDictImprove(f"{args.m}-dnn{args.n1}-{args.n2}-{args.n3}-decay{args.decay}-L2-dr{args.dr}-eval1-{args.d}-4-600-improveloss_mean-alpha{args.a}-c-e{args.e}-{formated_date}", MODEL_MAPPING, experiment.config, dataset, explanations)
    elif args.m == "bilstm":
        model = FrozenVLSTM(f"vanilla-lstm-n{experiment.config['n_layers']}-h{experiment.config['hidden_dim']}-dr{experiment.config['dropout']}", MODEL_MAPPING, experiment.config)
        

    experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()
    print(f"Time model training: {str(datetime.now()-start)}")
    # start = datetime.now()
    # formated_date = start.strftime(DATE_FORMAT)
    # model = VLSTM("v-lstm", MODEL_MAPPING, experiment.config)
    # experiment.with_data(dataset).with_model(model).run()

    # print(f"Time: {str(datetime.now()-start)}")

except:
    import sys, traceback
    traceback.print_exc(file=sys.stdout)
