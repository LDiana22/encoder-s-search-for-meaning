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
PREFIX_DIR = "experiments/independent"
MODEL_MAPPING = "experiments/model_mappings/independent"

MODEL_NAME = "mlp+frozen_bilstm_gumb-emb-one-dict-long"


CONFIG = {
    "toy_data": False, # load only a small subset

    "cuda": True,

    "embedding": "glove",

    "restore_checkpoint" : True,
    "checkpoint_file": "experiments/v-lstm",
    "train": True,

    "dropout": 0.05,
    "weight_decay": 5e-06,

    "patience": 10,

    "epochs": 200,

    "objective": "cross_entropy",
    "init_lr": 0.0001,

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
        plot_path = self.model.get_plot_path("train_plot")
        print(f"Plotting at {plot_path}")
        plot_training_metrics(plot_path, training_losses, v_losses, training_acc, v_acc)
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
    self.train_data, self.valid_data = self.train_data.split(random_state=random.getstate())
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


import argparse
from datetime import datetime


start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)



"""# Main"""


start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)
experiment = Experiment(f"e-v-{formated_date}").with_config(CONFIG).override({
    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 300, 
    "cuda": True,
    "restore_checkpoint" : True,
    "checkpoint_file": "experiments/gumbel-seed-true/v-lstm/snapshot/2020-04-10_15-04-57_e2",
    "train": False,
    "patience":300,
    "epochs":300
})
print(experiment.config)

start = datetime.now()
dataset = IMDBDataset(experiment.config)
print(f"Time data load: {str(datetime.now()-start)}")


start = datetime.now()

test_instances = dataset.test_data
model = VLSTM("loaded", MODEL_MAPPING, experiment.config)
experiment.with_data(dataset).with_model(model).run()

print(f"Time model load: {str(datetime.now()-start)}")




def predict_sentiment(sentences):
    model.eval()
    predictions = []
    for sentence in sentences:
      tokenized = [tok for tok in sentence.split(" ")]
      if not tokenized:
        tokenized = [""]
      indexed = [dataset.TEXT.vocab.stoi[t] for t in tokenized]
      length = [len(indexed)]
      tensor = torch.LongTensor(indexed).to(model.device)
      tensor = tensor.unsqueeze(1)
      length_tensor = torch.LongTensor(length)
      prediction = torch.sigmoid(model(tensor, length_tensor)).item()
      predictions.append([1-prediction, prediction])
    return np.array(predictions)


from lime import lime_text
explainer = lime_text.LimeTextExplainer(class_names=["pos", "neg"])

LIME_LIST = "experiments/lime/v-lstm/word_list_test_set.txt"
LIME_HTML = "experiments/lime/v-lstm/html/test"

start = datetime.now()

with open(LIME_LIST, "w") as pos:
  for i,example in enumerate(test_instances.examples):
    if i == 10:
      break
    explanations = explainer.explain_instance(" ".join(example.text), predict_sentiment, labels=[1])
    pos_expl = explanations.as_list(1)
    pos.write(f"{i}\n\n{' '.join(example.text)}-\n\n{pos_expl}\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    explanations.save_to_file(f"{LIME_HTML}-{i}.html")


print(f"Time LIME explain test set: {str(datetime.now()-start)}") #1.41.40