import pandas as pd
import os
import re
import argparse
from datetime import datetime
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
##################### LOAD RAW PREDICTOR #########################
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
import random
import io
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.enabled = False 
torch.cuda.manual_seed_all(0)

torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

checkpoint = "experiments/soa-dicts/vanilla-lstm-n2-h256-dr0.5/snapshot/2020-06-16_22-06-00_e5"
start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)
PREFIX_DIR = "experiments/soa-dicts"
MODEL_MAPPING = "experiments/soa-dicts/model_mapping"

IMDB_PATH = "../.data/imdb/aclImdb"
CONFIG={

    "cuda": True,

    "weight_decay": 5e-06,

    "patience": 10,

    "objective": "cross_entropy",
    "lr": 0.001,
    "l2_wd": 0,

    "gumbel_decay": 1e-5,

    "prefix_dir" : PREFIX_DIR,

    "dirs": {
        "metrics": "metrics",
        "checkpoint": "snapshot",
        "dictionary": "dictionaries",
        "explanations": "explanations",
        "plots": "plots"
        },

    "max_vocab_size": 100000,
    "emb_dim": 300,
    "batch_size": 32,
        "hidden_dim": 256,
        "output_dim":1,
        "emb": None,
        "n_layers": 2,
        "max_dict": 1000, 
        "cuda": True,
        "restore_v_checkpoint" : True,
        # "checkpoint_v_file": "experiments/gumbel-seed-true/v-lstm/snapshot/2020-04-10_15-04-57_e2",
        "checkpoint_v_file" :"experiments/soa-dicts/vanilla-lstm-n2-h256-dr0.5/snapshot/2020-06-16_22-06-00_e5",
        #"checkpoint_v_file": "experiments/soa-dicts/vanilla-lstm-n1-h64-dr0.05/snapshot/2020-06-16_19-33-50_e4",
        #"checkpoint_v_file": "experiments/soa-dicts/vanilla-lstm-n2-h64-dr0.3/snapshot/2020-06-24_09-58-30_e4",
        #"checkpoint_v_file": args.cp,
        "train": True,
        "max_words_dict": 4,
        "patience": 5,
        "epochs": 0,
        "alpha": 0,
        "beta":0,
        "gamma":0,
        "n1": 30,
        "n2": 1,
        "n3": 30,
        "alpha_decay": 0.01,
        "dropout": 0.5, 
        "load_dictionary":True,
        "dict_checkpoint": "experiments/dictionaries_load/dictionaries/rake-inst-unsorted-dist100-600-600-4-filteredTrue/dictionary-2020-07-09_23-31-43.h5",
        #"dict_checkpoint": "experiments/dictionaries_load/dictionaries/test-rake-corpus-600-4-filtered/dictionary-2020-07-07_18-18-56.h5",
        # "dict_checkpoint": "experiments/independent/dictionaries/rake-polarity/dictionary.h5",
        # "dict_checkpoint": "experiments/dict_acquisition/dictionaries/rake-max-words-instance-300-4/dictionary-2020-06-02_16-00-44.h5",
        #"dict_checkpoint":"experiments/dict_acquisition/dictionaries/textrank-filtered_True-p5-d300/dictionary-2020-06-05_14-56-57.h5",
        #"dict_checkpoint": "experiments/dict_acquisition/dictionaries/rake-instance-600-4-filteredTrue/dictionary-2020-06-07_23-18-09.h5",
        "toy_data": False,
        "lr": 0.01,
        "l2_wd": 0.01, 
        "filterpolarity": True,
        "phrase_len":4,
        "id": "vanilla",
        "train": False,
        "restore_checkpoint" : True,
        "checkpoint_file": "experiments/soa-dicts/bilstm_mlp_improve_15-25_l20.1_dr0.5_soa_vlstm2-256-0.5_pretrained_rake-4-600-dnn15-1-25-decay0.0-L2-dr0.5-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e30-2020-06-17_14-52-49/snapshot/2020-06-17_16-02-07_e6"
    }


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
    # self.test_data = self._load_data(TEXT, LABEL, IMDB_PATH, "test")
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
    # print(f"Test {len(self.test_data)}")
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
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
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

        #UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        #PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        
        
        self.input_size = model_args["max_vocab_size"]
        self.embedding = nn.Embedding(self.input_size, model_args["emb_dim"])
        if model_args["emb"]=="glove":

            self.embedding.weight.data.copy_(model_args["vectors"])
            self.embedding.weight.data[model_args["unk_idx"]] = torch.zeros(model_args["emb_dim"])
            self.embedding.weight.data[model_args["pad_idx"]] = torch.zeros(model_args["emb_dim"])
        else:
            nn.init.uniform_(self.embedding.weight.data,-1,1)


        self.lstm = nn.LSTM(model_args["emb_dim"], 
                           model_args["hidden_dim"], 
                           num_layers=model_args["n_layers"], 
                           bidirectional=True, 
                           dropout=model_args["dropout"])
        self.lin = nn.Linear(2*model_args["hidden_dim"], model_args["output_dim"])
        self.dropout = nn.Dropout(model_args["dropout"])

        #self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
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

dataset = IMDBDataset(CONFIG)

model = FrozenVLSTM(f"raw-pred-vanilla-lstm", MODEL_MAPPING, CONFIG)

model.load_checkpoint(checkpoint)
model.eval()
import spacy
nlp = spacy.load('en')

VANILLA_CACHE = "vanilla-predictions.csv"

def compute_vanilla_preds(df):
    df.to_csv(VANILLA_CACHE, sep="~")

def load_vanilla():
    return pd.read_csv(VANILLA_CACHE, sep="~", names=["review", "vanilla_prediction"])


def prepare_text_for_classification(texts):
    return [torch.LongTensor([dataset.TEXT.vocab.stoi[t] for t in [tok.text for tok in nlp.tokenizer(text)]]) for text in texts]

##################################################################
def fix_file(path):
    new_path = path[:path.rindex(".")]+"_fix.txt"
    print(f"New path {new_path}")
    with open(new_path, "w") as f:
        with open(path, "r") as g:
            line = g.readline()
            count = 0
            while line:
                count += 1
                if len(line.split("~")) > 4:
                    sublines = re.split('C: ((\+|-)?\d\.\d+(e-)?\d*)', line)
                    count=0
                    f.write(sublines[0] + sublines[1])
                    f.write("\n")
                    f.write(sublines[4] + sublines[5])
                else:
                    f.write("".join(line.split("C: ")))
                f.write("\n")
                line = g.readline() 
    return new_path

def load_explanations(path, args=None):
    print(f"Loading from {path}")
    df = pd.read_csv(path, sep="~", header=0, names=["review", "explanation", "contribution"])
    #df["contribution"] = df["contribution"].apply(lambda c: float(str(c).split(":")[0]))
    df["contribution"] = df["contribution"].apply(lambda c: float(c))
    df["frequency"] = df["explanation"].apply(lambda f: list(re.findall(r'(\d+)', str(f)))).apply(lambda x: x[0] if x else None)
    df["confidence_score"] = df["explanation"].apply(lambda f: re.findall(r'\d+\.\d+', str(f))).apply(lambda x: float(x[0]) if x else None)
    df["prediction"] = df["explanation"].apply(lambda f: re.findall(r'\d+\.\d+',str(f))).apply(lambda x: float(x[1]) if len(x)>1 else None)
    df["label"] = df["explanation"].apply(lambda x: re.findall(r'\d+\.\d+', str(x))).apply(lambda x: float(x[2]) if len(x)>2 else None)
    df["raw_pred"] = df["explanation"].apply(lambda x: re.findall(r'\d+\.\d+', str(x))).apply(lambda x: float(x[3]) if len(x)>3 else None)
    texts = prepare_text_for_classification(df["review"].values)
    lens = [x.shape[0] for x in texts]
    max_len = max(lens)
    len_texts = torch.LongTensor(lens)
    texts = torch.stack([F.pad(t, (0, max_len - t.shape[0])).unsqueeze(1) for t in texts])
    instances = len_texts.shape[0]
    i,batch=0,32
    #results = [model(texts[i].unsqueeze(1), torch.tensor([len_texts[i]])) for i in range(0, instances)]
    results=[]
    with torch.no_grad():
        for i in range(0, instances, batch):
            try:
                ipdb.set_trace(context=10)
                results.append(torch.sigmoid(model(texts[i], torch.tensor([len_texts[i]]))))
                torch.cuda.empty_cache()
            except:
                ipdb.set_trace(context=10)
                import sys, traceback
                traceback.print_exc(file=sys.stdout)
    #results = [model(texts[i:min(i+batch, instances)], len_texts[i:min(i+batch, instances)]) for i in range(0,instances, batch)]
    #import itertools
    #results = list(itertools.chain.from_iterable(results))
    print(len(results))
    ipdb.set_trace(context=10)
    df["vanilla_prediction"] = results
    return df
##################################################################


def plot_hist(contributions, title, path):
    import matplotlib.pyplot as plt
    plt.hist(contributions, label="contribution")
    # plt.hist(df[(df["prediction"]==df["label"])]["contribution"], label="contribution")
    plt.title(title)
    plt.legend()
    plt_path = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\\explanations\\" 
    plt.savefig(plt_path + "contributions_hist_100pn.png") 
'Contributions distribution of the first 100 correctly classified instances'

##################################################################

def print_metrics(df, full_path, field="contribution"):
    mean, min_c, max_c, std = df[field].mean(), df[field].min(), df[field].max(), df[field].std()
    with open(full_path, "w") as f:
        f.write(f"Mean: {mean}\n")
        f.write(f"Min: {min_c}\n")
        f.write(f"Max: {max_c}\n")
        f.write(f"StdDev: {std}\n")
        f.write(f"Positive %: {df[df[field]>0][field].count()*100/df[field].count()}\n\n")
        f.write(f"Positive contributions count {df[df[field]>0][field].count()}\n")
        f.write(f"All contributions count {df[field].count()}\n")

def print_percentages(df, full_path):
    df["c"] = df.apply(lambda x: -1*x["contribution"] if x["label"]==0 else x["contribution"], axis=1)
    dp = df[round(df["c"]+ df["raw_pred"])!=round(df["prediction"])].count()["contribution"]
    
    ccp = df[(round(df["raw_pred"])!=round(df["prediction"])) & (df["label"]==round(df["prediction"]))].count()["contribution"]
    icp = df[(round(df["raw_pred"])!=round(df["prediction"])) & (df["label"]!=round(df["prediction"]))].count()["contribution"]

    cp = df[round(df["raw_pred"])!=round(df["prediction"])].count()["contribution"]
    with open(full_path, "w") as f:
        f.write(f"Changed prediction: {cp}\n")
        f.write(f"Different predictions (should be equal to changed pred): {dp}\n")

        f.write(f"Correctly changed predictions: {ccp}\n")
        f.write(f"Correctly changed predictions (out of changed predictions): {ccp*100.0/cp}\n")

        f.write(f"Incorrectly changed predictions: {icp}\n")
        f.write(f"Incorrectly changed predictions (out of changed predictions): {icp*100.0/cp}\n")

       
        f.write(f"Round prediction != prediction: {dp}\nC+raw_pred != pred {df[round(df['c']+ df['raw_pred'])!=round(df['prediction'])].count()['contribution']}")

##################
# RUN COMMAND
#python qualitative_grep.py -p experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations -f e_test-7_2020-05-24_03-16-15.txt
##################
start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

parser = argparse.ArgumentParser(description='Config params.')

# experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations\e_test-7_2020-05-24_03-16-15.txt
#  experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations\e_test-7_2020-05-24_03-16-15_fix.txt
parser.add_argument('-p', metavar='path', type=str,
                    help='expl path')
parser.add_argument('-f', metavar='file name', type=str,
                    help='expl file')
parser.add_argument('--fix', help='fix the explanation file format', action='store_true')

args = parser.parse_args()
##################################################################

e_path = os.path.join(args.p, args.f)
if args.fix:
    print("Fixing...")
    path = fix_file(e_path)
    print(f"Fixed {path}")
else:
    path = e_path # 'experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\\explanations\\e_test-7_2020-05-24_03-16-15_fix.txt'

print("Loading explanations...")
df = load_explanations(path)
print("Loaded")
##################################################################

def sample_metrics(df, path, sample_count=10):
    for i in range(10):
        sample_path = os.path.join(path,f"metrics_sample-{i}.txt")
        print(sample_path)
        sample = pd.DataFrame({"contribution":df["contribution"].sample(100)})
        print_metrics(sample, sample_path)
        sample.to_pickle(os.path.join(path,f"samples-dump-{i}.h5")) 

        hist_path = os.path.join(path, f"hist_sample-{i}.png")
        hist_title = "Sample contribution distribution for correctly classified test instances"
        plot_hist(sample['contribution'], hist_title, hist_path)

##################################################################

#sample_metrics(df[df["prediction"]==df["label"]], args.p)

print("all instances...")
all_metrics_path = os.path.join(args.p, "all_instances_metrics.txt")
print_metrics(df, all_metrics_path)
print_percentages(df, os.path.join(args.p, "percentages-changed-preds.txt"))

all_hist_path = os.path.join(args.p, "all_instances_hist.png")
plot_hist(df["contribution"], "Histogram all contributions", all_hist_path)

print("all correct instances...")
all_metrics_path = os.path.join(args.p, "all_correct_metrics.txt")
print_metrics(df[round(df["prediction"])==df["label"]], all_metrics_path)
pos_hist_path = os.path.join(args.p, "all_correct_hist.png")
plot_hist(df["contribution"], "Histogram positives' contributions", pos_hist_path)

print("all incorrect instances...")
all_metrics_path = os.path.join(args.p, "all_incorrect_metrics.txt")
print_metrics(df[round(df["prediction"])!=df["label"]], all_metrics_path)
neg_hist_path = os.path.join(args.p, "all_incorrect_hist.png")
plot_hist(df["contribution"], "Histogram negatives' contributions", neg_hist_path)

path = os.path.join(args.p, "descending_contribution.txt")
df.sort_values(by=["contribution"], ascending=False).to_csv(path)

path = os.path.join(args.p, "descending_contribution_correct.txt")
df[df["label"]==round(df["prediction"])].sort_values(by=["contribution"], ascending=False).to_csv(path)

path = os.path.join(args.p, "descending_contribution_incorrect.txt")
df[df["label"]!=round(df["prediction"])].sort_values(by=["contribution"], ascending=False).to_csv(path)

# print(df.head())
# print(df[(df["prediction"]==df["label"])].shape)
# print(df[(df["prediction"]==df["label"])]["contribution"].mean())
# print(df[(df["prediction"]==df["label"])]["contribution"].min())
# print(df[(df["prediction"]==df["label"])]["contribution"].max())
# print(df[(df["prediction"]==df["label"])]["contribution"].std())
# print(df[(df["prediction"]==df["label"])& (df["contribution"]>0)]["contribution"].count())
# print(100*df[(df["prediction"]==df["label"])& (df["contribution"]>0)]["contribution"].count()/df[(df["prediction"]==df["label"])]["contribution"].count())

# # print(df[df["contribution"]>0]["contribution"])
# top_100_positives = df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]
# # top_50pos = df[(df["prediction"]==df["label"]) & (df["prediction"]==1.0)]["contribution"][:50]
# # top_50neg = df[(df["prediction"]==df["label"]) & (df["prediction"]==0.0)]["contribution"][:50]
# # top_100 = top_50pos.append(top_50neg)

# top_100 = df["contribution"].sample(100)

# print(top_100.count())
# print(top_100[top_100>0].count())
# print(df[(df["prediction"]==df["label"])][:100][df["contribution"]>0].count())
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100])
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].mean())
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].max())
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].min())
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].std())

# contributions_top = df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"]



# file_name = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\\explanations\\sample100.h5"
# top_100.to_pickle(file_name)  # where to save it, usually as a .pkl
# # Then you can load it back using:

# # df = pd.read_pickle(file_name)
