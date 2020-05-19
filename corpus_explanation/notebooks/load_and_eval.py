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


  def load_dict(self, file):
    return pickle.load(open(file, "rb"))

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

    file = os.path.join(self.path, "dictionary-{formated_date}.txt")
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


class RakeCorpusPolarityFiltered(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args.get("max_dict", None)
    self.max_words = args["max_words_dict"]
    # self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    if args["load_dict"]:
        print(f"Loading RakeCorpusPolarityFiltered from: {args['load_dict']}")
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

    def train_model(self, iterator, args=None):
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
            for param in self.vanilla.parameters():
                param.requires_grad=False
            self.vanilla.eval()
            print("Vanilla frozen")
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

        self.optimizer = optim.AdamW(list(set(self.parameters()) - set(self.vanilla.parameters())))
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

        output, hidden = self.vanilla.raw_forward(embedded, text_lengths)
        self.raw_predictions = torch.sigmoid(output).squeeze()
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
            expl = "text, explanation (freq:confidence), prediction, true label\n"
            e_list = []
            # distr = [torch.tensor([]).to(self.device) for i in range(len(self.dictionaries.keys()))]
            distr = torch.tensor([]).to(self.device)
        self.eval()
        e_loss = 0
        e_acc, e_prec, e_rec = 0,0,0
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
        e_acc, e_prec, e_rec = 0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0
        e_contributions, e_len = 0, 0 

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
        if not alpha:
            alpha = self.alpha
        # if epoch == 10:
        #     alpha = 0.25
        # elif epoch == 15:
        #     alpha = 0

        output = torch.sigmoid(output)
        min_contributions = 1 - torch.sign(target - 0.5)*(output-self.raw_predictions)
        return alpha*bce(output, target) + (1-alpha)*(sum(min_contributions)/100)


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

    "restore_checkpoint" : False,
    "checkpoint_file": None,
    "train": True,

    "dropout": 0.05,
    "weight_decay": 5e-06,

    "alpha": 0.5,

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
    "load_dictionary": False,

    "hidden_dim": 256,
    "n_layers": 2,
    "max_dict": 1000, 
    "cuda": True,
    "restore_v_checkpoint" : True,
    "checkpoint_v_file": "experiments/gumbel-seed-true/v-lstm/snapshot/2020-04-10_15-04-57_e2",
    "train": True,
    "max_words_dict": 5,
    "patience":20,
    # "epochs": args.e,
    'alpha': 0.5,
    # "n1": args.n1,
    # "n3": args.n3,
    # "alpha_decay": args.decay,
    # "dropout": args.dr, 
    "load_dict":True,
    "dict_checkpoint": "experiments/independent/dictionaries/rake-polarity/dictionary.h5"
}
print(CONFIG)

# %% [code]
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
DATE_REGEXP = '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}'

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

checkpoint = "experiments/independent/bilstm_mlp_improve-dnn15-1-30-decay0.0-L2-dr0.3-eval1-textrank-improve100loss-alpha0.5-c-tr10/snapshot/2020-05-17_18-53-24_e18"

start = datetime.now()
dataset = IMDBDataset(experiment.config)
_,_, test_iterator = self.data.iterators()
print(f"Time data load: {str(datetime.now()-start)}")

explanations = RakeCorpusPolarityFiltered(f"rake-polarity", dataset, experiment.config)


model = model = MLPAfterIndependentOneDictImprove(f"{args.m}-dnn{args.n1}-{args.n2}-{args.n3}-decay{args.decay}-L2-dr{args.dr}-eval1-{args.d}-improve100loss-alpha{args.a}-c-tr10", MODEL_MAPPING, CONFIG, dataset, explanations)
print(model_args["checkpoint_v_file"])
model.load_checkpoint(checkpoint)
for param in model.parameters():
    param.requires_grad=False

print("Evaluating...")
metrics = model.evaluate(test_iterator, "test_f")
print("Test metrics:")
print(metrics)
model.save_results(metrics, "test")