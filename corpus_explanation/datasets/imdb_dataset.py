from .preprocessing import remove_br_html_tags 

import glob
import os
import io
import random
import re
import spacy
from torchtext import datasets
from torchtext import data as data
from torchtext.vocab import GloVe
import torch

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class IMDBDataset:

  def __init__(self, args, max_length=250):
    self.args = args
    TEXT = data.Field(lower=True, 
                      include_lengths=True,
                      tokenize='spacy',
                      preprocessing=remove_br_html_tags)
    LABEL = data.LabelField(dtype = torch.float)
    print("Loading the IMDB dataset...")
    self.train_data = self._load_data(TEXT, LABEL, ".data/imdb/aclImdb", "train")
    self.test_data = self._load_data(TEXT, LABEL, ".data/imdb/aclImdb", "test")
    # train_data, self.test_data = self._load_data(TEXT, LABEL, ".data/imdb/aclImdb")
    # train_data, self.test_data = datasets.IMDB.splits(TEXT, LABEL, path=".data/imdb/aclImdb")
    self.train_data, self.valid_data = self.train_data.split(random_state=random.seed(SEED))
    print("IMDB...")
    print(f"Train {len(self.train_data)}")
    print(f"Valid {len(self.valid_data)}")
    print(f"Test {len(self.test_data)}")
    TEXT.build_vocab(self.train_data, 
                 max_size = args["max_vocab_size"],
                 vectors = GloVe(name='6B', dim=args["emb_dim"]), 
                 unk_init = torch.Tensor.normal_)


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
    return self.training_data

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
