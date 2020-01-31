from .preprocessing import remove_br_html_tags 

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
    super().__init__()
    self.args = args
    TEXT = data.Field(lower=True, 
                      include_lengths=True,
                      tokenize='spacy',
                      preprocessing=remove_br_html_tags)
    LABEL = data.LabelField(dtype = torch.float)
    print("Loading the IMDB dataset...")
    train_data, self.test_data = datasets.IMDB.splits(TEXT, LABEL, path=".data/imdb/aclImdb")
    self.train_data, self.valid_data = train_data.split(random_state=random.seed(SEED))
    print("IMDB...")
    print(f"Train {len(self.train_data)}")
    print(f"Valid {len(self.valid_data)}")
    print(f"Test {len(self.test_data)}")
    return
    TEXT.build_vocab(self.train_data, 
                 max_size = args["max_vocab_size"],
                 vectors = GloVe(name='6B', dim=args["emb_dim"]), 
                 unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(self.train_data)

    self.device = "cuda" if args["cuda"] else "cpu"

  def _load_data(self):
    for label in ['pos', 'neg']:
        for fname in glob.iglob(os.path.join(".data", label, '*.txt')):
            with io.open(fname, 'r', encoding="utf-8") as f:
                text = f.readline()
            examples.append(data.Example.fromlist([text, label], fields))

  def iterators(self):
    return data.BucketIterator.splits(
      (self.train_data, self.valid_data, self.test_data), 
      batch_size = self.args["batch_size"],
      sort_within_batch = True,
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