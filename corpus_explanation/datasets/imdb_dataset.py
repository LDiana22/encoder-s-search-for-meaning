import random
from torchtext import datasets
from torchtext import data as data
from torchtext.vocab import GloVe
import torch
import spacy

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class IMDBDataset:

  def __init__(self, args, max_length=250):
    super().__init__()
    # import ipdb
    # ipdb.set_trace(context=10 )
    TEXT = data.Field(lower=True, include_lengths=True, tokenize='spacy')
    LABEL = data.LabelField(dtype = torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    print(len(train_data.examples))
    print(len(valid_data.examples))
    print(len(test_data.examples))
    # TEXT = data.Field(lower=True, batch_first=True)
    # LABEL = data.Field(sequential=False)
    
    # self.train_data, self.test_data = datasets.IMDB.splits(TEXT, LABEL)
    
    # TEXT.build_vocab(self.train_data, vectors=GloVe(name='6B', dim=args["emb_dim"]), max_size=args["max_vocab_size"], min_freq=10)
    # LABEL.build_vocab(self.train_data,)
    
    
    # train_iter, test_iter = data.BucketIterator.splits((self.train_data, self.test_data), batch_size= args["batch_size"])
    # train_iter.repeat = False
    # test_iter.repeat = False
    # print(len(self.train_data.examples))
    # print(len(self.test_data.examples))




  def training(self):
    return self.training_data

  def dev(self):
    return self.valid_data

  def test(self):
    return self.test_data 

  def override(self, args):
    self.args.update(args)
    return self