from experiment_framework import Experiment
from models import rnn
from utils import default_config as dc
from datasets import imdb_dataset as imdb
from text_mining import rake_dict as rake
import torch


dc.CONFIG["input_size"] = (12,)
model = rnn.RNN("m-id-test", dc.MODEL_MAPPING, dc.CONFIG)

print("Dataset")
dataset = None#imdb.IMDBDataset(dc.CONFIG)
dictionary = rake.RakePerClassDictionary("rake-per-class", dataset, dc.CONFIG)

experiment = Experiment("e1").with_config(dc.CONFIG).with_data(dataset).with_dictionary(dictionary).with_model(model).run()

