from experiment_framework import Experiment
from models import rnn
from utils import default_config as dc
from datasets import imdb_dataset as imdb
import torch


dc.CONFIG["input_size"] = (12,)
model = rnn.RNN("m-id-test", dc.MODEL_MAPPING, dc.CONFIG)

dataset = imdb.IMDBDataset(dc.DATASET_ARGS)
experiment = Experiment("e1").with_config(dc.CONFIG).with_data(dataset).with_model(model).run()
