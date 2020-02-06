from experiment_framework import Experiment
from models import vanilla
from utils import default_config as dc
from datasets import imdb_dataset as imdb
from text_mining import rake_dict as rake
import torch

from datetime import datetime

start = datetime.now()

formated_date = start.strftime("%Y-%m-%d_%H-%M-%S")

experiment = Experiment(f"e-v-{formated_date}").with_config(dc.CONFIG).override({
	"hidden_dim": 256,
	"n_layers": 2,
	"max_dict": 300, 
	"cuda": True,
	# "epochs":1
	})

dataset = imdb.IMDBDataset(dc.CONFIG)
dictionary = rake.RakePerClassDictionary("rake-per-class", dataset, dc.CONFIG)

model = vanilla.LSTM("m-vanilla-bi-lstm", dc.MODEL_MAPPING, experiment.config, dataset.TEXT)
experiment.with_data(dataset).with_dictionary(dictionary).with_model(model).run()

print(f"Time: {str(datetime.now()-start)}")