# -*- coding: utf-8 -*-
from experiment_framework import Experiment
from models import rnn
from utils import default_config as dc
from datasets import fb_dataset as fbd

embeddings = None

dataset = fbd.FullBeerDataset(embeddings, dc.DATASET_ARGS)
model1 = rnn.RNN("m-id-1", dc.MODEL_MAPPING, dc.MODEL_ARGS)
experiment1 = Experiment("e1").with_model(model1).with_config(dc.CONFIG).with_data(dataset).run()
