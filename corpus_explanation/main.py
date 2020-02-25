import argparse
from datetime import datetime

from datasets import imdb_dataset as imdb
from experiment_framework import Experiment
from models import generator
from models import vanilla
from utils import constants as ct 
from utils import default_config as dc
from text_mining import rake_dict as rake


start = datetime.now()

formated_date = start.strftime(ct.DATE_FORMAT)

parser = argparse.ArgumentParser(description='Config params.')
parser.add_argument('-e', metavar='epochs', type=int, default=dc.CONFIG["epochs"],
                    help='Number of epochs')

parser.add_argument('--td', type=bool, default=dc.CONFIG["toy_data"],
                    help='Toy data (load just a small data subset)')

parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no_train', dest='train', action='store_false')
parser.set_defaults(train=dc.CONFIG["train"])

parser.add_argument('--restore', dest='restore', action='store_true')
parser.set_defaults(restore=dc.CONFIG["restore_checkpoint"])

parser.add_argument('--cuda', type=bool, default=dc.CONFIG["cuda"])
args = parser.parse_args()

experiment = Experiment(f"e-v-{formated_date}").with_config(dc.CONFIG).override({
	"hidden_dim": 256,
	"n_layers": 2,
	"max_dict": 300, 
	"cuda": args.cuda,
  "restore_checkpoint" : bool(args.restore),
  "train": args.train,
  "toy_data": args.td,	
	"epochs": args.e,
	})

dataset = imdb.IMDBDataset(experiment.config)
explanations = rake.RakePerClassExplanations("rake-per-class-300", dataset, experiment.config)

model = generator.MLPGen("mlp-gen_vanilla-bi-lstm_mixed-expl", dc.MODEL_MAPPING, experiment.config, dataset, explanations)
# model = vanilla.LSTM("v-lstm", dc.MODEL_MAPPING, experiment.config, dataset.TEXT)
experiment.with_data(dataset).with_dictionary(explanations).with_model(model).run()

print(f"Time: {str(datetime.now()-start)}")