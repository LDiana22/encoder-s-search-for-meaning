from abc import ABC
from modelsummary import  summary
import os.path
from datetime import datetime
import torch
from contextlib import redirect_stdout
from torch import nn
from datetime import datetime

from utils import file_helpers as fh


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
        if self.args["cuda"]:
            self.device='cuda'
        else:
            self.device='cpu'
        self.model_dir = model_dir = os.path.join(self.args["prefix_dir"], self.id)
        self.__create_directories()
        # define self.metrics = {}

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
        self.metric_keys = list(metrics.keys())
        self.dict_checkpoint.update(metrics)
        torch.save(self.dict_checkpoint, checkpoint_file)

    def load_checkpoint(self, newest_file_name):
        checkpoint_dir = os.path.join(self.model_dir, self.args["dirs"]["checkpoint"])           

        path = os.path.join(checkpoint_dir, newest_file_name)
        print(f"Loading checkpoint: {path}") 
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.metrics = {}
        for key in checkpoint.keys():
            if key not in ['epoch', 'model_state_dict', 'optimizer_state_dict']:
                self.metrics[key] = checkpoint[key]

    def save_results(self, metrics):
        metrics_path = os.path.join(self.model_dir, self.args["dirs"]["metrics"])
        results_file = os.path.join(metrics_path, f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        with open(results_file, "w") as f:
            f.write(str(metrics))

    def train_model(self, iterator):
        """
        Abstract method, avoiding multiple inheritance
        Return a metrics dict with the keys prefixed by 'train'
        e.g. metrics={"train_acc": 90.0, "train_loss": 0.002}
        """
        pass

    def evaluate(self, iterator, prefix="test"):
        """
        Abstract method, avoiding multiple inheritance

        Return a metrics dict with the keys prefixed by prefix
        e.g. metrics={f"{prefix}_acc": 90.0, f"{prefix}_loss": 0.002}
        """
        pass

