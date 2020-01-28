from abc import ABC
from modelsummary import  summary
import os.path
from datetime import datetime
import torch
from contextlib import redirect_stdout
from torch import nn


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

    def override(self, args):
        self.args.update(args)

    def __create_directories(self):
        """
        All the directories for a model are placed under the directory 
            prefix_dir / model_id / {dirs}
        """ 
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

            with redirect_stdout(map_file):
                summary(model, torch.zeros(self.args["input_size"]),show_input=True)
                print(self.delim, file=map_file)
                summary(model, torch.zeros(self.args["input_size"]),show_input=False)

    def checkpoint(self, metrics ={}):
        checkpoint_file = os.path.join(self.model_dir, self.args["dirs"]["checkpoint"], 
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.dict_checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
            }
        self.dict_checkpoint.update(metrics) 
        self.metrics = metrics
        torch.save(self.dict_checkpoint, checkpoint_file)

    def load_checkpoint(self, newest_file=None):
        checkpoint_dir = os.path.join(self.model_dir, self.args["dirs"]["checkpoint"])           
        if not newest_file:
            newest_file = max([os.path.join(path, basename) for basename in os.listdir(checkpoint_dir)],
                                key=os.path.getctime)

        path = os.path.join(checkpoint_dir, newest_file)
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        for key in self.metrics.keys():
            self[key] = checkpoint[key]