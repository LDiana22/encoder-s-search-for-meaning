# -*- coding: utf-8 -*-
from utils import file_helpers as fh
import torch

class Experiment(object):
    """Holds all the experiment parameters and provides helper functions."""
    def __init__(self, e_id):
        self.id = e_id
        
    def restore_model(self):
        if self.restore_checkpoint:
            checkpoint = self.model.args.dirs.checkpoint
            if checkpoint is not None:
                last_checkpoint = fh.get_last_checkpoint(checkpoint)
                if last_checkpoint is not None:
                    print(f"Loading latest checkpoint: {last_checkpoint}")
                    self.model = torch.load(last_checkpoint)    
                else:
                    print(f"No checkpoint found at {checkpoint}")
    def setup(self):
        self.restore_model()
        return self
                

    ### DECLARATIVE API ###

    def with_data(self, data):
        self.data = data
        return self

    def with_config(self, config):
        self.config = config.copy()
        return self

    def override(self, config):
        self.config.update(config)
        return self

    def with_model(self, model):
        self.model = model
        return self
    #### END API ######
    
    @property
    def experiment_name(self):
        return f'E-{self.id}_M-{self.model.id}'

    """ Dirs
    - *_dir - full path to dir
    """
    @property
    def experiments_dir(self):
        return "experiments"

    def run(self):
        pass
        
        