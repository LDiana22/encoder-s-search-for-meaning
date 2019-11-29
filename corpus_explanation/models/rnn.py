from models import abstract_model as am

from torch import nn

class RNN(am.AbstractModel, nn.Module):
    """
    RNN models
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
        super().__init__(id, mapping_file_location, model_args)

    def forward(self, x):
       pass
