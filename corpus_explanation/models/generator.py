from models import abstract_model as am
from torchsummary import summary
from torch import nn


class LSTMGenPlusVanillaLSTM(am.AbstractModel):
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
        if self.args["cuda"]:
            self.device='cuda'
        else:
            self.device='cpu'
        self.lin = nn.Linear(model_args["input_size"][0], 1).to(self.device)
        super().save_model_type(self)

        
            
    def forward(self, x):
        x = x.to(self.device)
        return self.lin(x).to(self.device)


