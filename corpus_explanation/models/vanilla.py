from models import abstract_model as am
import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary

class LSTM(am.AbstractModel):
    """
    Baseline - no generator model
    """
    def __init__(self, id, mapping_file_location, model_args, TEXT):
        """
        id: Model id
        mapping_file_location: directory to store the file "model_id" 
                               that containes the hyperparameters values and 
                               the model summary
        logs_location: directory for the logs location of the model
        model_args: hyperparameters of the model
        """
        super().__init__(id, mapping_file_location, model_args)
        self.device = torch.device('cuda' if model_args["cuda"] else 'cpu')
        
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        self.input_size = len(TEXT.vocab)
        self.embedding = nn.Embedding(self.input_size, model_args["emb_dim"], padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.data[UNK_IDX] = torch.zeros(model_args["emb_dim"])
        self.embedding.weight.data[PAD_IDX] = torch.zeros(model_args["emb_dim"])

        self.lstm = nn.LSTM(model_args["emb_dim"], 
                           model_args["hidden_dim"], 
                           num_layers=model_args["n_layers"], 
                           bidirectional=True, 
                           dropout=model_args["dropout"])
        self.lin = nn.Linear(2*model_args["hidden_dim"], model_args["output_dim"]).to(self.device)
        self.dropout = nn.Dropout(model_args["dropout"])
        
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

        self = self.to(self.device)
        super().save_model_type(self)

    def forward(self, text, text_lengths):
        text = text.to(self.device)
        #text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        return self.raw_forward(embedded, text_lengths)

    def raw_forward(self, embedded, text_lengths):

        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return torch.sigmoid(self.lin(hidden)).squeeze(1).to(self.device)
