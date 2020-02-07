from models import abstract_model as am
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

        self.to(self.device)
        super().save_model_type(self)

    def forward(self, text, text_lengths):
        #text = [sent len, batch size]
        text.to(self.device)
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
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
            
        return self.lin(hidden)


    def train_model(self, iterator):
        """
        metrics.keys(): [train_acc, train_loss, train_prec,
                        train_rec, train_f1, train_macrof1,
                        train_microf1, train_weightedf1]
        """
        e_loss = 0
        e_acc, e_prec, e_rec = 0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0

        self.train()

        for batch in iterator:
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = self.forward(text, text_lengths).squeeze(1)
            loss = self.criterion(predictions, batch.label)

            predictions = torch.round(torch.sigmoid(predictions)).detach().numpy()
            #metrics
            acc = accuracy_score(batch.label, predictions)
            prec = precision_score(batch.label, predictions, zero_division=0)
            rec = recall_score(batch.label, predictions, zero_division=0)
            f1 = f1_score(batch.label, predictions, zero_division=0)
            macrof1 = f1_score(batch.label, predictions, average='macro', zero_division=0)
            microf1 = f1_score(batch.label, predictions, average='micro', zero_division=0)
            wf1 = f1_score(batch.label, predictions, average='weighted', zero_division=0)

            loss.backward()

            e_loss += loss.item()
            e_acc += acc
            e_prec += prec
            e_rec += rec
            e_f1 += f1
            e_macrof1 += macrof1
            e_microf1 += microf1
            e_wf1 += wf1
        
        metrics ={}
        size = len(iterator)
        metrics["train_loss"] = e_loss/size
        metrics["train_acc"] = e_acc/size
        metrics["train_prec"] = e_prec/size
        metrics["train_rec"] = e_rec/size
        metrics["train_f1"] = e_f1/size
        metrics["train_macrof1"] = e_macrof1/size
        metrics["train_microf1"] = e_microf1/size
        metrics["train_weightedf1"] = e_wf1/size
        return metrics

    def evaluate(self, iterator, prefix="test"):
        """
            Return a metrics dict with the keys prefixed by prefix
            metrics = {}
        """
        self.eval()

        e_loss = 0
        e_acc, e_prec, e_rec = 0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0
        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text
                predictions = self.forward(text, text_lengths).squeeze(1)
                loss = self.criterion(predictions, batch.label)
    
                predictions = torch.round(torch.sigmoid(predictions))

                acc = accuracy_score(batch.label, predictions)
                prec = precision_score(batch.label, predictions)
                rec = recall_score(batch.label, predictions)
                f1 = f1_score(batch.label, predictions)
                macrof1 = f1_score(batch.label, predictions, average='macro')
                microf1 = f1_score(batch.label, predictions, average='micro')
                wf1 = f1_score(batch.label, predictions, average='weighted')

                e_loss += loss.item()
                e_acc += acc
                e_prec += prec
                e_rec += rec
                e_f1 += f1
                e_macrof1 += macrof1
                e_microf1 += microf1
                e_wf1 += wf1

        metrics ={}
        size = len(iterator)
        metrics[f"{prefix}_loss"] = e_loss/size
        metrics[f"{prefix}_acc"] = e_acc/size
        metrics[f"{prefix}_prec"] = e_prec/size
        metrics[f"{prefix}_rec"] = e_rec/size
        metrics[f"{prefix}_f1"] = e_f1/size
        metrics[f"{prefix}_macrof1"] = e_macrof1/size
        metrics[f"{prefix}_microf1"] = e_microf1/size
        metrics[f"{prefix}_weightedf1"] = e_wf1/size
        return metrics