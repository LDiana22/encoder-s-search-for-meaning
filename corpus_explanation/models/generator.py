from models import abstract_model as am
from models import vanilla as van
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary

class MLPGen(am.AbstractModel):
    """
    MLP generator - dictionary for all classes (mixed)
    """
    def __init__(self, id, mapping_file_location, model_args, TEXT, explanations):
        """
        id: Model id
        mapping_file_location: directory to store the file "model_id" 
                               that containes the hyperparameters values and 
                               the model summary
        logs_location: directory for the logs location of the model
        model_args: hyperparameters of the model
        explanations: Dictionary of explanations [{phrase: {class:freq}}]
        """
        super().__init__(id, mapping_file_location, model_args)

        self.vanilla = van.LSTM("gen-van-lstm", mapping_file_location, model_args, TEXT)

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        self.input_size = len(TEXT.vocab)
        self.embedding = nn.Embedding(self.input_size, model_args["emb_dim"], padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.data[UNK_IDX] = torch.zeros(model_args["emb_dim"])
        self.embedding.weight.data[PAD_IDX] = torch.zeros(model_args["emb_dim"])

        self.emb_dim = model_args["emb_dim"]
        self.gen = nn.LSTM(model_args["emb_dim"], 
                           model_args["hidden_dim"], 
                           num_layers=model_args["n_layers"], 
                           bidirectional=True,
                           dropout=model_args["dropout"])

        self.fc = nn.Linear(model_args["hidden_dim"] * 2, model_args["output_dim"])

        self.lin = nn.Linear(model_args["emb_dim"], model_args["hidden_dim"])

        self.dictionaries = explanations.get_dict()

        self.gen_lin, self.gen_softmax, self.explanations = [], [], []
        for class_label in self.dictionaries.keys():
            dictionary = self.dictionaries[class_label]
            stoi_expl = self.__pad([
                torch.tensor([TEXT.vocab.stoi[word] for word in phrase.split()]).to(self.device)
                for phrase in dictionary.keys()], explanations.max_words)
            
            self.gen_lin.append(nn.Linear(model_args["hidden_dim"], len(stoi_expl)))
            self.gen_softmax.append(nn.Softmax(2))
            self.explanations.append(stoi_expl)

        self.dropout = nn.Dropout(model_args["dropout"])

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

        self = self.to(self.device)
        super().save_model_type(self)

    def __pad(self, tensor_list, length):
        """
        0 pad to the right for a list of variable sized tensors
        e.g. [torch.tensor([1,2]), torch.tensor([1,2,3,4]),torch.tensor([1,2,3,4,5])], 5 ->
                [tensor([1, 2, 0, 0, 0]), tensor([1, 2, 3, 4, 0]), tensor([1, 2, 3, 4, 5])]
        """
        return torch.stack([torch.cat([tensor, tensor.new(5-tensor.size(0)).zero_()])
            for tensor in tensor_list]).to(self.device)

    def forward(self, text, text_lengths):
        
        batch_size = text.size()[1]
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        ##GEN
        # # [sent len, batch, 2*hidden_dim]
        # expl_activ, (_, _) = self.gen(embedded)
        # expl_activ = nn.Dropout(0.4)(expl_activ)
        # [sent, batch, hidden]
        expl_activ = self.lin(embedded)
        # expl_activ = nn.Dropout(0.4)(expl_activ)




        context_vector, final_dict, expl_distributions = [], [], []



        for i in range(len(self.dictionaries.keys())):
            # explanations[i] -> [dict_size, max_words, emb_dim]

            # [dict_size, max_words, emb_dim]
            v_emb = self.embedding(self.explanations[i])

            #[batch,dict_size, max_words, emd_dim]
            vocab_emb = v_emb.repeat(batch_size,1,1,1)
            #[batch,dict_size, max_words* emd_dim]
            vocab_emb = vocab_emb.reshape(vocab_emb.size(0),vocab_emb.size(1),-1)

            # [sent, batch, dict_size]
            lin_activ = self.gen_lin[i](expl_activ)
            # expl_activ = nn.Dropout(0.2)(lin_activ)
            # [sent, batch, dict_size]
            expl_dist = self.gen_softmax[i](lin_activ)
            
            # [batch, sent, dict_size]
            expl_distribution = torch.transpose(expl_dist, 0, 1)

            aggregation = nn.Conv1d(in_channels=expl_distribution.size(1), out_channels=1, kernel_size=1)
            expl_distribution = aggregation(expl_distribution)

            expl_distributions.append(expl_distribution)

            # [batch,sent, max_words*emb_dim]
            expl = torch.bmm(expl_distributions[i], vocab_emb)

            #[batch,max_words,emb_dim]
            context_vector.append(torch.max(expl, dim=1).values.reshape(batch_size, v_emb.size(1),-1))


            sep = torch.rand((batch_size,1,self.emb_dim), device=self.device)
            # [batch, 1+1, emb_dim]
            final_dict.append(torch.cat((sep, context_vector[i]), 1))


        final_expl = final_dict[0]
        for i in range(1, len(final_dict)):
            final_expl = torch.cat((final_expl, final_dict[i]), 1)

        #[batch, sent, emb]
        x = torch.transpose(embedded,0,1)

        # [batch, sent_len+2, emb_dim]
        concat_input = torch.cat((x,final_expl),1) 

        #[sent_len+1, batch, emb_dim]
        final_input = torch.transpose(concat_input,0,1)
        
        output = self.vanilla.raw_forward(final_input, text_lengths)
        

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
        self.expl_distributions = expl_distributions  
        return output