import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import rationale_net.models.attention as lstm

import rationale_net.models.transformer as transformer

class Encoder(nn.Module):

    def __init__(self, embeddings, args, expl_vocab):
        super(Encoder, self).__init__()
        ### Encoder
        self.args = args
        # embedded possible explanations (word indices)
        #self.expl_vocab = expl_vocab
        vocab_size, hidden_dim = embeddings.shape
        self.embedding_dim = hidden_dim
        self.embedding_layer = nn.Embedding( vocab_size, hidden_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = True
        
        self.embedding_layer2 = nn.Embedding( vocab_size, hidden_dim)
        self.embedding_layer2.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer2.weight.requires_grad = True
        
        #self.expl_vocab = args.expl_vocab
        
        self.embedding_fc = nn.Linear( hidden_dim, hidden_dim )
        self.embedding_bn = nn.BatchNorm1d( hidden_dim)
        
        #self.expl_tensors = torch.tensor([expl_vocab[e_id]["emb"].data for e_id in sorted(expl_vocab.keys())])
        #self.embedded_vocab = self.embedding_layer2(self.expl_tensors)
        self.model_form = args.model_form
        
        if self.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time=(not args.use_as_tagger))
            self.fc = nn.Linear( len(args.filters)*args.filter_num,  args.hidden_dim)
        elif self.model_form == 'lstm':
            self.lstm = lstm.AttentionModel(args.batch_size, args.hidden_dim, 3* args.hidden_dim, vocab_size, hidden_dim, embeddings)
        elif self.model_form=='transformer':
            self.transformer = transformer.Transformer(hidden_dim, args.hidden_dim, vocab_size, 250, 10, 4, args.dropout, True)
            self.tr_layer = nn.Linear(self.embedding_dim, args.num_class)
        else:
            raise NotImplementedError("Model form {} not yet supported for encoder!".format(args.model_form))


        self.dropout = nn.Dropout(args.dropout)
        self.hidden = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, x_indx, mask=None):
        '''
            x_indx:  batch of word indices
            mask: Mask to apply over embeddings for tao rationales
        '''
        
        x = self.embedding_layer(x_indx.squeeze(1))
        if not mask is None:
            vocab_emb = self.embedding_layer2(self.args.expl_vocab)
            explanation =  torch.mul(vocab_emb, mask.unsqueeze(-1))
            x = torch.cat((x, explanation),1)
        hidden = None
        if self.model_form == 'cnn':        
            x = F.relu( self.embedding_fc(x))
            x = self.dropout(x)
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            hidden = self.cnn(x)
            hidden = F.relu( self.fc(hidden) )
            hidden = self.dropout(hidden)
            logit = self.hidden(hidden)
        elif self.model_form == 'lstm':
            logit = self.lstm(x.squeeze(1))
            hidden = None
        elif self.model_form == 'transformer':
            logit = self.tr_layer(x)
        else:
            raise Exception("Model form {} not yet supported for encoder!".format(self.model_form))

        return logit, hidden