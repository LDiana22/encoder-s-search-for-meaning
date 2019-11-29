import torch
import torch.nn as nn
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import rationale_net.utils.helpers as helpers
import rationale_net.models.transformer as transformer

import rationale_net.models.attention as lstm
#from semantic_text_similarity.models import WebBertSimilarity
"""
class Similarity(object):
    class __BertSimilarity:
        def __init__(self):
            self.compute = WebBertSimilarity(device='cuda') 
    instance = None
    def __new__(cls):
        if not Similarity.instance:
            Similarity.instance = Similarity.__BertSimilarity()
        return Similarity.instance

bert_similarity = Similarity()
"""
#bert_similarity = WebBertSimilarity()

'''
    The generator selects a rationale z from a document x that should be sufficient
    for the encoder to make it's prediction.

    Several froms of Generator are supported. Namely CNN with arbitary number of layers, and @taolei's FastKNN
'''
class Generator(nn.Module):

    def __init__(self, embeddings, args, expl_vocab):
        super(Generator, self).__init__()
        vocab_size, hidden_dim = embeddings.shape
        self.embedding_dim = hidden_dim
        self.embedding_layer = nn.Embedding( vocab_size, hidden_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False
        self.args = args
        #self.args = {"model_form": args.model_form, "expl_vocab": args.expl_vocab, "cuda": args.cuda}
        self.z_dim = len(expl_vocab.keys())
       # self.model_form = self.args.model_form
        if self.args.model_form =='transformer':
            self.transformer = transformer.Transformer(self.embedding_dim, self.args.hidden_dim, vocab_size, self.z_dim, 10, 4, self.args.dropout, True)
            self.tr_layer = nn.Linear(self.embedding_dim, self.z_dim)
        elif self.args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time = False)
        elif self.args.model_form == 'lstm':
            self.lstm = lstm.AttentionModel(args.batch_size, self.z_dim, args.hidden_dim, vocab_size, self.embedding_dim, embeddings)        
        self.layer = nn.Linear((len(args.filters)* args.filter_num), self.args.hidden_dim)
        self.hidden = nn.Linear(self.args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        if self.args.cuda:
            self.device='cuda'
        else:
            self.device='cpu'


    def __range01(self, x):
        return torch.div(x-torch.min(x), torch.max(x)-torch.min(x))
    
    def gumbel_softmax(self, logits, temp, cuda):
        probs = helpers.gumbel_softmax(logits, temp, cuda)
        probs = torch.sum(probs, 1)
        return probs
    
    def  __z_forward(self, activ):
        '''
            Returns prob of each token being selected out of the vocabulary tokens
        '''
        if self.args.model_form=='cnn':
            activ = activ.transpose(1,2)
            layer_out = self.layer(activ)
            logits = self.hidden(layer_out) # batch, length, z_dim
            probs = self.gumbel_softmax(logits, self.args.gumbel_temprature, self.args.cuda)
        elif self.args.model_form == 'lstm':
            probs = helpers.gumbel_softmax(logits, self.args.gumbel_temprature, self.args.cuda)
        elif self.args.model_form == 'transformer':
            logits = self.tr_layer(activ)
            probs = helpers.gumbel_softmax(logits, self.args.gumbel_temprature, self.args.cuda)
        mask = self.__range01(probs)
       # mask = torch.div(probs,torch.norm(probs,2))
        return mask #batch, length


    def forward(self, x_indx):
        '''
            Given input x_indx of dim (batch, length), return z (batch, length) such that z
            can act as element-wise mask on the explanation dict
        '''
        if self.args.model_form == 'cnn':
            x = self.embedding_layer(x_indx.squeeze(1))
            if self.args.cuda:
                x = x.cuda()
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            activ = self.cnn(x)        
            z = self.__z_forward(F.relu(activ))
        elif self.args.model_form == 'rcnn':
            pass
        elif self.args.model_form == 'lstm':
            z = self.lstm(x_indx.squeeze(1))
        elif self.args.model_form == 'transformer':
            #x_indx [batch,1,250]
            #[batch, max_sent_size_250]
            x = self.embedding_layer(x_indx.squeeze(1))
            # x [batch, max_sent_size_250, emb_dim]
            activ = self.transformer(x)
            activ = torch.sum(activ, 1)
            z=self.__z_forward(activ)
        else:
            raise NotImplementedError("Model form {} not yet supported for generator!".format(self.args.model_form))

        mask = self.sample(z)
        return mask, z


    def sample(self, z):
        '''
            Get mask from probablites at each token. Use gumbel
            softmax at train time, hard mask at test time
        '''
        mask = z
#        if self.training:
#            mask = z
#        else:
#            ## pointwise set <.5 to 0 >=.5 to 1
#            torch.set_printoptions(profile="full")
#            print("%%")
#            print(mask)
#            mask = helpers.get_hard_mask(z)
#            print(mask)
#            print("%%")
        return mask

    def masks_to_vocab_idx(self, masks):
        """
        masks (batch, vocab_size)
        return (batch, [words]) -> batch, expl_dim
        """
        return [[idx for idx in range(len(instance)) if instance[idx]!=0] for instance in masks]

    def loss(self, mask, x_indx):
        '''
            x_indx - batch, ipt_size
            mask - batch, vocab_size
            text (batch-sized list of text)
            Compute the generator specific costs, i.e selection cost, continuity cost, and global vocab cost
        '''
        batch_size=x_indx.size()[0]
        x_indx = x_indx.view(batch_size,-1)
        #expl_idx = [[idx for idx in range(mask[i].size()[0]) if mask[i][idx]>=0.5] for i in range(batch_size)]
        
        #expl1_idx = mask.argmax(-1)[0]
        
        selection_cost = torch.mean(torch.sum(mask, dim=1))
        
        ##expl = self.args.expl_vocab[expl_idx]
        
        ##expl_emb = self.embedding_layer(expl)
        
        #vocab_Size, emb
        vocab_emb = self.embedding_layer(self.args.expl_vocab) # vocab_dim, emb_dim
        vocab_emb = vocab_emb.view(1, vocab_emb.size()[0],-1).expand((batch_size, vocab_emb.size()[0],-1))
        ##expl_emb = [vocab_emb[e_idx] for e_idx in expl_idx]
        
        #batch,1,vocab_size
        mask_1 = mask.view(mask.size()[0],1,-1)
        ###batch, vocab_Size, emb_size
        explanation =  torch.bmm(mask_1, vocab_emb) 
        #batch, 1, emb_size
        
        #explanation =  torch.mul(expl_emb, mask.unsqueeze(-1)) 
        
        #aggregated_explanation = torch.sum(explanation, dim=-2).view(batch_size, 1,-1)
        
        #explanation_vals = torch.masked_select(explanation, explanation.ge(0.5))
        
        #explanations = explanation.expand(batch_size, explanation.size()[1], explanation.size()[2])
        
        #expl = explanation[:expl1_idx:]
        
        ##bacth, 250, emb
        x_emb = self.embedding_layer(x_indx)  
        #w_idx = self.masks_to_vocab_idx(masks)
        #expl_text = [[self.args.expl_vocab[i]['text'] for i in range(len(expl))] for expl in w_idx]
        # (batch, expl_size)
        #with open("gen/expl.txt", "w") as f:
        #    f.write(f"{text}\n&&\n{expl_text}\n{str(sim)}\n==========")
        cos = nn.CosineSimilarity(dim=2)
        semantic_cost = -cos(x_emb, explanation)
        
        #semantic_cost = torch.zeros([batch_size, 1], dtype=torch.float32, device=self.device)
        #for i in range(batch_size): # batch
        #    similarities_cost = [1-cos(word,expl_emb[i][0]) for word in x_emb[i][0]]
        #    semantic_cost[i] = sum(similarities_cost)
       
        #l_padded_mask =  torch.cat( [mask[:,0].unsqueeze(1), mask] , dim=1)
        #r_padded_mask =  torch.cat( [mask, mask[:,-1].unsqueeze(1)] , dim=1)
        #continuity_cost = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=1) )
        return selection_cost, torch.mean(semantic_cost)

