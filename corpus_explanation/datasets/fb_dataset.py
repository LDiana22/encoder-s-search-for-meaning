import gzip

import torch.utils.data as data
import tqdm


from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.corpus_analysis.vocabulary import TfidfVocabulary

SMALL_TRAIN_SIZE = 800

class FullBeerDataset(data.dataset):

    def __init__(self, word_to_indx, args, max_length=250, stem='data/beer_review/reviews.aspect'):
        self.base_path = stem
        self.aspect= args.aspect
        self.objective = args.objective
        
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.aspects_to_num = {'appearance':0, 'aroma':1, 'palate':2}#,'taste':3, 'overall':4}
        self.class_map = {0: 0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 8:2, 9:2, 10:2}
        self.name_to_key = {'train':'train', 'dev':'heldout', 'test':'heldout'}
        self.class_balance = {}
        
        self.training_set = self._get_data("train")
        self.dev_set = self._get_data("dev")
        self.test_set = self._get_data("test")

        if self.name == 'train':
            self.explanations_vocab = self._get_explanations_vocab()
        print ("Class balance", self.class_balance)

    def training(self):
        return self.training_set
    
    def dev(self):
        return self.dev_set

    def test(self):
        return self.test_set

    def override(self, args):
        self.args.update(args)
        return self

    def _get_data(self, name="train"):
        dataset = []
        file_name = f"{self.base_path}{self.aspects_to_num[self.aspect]}.{self.name_to_key[name]}.txt.gz"
        with gzip.open(file_name) as gfile:
            lines = gfile.readlines()
            lines = list(zip( range(len(lines)), lines) )
            if name == 'dev':
                lines = lines[:5000]
            elif name == 'test':
                lines = lines[5000:10000]
            elif name == 'train':
                lines = lines[0:20000]
            for indx, line in tqdm.tqdm(enumerate(lines)):
                uid, line_content = line
                sample = self.processLine(line_content, self.aspects_to_num[self.aspect], indx)

                if not sample['y'] in self.class_balance:
                    self.class_balance[ sample['y'] ] = 0
                self.class_balance[ sample['y'] ] += 1
                sample['uid'] = uid
                dataset.append(sample)
        return dataset

    def _get_explanations_vocab(self):
        vocabulary = TfidfVocabulary(self.args, self.dataset)
        possible_explanations = vocabulary.possible_explanations()
        explanations_vocab = {}
        for e_id, text in possible_explanations.items():
            index=get_indices_tensor([text],
                                       self.word_to_indx, 1)
            if index[0] != 0:
                idx = len(explanations_vocab)
                explanations_vocab[idx] = { "emb":
                                            index, # one word
                                           "text": text} 
        return explanations_vocab

    ## Convert one line from beer dataset to {Text,explanations_vocab Tensor, Labels}
    def processLine(self, line, aspect_num, i):
        if isinstance(line, bytes):
            line = line.decode()
        labels = [ float(v) for v in line.split()[:5] ]
        if self.objective == 'mse':
            label = float(labels[aspect_num])
            self.args.num_class = 1
        else:
            # 0 [0-3]/ 1 [4-7]/ 2 [8-10] for the corresponding aspect
            label = int(self.class_map[ int(labels[aspect_num] *10) ])
            self.args.num_class = 3
        text_list = line.split('\t')[-1].split()[:self.max_length]
        text = " ".join(text_list)
        x =  get_indices_tensor(text_list, self.word_to_indx, self.max_length)
        # x = encoding of the text
        sample = {'text':text,'x':x, 'y':label, 'i':i}
        return sample
