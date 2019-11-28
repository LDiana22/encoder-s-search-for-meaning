# -*- coding: utf-8 -*-
import torch
import rationale_net.datasets.factory as dataset_factory
import rationale_net.utils.embedding as embedding
import tqdm
import rationale_net.utils.helpers as helpers
import scripts.args as generic
import numpy as np

def decode_mask(mask, vocab):
    return [vocab[i] for i in range(len(mask)) if mask[i]!=0]


args = generic.parse_args()
embeddings, word_to_indx = embedding.get_embedding_tensor(args)
print("Loaded embeddings")
 
train_data, dev_data, test_data, explanation_vocab = dataset_factory.get_dataset(args, word_to_indx)

args.expl_vocab = torch.tensor([explanation_vocab[e_id]["emb"].data for e_id in sorted(explanation_vocab.keys())])
args.expl_text = np.array([explanation_vocab[e_id]["text"] for e_id in sorted(explanation_vocab.keys())])
print(args.expl_vocab)
print(len(args.expl_vocab))
if args.cuda:
    args.expl_vocab = args.expl_vocab.cuda()


gen = torch.load("snapshot/palate-filtered-v2/demo_run_palate_filtered-v2.pt.gen")#, map_location=torch.device('cpu'))
gen.training=False

test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)    

data_iter = test_loader.__iter__()

e=[]

torch.set_printoptions(profile="full")

num_batches_per_epoch = len(data_iter)
with open("explanations_palate_filtered-v2.txt", "w") as f:
    for _ in tqdm.tqdm(range(num_batches_per_epoch)):
        batch = data_iter.next()
        
        x_indx = helpers.get_x_indx(batch, args, True)
        if args.cuda:
            x_indx=x_indx.cuda()
        text = batch['text']
        masks, z = gen(x_indx)
        for i, mask in enumerate(masks):
            print(i)
            print(mask)
            idx=(mask>0.5).nonzero().squeeze().tolist()
            print(idx)
            explanation = args.expl_text[idx]
            #explanations = decode_mask(mask, args.expl_text)
            print("**\n",text[i], "\n~\n",explanation, "\n**\n",file=f)
            #e.append(explanations[0]['text'])
            e.append(explanation)
print(e)
