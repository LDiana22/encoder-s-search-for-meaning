# -*- coding: utf-8 -*-
import torch
import rationale_net.datasets.factory as dataset_factory
import rationale_net.utils.embedding as embedding
import tqdm
import rationale_net.utils.helpers as helpers
import scripts.args as generic

def decode_mask(mask, vocab):
    return [vocab[i] for i in range(len(mask)) if mask[i]!=0]


args = generic.parse_args()

embeddings, word_to_indx = embedding.get_embedding_tensor(args)
print("Loaded embeddings")
 
train_data, dev_data, test_data, explanation_vocab = dataset_factory.get_dataset(args, word_to_indx)

gen = torch.load("snapshot/demo_run.pt.gen")
gen.training=False

test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)    

data_iter = test_loader.__iter__()

num_batches_per_epoch = len(data_iter)
with open("explanations.txt", "w") as f:
    for _ in tqdm.tqdm(range(num_batches_per_epoch)):
        batch = data_iter.next()
        
        x_indx = helpers.get_x_indx(batch, args, True)
        text = batch['text']
        masks, z = gen(x_indx)
        for i, mask in enumerate(masks):
            explanations = decode_mask(mask, explanation_vocab)
            print(text[i], "\n**\n", explanations, file=f)