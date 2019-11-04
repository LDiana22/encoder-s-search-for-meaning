from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

import rationale_net.datasets.factory as dataset_factory
import rationale_net.utils.embedding as embedding
import rationale_net.utils.model_factory as model_factory
import scripts.args as generic
import rationale_net.utils.model_train as model_helper
import os
import pickle



if __name__ == '__main__':
    # update args and print
    args = generic.parse_args()
    print("Loading embeddings")
    embeddings, word_to_indx = embedding.get_embedding_tensor(args)
    print("Loaded embeddings")
    
 
    train_data, dev_data, test_data, explanation_vocab = dataset_factory.get_dataset(args, word_to_indx)

    results_path_stem = args.results_path.split('/')[-1].split('.')[0]
    args.model_path = '{}.pt'.format(os.path.join(args.save_dir, results_path_stem))

    # model
    gen, model = model_factory.get_model(args, embeddings, train_data, explanation_vocab)

    print()
    # train
    if args.train :
        epoch_stats, model, gen = model_helper.train_model(train_data, dev_data, model, gen, args)
        args.epoch_stats = epoch_stats
        save_path = args.results_path
        print("Save train/dev results to", save_path)
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path,'wb') )


    # test
    if args.test :
        test_stats = model_helper.test_model(test_data, model, gen, args)
        args.test_stats = test_stats
        args.train_data = train_data
        args.test_data = test_data

        save_path = args.results_path
        print("Save test results to", save_path)
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path,'wb') )
