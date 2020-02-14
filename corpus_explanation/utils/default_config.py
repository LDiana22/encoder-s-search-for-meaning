# -*- coding: utf-8 -*-

CONFIG = {
    "toy_data": False, # load only a small subset

    "cuda": False,

    "embedding": "glove",

    "restore_checkpoint" : False,
    "checkpoint_file": None,
    "train": True,

    "dropout": 0.05,
    "weight_decay": 5e-06,

    "patience": 5,

    "epochs": 50,

    "objective": "cross_entropy",
    "init_lr": 0.0001,

    "gumbel_decay": 1e-5,

    "prefix_dir" : "experiments",

    
    "dirs": {
        "metrics": "metrics",
        "checkpoint": "snapshot",
        },

    "aspect": "palate", # aroma, palate, smell, all
    "max_vocab_size": 25000,
    "emb_dim": 300,
    "batch_size": 32,
    "output_dim": 1,
}

MODEL_MAPPING = "experiments/models_mappings"
