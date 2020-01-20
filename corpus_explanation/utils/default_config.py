# -*- coding: utf-8 -*-

CONFIG = {
    "cuda": False,

    "embedding": "glove",

    "restore_checkpoint" : False,

    "dropout": 0.05,
    "weight_decay": 5e-06,

    "patience": 5,

    "epochs": 50,

    "objective": "cross_entropy",
    "init_lr": 0.0001,

    "gumbel_decay": 1e-5,

    "prefix_dir" : "experiments",

    
    "dirs": {
        "logs_dir": "metrics",
        "checkpoint": "snapshot"
        },

    "aspect": "palate", # aroma, palate, smell, all
    "max_vocab_size": 25000,
    "emb_dim": 300,
    "batch_size": 32
}

MODEL_MAPPING = "experiments/models_mappings"
