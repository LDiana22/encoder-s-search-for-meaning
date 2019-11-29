# -*- coding: utf-8 -*-

CONFIG = {
    "cuda": True,

    "embedding": "glove",
    "restore_checkpoint" : False
}

MODEL_ARGS = {
    "dropout": 0.05,
    "weight_decay": 5e-06,

    "patience": 5,

    "epochs": 50,
    "batch_size": 32,

    "objective": "cross_entropy",
    "init_lr": 0.0001,

    "gumbel_decay": 1e-5,

    "prefix_dir" : "experiments",
    
    "dirs": {
        "logs_dir": "metrics",
        "checkpoint": "snapshot"
        }
}

MODEL_MAPPING = "experiments/models_mappings"

DATASET_ARGS = {
    "aspect": "palate", # aroma, palate, smell, all
}
