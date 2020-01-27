# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:02:48 2019

@author: Diana
"""

<<<<<<< HEAD
import pickle
with open("app.results", "rb") as f:
=======
import sys
import pickle

file = sys.argv[1]
with open(file, "rb") as f:
>>>>>>> corpus
    res = pickle.load(f)
    for rationale in res['test_stats']['rationales']:
        idx = [i for i in range(len(rationale.split())) if rationale.split()[i]!="_"]
        print(idx)
<<<<<<< HEAD
    print(res)
=======
>>>>>>> corpus
