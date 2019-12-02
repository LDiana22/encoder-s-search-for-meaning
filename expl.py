# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:02:48 2019

@author: Diana
"""

import pickle
with open("app.results", "rb") as f:
    res = pickle.load(f)
    for rationale in res['test_stats']['rationales']:
        idx = [i for i in range(len(rationale.split())) if rationale.split()[i]!="_"]
        print(idx)
    print(res)