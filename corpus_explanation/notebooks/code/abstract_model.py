from abc import ABC
from collections import OrderedDict 
from collections import ChainMap
from contextlib import redirect_stdout
import copy
import glob
import io
import itertools
import os
import os.path
import pickle
import random
import re
import spacy
import string
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datetime import datetime

import torch

from torch import nn
from torch.utils import data as utils

import torch.nn.functional as F

import torch.optim as optim

from torchtext import datasets
from torchtext import data as data
from torchtext.vocab import GloVe
from torchtext.data import Pipeline

import numpy as np

import yake
from summa import keywords
from rake_nltk import Rake
