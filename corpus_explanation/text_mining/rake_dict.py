import pickle
import os
from rake_nltk import Rake


class RakeDictionary():

  def __init__(self, id, dataset, args): 
    self.args = args 
    self.path = os.path.join(self.args["prefix_dir"], self.args["dirs"]["dictionary"], id)
    self.id = id
    self.dataset = dataset
    self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    self.dictionary = self._build_dict()
    self._save_dict()

  def _build_dict(self):
    dictionary = {}
    corpus = self.dataset.get_training_corpus()
    for text_class in corpus.keys():
      self.rake.extract_keywords_from_text(" ".join(corpus[text_class]))
      dictionary[text_class] = self.rake.get_ranked_phrases()
    return dictionary

  def _save_dict(self):
    if not os.path.isdir(self.path):
      os.makedirs(self.path)
    file = os.path.join(self.path, "dictionary.h5")
    with open(file, "wb") as f:
      f.write(pickle.dumps(self.dictionary))

    file = os.path.join(self.path, "dictionary.txt")
    with open(file, "w") as f:
      f.write(str(self.dictionary))
