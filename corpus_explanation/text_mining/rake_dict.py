import pickle
import os
from rake_nltk import Rake
from .abstract_dict import AbstractDictionary

class RakePerClassDictionary(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    self.dictionary = self._build_dict()
    self._save_dict()

  def _build_dict(self):
    """
    Builds a dictionary of keywords for each label.
    """
    dictionary = {}
    corpus = self.dataset.get_training_corpus()
    for text_class in corpus.keys():
      self.rake.extract_keywords_from_text(" ".join(corpus[text_class]))
      dictionary[text_class] = self.rake.get_ranked_phrases()
    return dictionary
