import pickle
import os
from rake_nltk import Rake
from .abstract_dict import AbstractDictionary

class RakePerClassDictionary(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args.get("max_dict", None)
    self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    self.dictionary = self._build_dict()
    self._save_dict()

  def _build_dict(self):
    """
    Builds a dictionary of keywords for each label.
    """
    dictionary = {}
    corpus = self.dataset.get_training_corpus()

    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
      self.rake.extract_keywords_from_text(" ".join(corpus[text_class]))
      dictionary[text_class] = self.rake.get_ranked_phrases()[:max_per_class]
      if max_per_class:
        dictionary[text_class] = dictionary[text_class][:max_per_class]
    return dictionary
