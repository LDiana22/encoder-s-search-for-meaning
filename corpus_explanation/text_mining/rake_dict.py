import pickle
import os
from rake_nltk import Rake
from .abstract_dict import AbstractDictionary

import spacy

class RakePerClassExplanations(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    self.max_dict = args.get("max_dict", None)
    self.max_words = args["max_words_dict"]
    self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    self.dictionary = self.get_dict()
    self.tokenizer = spacy.load("en")
    self._save_dict()

  def get_dict(self):
    """
    Builds a dictionary of keywords for each label.
    # {"all":{word:freq}} OR
    {"pos":{word:freq}, "neg":{word:freq}}
    """
    if hasattr(self, 'dictionary') and not self.dictionary:
      return self.dictionary
    dictionary = {} 
    corpus = self.dataset.get_training_corpus()

    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
      dictionary[text_class] = {}
      class_corpus = " ".join(corpus[text_class])
      self.rake.extract_keywords_from_text(class_corpus)
      phrases = self.rake.get_ranked_phrases()[:max_per_class]
      if max_per_class:
        phrases = phrases[:max_per_class]
      # get word freq
      # tok_words = self.tokenizer(class_corpus)
      # word_freq = Counter([token.text for token in tok_words if not token.is_punct])
      # build dict
      for phrase in phrases:
        # trim phrase to max words
        phrase = " ".join(phrase.split()[:self.max_words])
        dictionary[text_class][phrase] = class_corpus.count(phrase)

    return dictionary


class MixedRakePerClassExplanations(AbstractDictionary):

  def __init__(self, id, dataset, args): 
    super().__init__(id, dataset, args)
    
  def get_dict(self):
    """
    Builds a dictionary of keywords for each label and returns one dictionary as 
    """
    if hasattr(self, 'dictionary'):
      return self.dictionary
    dictionary = {}
    corpus = self.dataset.get_training_corpus()

    max_per_class = int(self.max_dict / len(corpus.keys())) if self.max_dict else None
    for text_class in corpus.keys():
      self.rake.extract_keywords_from_text(" ".join(corpus[text_class]))
      phrases = self.rake.get_ranked_phrases()[:max_per_class]
      for phrase in phrases:
        dictionary[phrase] = dictionary.get(phrase, {}).update({text_class: 1}) # TODO
      if max_per_class:
        dictionary[text_class] = dictionary[text_class][:max_per_class]
    return dictionary