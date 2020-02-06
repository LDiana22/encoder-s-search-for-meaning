import os
import pickle

class AbstractDictionary:
  def __init__(self, id, dataset, args):
    """
    A dictionary consists of a list of entries per each class.
    For a corpus dictionary, there is only one "dummy class" considered.
    """
    self.id = id
    self.dataset = dataset
    self.args = args 
    self.path = os.path.join(self.args["prefix_dir"], self.args["dirs"]["dictionary"], id)
    self.metrics = {}

  def _save_dict(self):
    if not os.path.isdir(self.path):
      os.makedirs(self.path)
    file = os.path.join(self.path, "dictionary.h5")
    with open(file, "wb") as f: 
      f.write(pickle.dumps(self.dictionary))

    file = os.path.join(self.path, "dictionary.txt")
    with open(file, "w", encoding="utf-8") as f:
      f.write(str(self.dictionary))

    self.print_metrics()

  def _compute_metrics(self):
    overlap = 0 # number of overlapped entries for each label
    global_avg_w = 0 # global average words per instance
    global_count = 0
    class_avg_w = {}
    word_intersection = None
    for class_label in self.dictionary.keys():
      instances = self.dictionary[class_label]
      no_instances = len(instances)
      if word_intersection is None:
        word_intersection = set(instances)
      else:
        word_intersection = set(instances).intersection(word_intersection)
        overlap = len(word_intersection)
      sum_number_of_words = sum([len(instance.split(" ")) for instance in instances])
      class_avg_w[class_label] = sum_number_of_words/no_instances
      global_avg_w += sum_number_of_words
      global_count += no_instances
    if global_count:
      global_avg_w = global_avg_w/global_count
    self.metrics = {
      "dictionary_entries": global_count,
      "overlap_count": overlap,
      "global_average_words_per_instance": global_avg_w,
      "class_average": class_avg_w,
      "overlap_words": word_intersection
    }

  def print_metrics(self):
    if not self.metrics:
      self._compute_metrics()
    metrics_path = os.path.join(self.path, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
      f.write(str(self.metrics))
