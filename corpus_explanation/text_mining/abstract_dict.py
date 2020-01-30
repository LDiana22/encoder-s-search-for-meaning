import os
import pickle

class AbstractDictionary:
  def __init__(self, id, dataset, args):
    self.id = id
    self.dataset = dataset
    self.args = args 
    self.path = os.path.join(self.args["prefix_dir"], self.args["dirs"]["dictionary"], id)

  def _save_dict(self):
    if not os.path.isdir(self.path):
      os.makedirs(self.path)
    file = os.path.join(self.path, "dictionary.h5")
    with open(file, "wb") as f:
      f.write(pickle.dumps(self.dictionary))

    file = os.path.join(self.path, "dictionary.txt")
    with open(file, "w", encoding="utf-8") as f:
      f.write(str(self.dictionary))
