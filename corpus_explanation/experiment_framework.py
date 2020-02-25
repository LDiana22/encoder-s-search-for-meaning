# -*- coding: utf-8 -*-
from utils import file_helpers as fh
import torch
import time
from datetime import datetime
import os
from tqdm import tqdm

class Experiment(object):
  """Holds all the experiment parameters and provides helper functions."""
  def __init__(self, e_id):
    self.id = e_id
      
  def restore_model(self):
    if self.config["restore_checkpoint"]:
      checkpoint = self.model.checkpoint_dir
      if self.config["checkpoint_file"] is None:
        last_checkpoint = fh.get_last_checkpoint_by_date(checkpoint)
      else:
        last_checkpoint = self.config["checkpoint_file"]
      if last_checkpoint is not None:
        self.model.load_checkpoint(last_checkpoint)
        return True  
      else:
        print(f"No checkpoint found at {checkpoint}")
    return False

  def setup(self):
      self.restore_model()
      return self
              

  ### DECLARATIVE API ###

  def with_data(self, data):
      self.data = data
      return self

  def with_dictionary(self, dictionary):
      self.dictionary = dictionary
      return self

  def with_config(self, config):
      self.config = config.copy()
      return self

  def override(self, config):
      self.config.update(config)
      return self

  def with_model(self, model):
      self.model = model
      return self
  #### END API ######
  
  @property
  def experiment_name(self):
      return f'E-{self.id}_M-{self.model.id}'

  """ Dirs
  - *_dir - full path to dir
  """
  @property
  def experiments_dir(self):
      return "experiments"

  def train_model(self):
    training_start_time = datetime.now()

    training_losses, training_acc = [], []
    v_losses, v_acc = [], []

    best_valid_loss = float('inf')
    n_epochs = self.config["epochs"]
    for epoch in tqdm(range(n_epochs)):
      start_time = datetime.now()

      train_metrics = self.model.train_model(self.train_iterator)
      valid_metrics = self.model.evaluate(self.valid_iterator, "valid")
      
      end_time = datetime.now()
      
      training_losses.append(train_metrics["train_loss"])
      training_acc.append(train_metrics["train_acc"])
      v_losses.append(valid_metrics["valid_loss"])
      v_acc.append(valid_metrics["valid_acc"])

      if valid_metrics["valid_loss"] < best_valid_loss:
        best_valid_loss = valid_metrics["valid_loss"]
        metrics = train_metrics
        metrics.update(valid_metrics)
        self.model.checkpoint(epoch, metrics)
    
      print(f'Epoch: {epoch+1:02} | Epoch Time: {str(end_time-start_time)}')
      print(f'\tTrain Loss: {train_metrics["train_loss"]:.3f} | Train Acc: {train_metrics["train_acc"]*100:.2f}%')
      print(f'\t Val. Loss: {valid_metrics["valid_loss"]:.3f} |  Val. Acc: {valid_metrics["valid_acc"]*100:.2f}%')

    
    print(f'Training Time: {str(datetime.now()-training_start_time)}')
    print(f'Training losses: {training_losses}')
    print(f'Training acc: {training_acc}')
    print(f'Valid losses: {v_losses}')
    print(f'Valid acc: {v_acc}')

  def run(self):
      if self.config["restore_checkpoint"]:
        loaded = self.restore_model()
        if not loaded:
          return
      self.train_iterator, self.valid_iterator, self.test_iterator = self.data.iterators()
      if self.config["train"]:
        print("Training...")
        self.train_model()
      print("Evaluating...")
      metrics = self.model.evaluate(self.test_iterator)
      self.model.save_results(metrics)
      
      