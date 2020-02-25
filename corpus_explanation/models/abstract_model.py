from abc import ABC
from modelsummary import  summary
import os.path
from datetime import datetime
import torch
from contextlib import redirect_stdout
from torch import nn
from datetime import datetime

from utils import file_helpers as fh

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AbstractModel(nn.Module):
    """
    Abstract Model
        - saves the mapping between the model-id and its parameters and
            model summary
        - creates the directories for the log files
    """
    def __init__(self, id, mapping_file_location, model_args):
        """
        id: Model id
        mapping_file_location: directory to store the file "model_id" 
                               that containes the hyperparameters values and 
                               the model summary
        logs_location: directory for the logs location of the model
        model_args: hyperparameters of the model
        """
        super().__init__()
        self.delim = "#################################"
        self.id = id
        self.mapping_location = mapping_file_location
        self.args = model_args
        self.device = torch.device('cuda' if model_args["cuda"] else 'cpu')
        self.model_dir = model_dir = os.path.join(self.args["prefix_dir"], self.id)
        self.__create_directories()

    def override(self, args):
        self.args.update(args)

    def __create_directories(self):
        """
        All the directories for a model are placed under the directory 
            prefix_dir / model_id / {dirs}
        """ 
        self.checkpoint_dir = os.path.join(self.model_dir, self.args["dirs"]["checkpoint"])
        for directory in self.args["dirs"].values():
            m_dir = os.path.join(self.model_dir, directory)
            if not os.path.isdir(m_dir):
                os.makedirs(m_dir)
        if not os.path.isdir(self.mapping_location):
            os.makedirs(self.mapping_location)

    def save_model_type(self, model):
        """
        Saves the hyperparameters 
        """
        mapping_file = os.path.join(self.mapping_location, self.id)        
        with open(mapping_file, "w") as map_file:
            print(self.delim, file=map_file)
            print(self.args, file=map_file)
            print(self.delim, file=map_file)
            print(self, file=map_file)
            print(self.delim, file=map_file)

    def checkpoint(self, epoch, metrics):
        checkpoint_file = os.path.join(self.checkpoint_dir, 
            f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_e{epoch}')
        self.dict_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }
        self.dict_checkpoint.update(metrics)
        torch.save(self.dict_checkpoint, checkpoint_file)

    def load_checkpoint(self, newest_file_name):
        checkpoint_dir = os.path.join(self.model_dir, self.args["dirs"]["checkpoint"])           

        path = os.path.join(checkpoint_dir, newest_file_name)
        print(f"Loading checkpoint: {path}") 
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.metrics = {}
        for key in checkpoint.keys():
            if key not in ['epoch', 'model_state_dict', 'optimizer_state_dict']:
                self.metrics[key] = checkpoint[key]

    def save_results(self, metrics):
        metrics_path = os.path.join(self.model_dir, self.args["dirs"]["metrics"])
        results_file = os.path.join(metrics_path, f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        with open(results_file, "w") as f:
            f.write(str(metrics))

    # def train_model(self, iterator):
    #     """
    #     Abstract method, avoiding multiple inheritance
    #     Return a metrics dict with the keys prefixed by 'train'
    #     e.g. metrics={"train_acc": 90.0, "train_loss": 0.002}
    #     """
    #     pass

    # def evaluate(self, iterator, prefix="test"):
    #     """
    #     Abstract method, avoiding multiple inheritance

    #     Return a metrics dict with the keys prefixed by prefix
    #     e.g. metrics={f"{prefix}_acc": 90.0, f"{prefix}_loss": 0.002}
    #     """
    #     pass


    def train_model(self, iterator):
        """
        metrics.keys(): [train_acc, train_loss, train_prec,
                        train_rec, train_f1, train_macrof1,
                        train_microf1, train_weightedf1]
        """
        e_loss = 0
        e_acc, e_prec, e_rec = 0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0

        self.train()

        for batch in iterator:
            self.optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = self.forward(text, text_lengths)
            batch.label = batch.label.to(self.device)
            loss = self.criterion(predictions, batch.label)

            y_pred = torch.round(predictions).detach().cpu().numpy()
            y_true = batch.label.cpu().numpy()
            #metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            macrof1 = f1_score(y_true, y_pred, average='macro')
            microf1 = f1_score(y_true, y_pred, average='micro')
            wf1 = f1_score(y_true, y_pred, average='weighted')

            loss.backward()
            self.optimizer.step()

            e_loss += loss.item()
            e_acc += acc
            e_prec += prec
            e_rec += rec
            e_f1 += f1
            e_macrof1 += macrof1
            e_microf1 += microf1
            e_wf1 += wf1
        
        metrics ={}
        size = len(iterator)
        metrics["train_loss"] = e_loss/size
        metrics["train_acc"] = e_acc/size
        metrics["train_prec"] = e_prec/size
        metrics["train_rec"] = e_rec/size
        metrics["train_f1"] = e_f1/size
        metrics["train_macrof1"] = e_macrof1/size
        metrics["train_microf1"] = e_microf1/size
        metrics["train_weightedf1"] = e_wf1/size

        return metrics

    def evaluate(self, iterator, prefix="test"):
        """
            Return a metrics dict with the keys prefixed by prefix
            metrics = {}
        """
        self.eval()

        e_loss = 0
        e_acc, e_prec, e_rec = 0,0,0
        e_f1, e_macrof1, e_microf1, e_wf1 = 0,0,0,0
        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text
                predictions = self.forward(text, text_lengths)
                batch.label = batch.label.to(self.device)
                loss = self.criterion(predictions, batch.label)
    
                predictions = torch.round(predictions)

                y_pred = torch.round(predictions).detach().cpu().numpy()
                y_true = batch.label.cpu().numpy()
                
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                macrof1 = f1_score(y_true, y_pred, average='macro')
                microf1 = f1_score(y_true, y_pred, average='micro')
                wf1 = f1_score(y_true, y_pred, average='weighted')

                e_loss += loss.item()
                e_acc += acc
                e_prec += prec
                e_rec += rec
                e_f1 += f1
                e_macrof1 += macrof1
                e_microf1 += microf1
                e_wf1 += wf1

        metrics ={}
        size = len(iterator)
        metrics[f"{prefix}_loss"] = e_loss/size
        metrics[f"{prefix}_acc"] = e_acc/size
        metrics[f"{prefix}_prec"] = e_prec/size
        metrics[f"{prefix}_rec"] = e_rec/size
        metrics[f"{prefix}_f1"] = e_f1/size
        metrics[f"{prefix}_macrof1"] = e_macrof1/size
        metrics[f"{prefix}_microf1"] = e_microf1/size
        metrics[f"{prefix}_weightedf1"] = e_wf1/size
        return metrics
