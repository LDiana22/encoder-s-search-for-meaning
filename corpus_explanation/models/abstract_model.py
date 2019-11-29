from abc import ABC

import os.path

class AbstractModel(ABC):
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
        self.id = id
        self.mapping_location = mapping_file_location
        self.args = model_args
        self.__create_directories()
        self.__save_model_type()

    def override(self, args):
        self.args.update(args)

    def __create_directories(self):
        """
        All the directories for a model are placed under the directory 
            prefix_dir / model_id / {dirs}
        """
        for directory in self.args["dirs"].values():
            model_dir = os.path.join(self.args["prefix_dir"],
                                     self.id, directory)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
        if not os.path.isdir(self.mapping_location):
            os.makedirs(self.mapping_location)

    def __save_model_type(self):
        """
        Saves the hyperparameters 
        """
        mapping_file = os.path.join(self.mapping_location, self.id)        
        with open(mapping_file, "w") as map_file:
            print(self.args, file=map_file)
            
