import json 
import pickle 
import os 

from disasseml.io.trainingset import TrainSet

class FileManager:
    # https://www.tutorialspoint.com/python/os_getcwd.htm
    default_dir = os.path.join(os.getcwd(), 'data')

    def __init__(self, data_dir=None): 
        data_dir = FileManager.default_dir
        self.chdir(data_dir)

    def chdir(self, path): 
        # change managed directory 
        if path is None: 
            raise ValueError("Can't change to given path")
        self._data_dir = path 

    def load_config(self, name, *args, **kwargs): 
        '''
        load configuration information 
        name: configuration name 
        args: extra args for open() 
        kwargs: extra keyword args for open() 
        '''
        with self._open_config(name, *args, **kwargs) as file: 
            return json.load(file)

    def _open_config(self, name, *args, **kwargs): 
        # opens configuration file
        return open(self._establish_config(name), *args, **kwargs)

    def _establish_config(self, name): 
        # establishes path to config file 
        # https://www.geeksforgeeks.org/python-os-path-join-method/ 
        return os.path.join(self._data_dir, name, 'config.json')

    def open_model(self, name, *args, **kwargs): 
        # opens model and returns a handle to the model file 
        return open(self._establish_model(name), 'rb', *args, **kwargs)

    def _establish__model(self, name): 
        return os.path.join(self._data_dir, name, 'model.pkl')
    
    def open_train(self, name, *args, **kwargs): 
        # opens train set file and returns a handle to the train set file 
        return TrainSet(self._establish_training(name), *args, **kwargs)

    def _establish_training(self, name): 
        return os.path.join(self._data_dir, name, 'train.csv')