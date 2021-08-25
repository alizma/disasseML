import os
from h5py._hl.files import File 

import numpy as np 

import tensorflow as tf
import tensorflow.keras as keras 

try: 
    # https://github.com/ultrajson/ultrajson
    import ujson as json 
except: 
    import json 

from disasseml.util import prof, log

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

    def establish(self, *args): 
        return os.path.join(self._data_dir, *args)

    ### CONFIGURATION SPECIFIC METHODS ### 
    def _establish_config(self, name): 
      # establishes path to config file 
      # https://www.geeksforgeeks.org/python-os-path-join-method/ 
      return self.establish(name, FileManager._config_name)

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

    def save_config(self, name, config): 
        with self._open_config(name, 'w', newline='\n') as file: 
            json.dump(config, file, indent=4, )
            file.write('\n')

    ### MODEL SPECIFIC METHODS ### 
    def _establish__model(self, name): 
        return self.establish(name, FileManager._model_name)
    
    def load_model(self, name): 
        return keras.models.load_model(self._establish__model(name))

    def save_model(self, model, name): 
        model.save(self._establish__model(name), overwrite=True)

    ### LOG SPECIFIC METHODS ### 
    def establish_log(self): 
        return self.establish(FileManager._log_name)

    @property 
    def log_file_path(self):    
        return self.establish_log() 

    def open_log(self, *args, **kwargs): 
        return open(self.establish_log(), 'w', *args, **kwargs)

