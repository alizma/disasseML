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
    def establish__model(self, name): 
        return self.establish(name, FileManager._model_name)
    
    def load_model(self, name): 
        return keras.models.load_model(self.establish__model(name))

    def save_model(self, model, name): 
        model.save(self.establish__model(name), overwrite=True)

    ### LOG SPECIFIC METHODS ### 
    def establish_log(self): 
        return self.establish(FileManager._log_name)

    @property 
    def log_file_path(self):    
        return self.establish_log() 

    def open_log(self, *args, **kwargs): 
        return open(self.establish_log(), 'w', *args, **kwargs)

    # TRAINING-SPECIFIC                                                                    
    def qualify_training(self, name):
        '''
        Get the qualified filename of a training set.
        '''
        return self.qualify(name, FileManager._training_name)

    def open_training(self, name, *args, **kwargs):
        '''
        Open training set file.
        :param name: The name of the training set.
        :param args: Extra arguments for open().
        :param kwargs: Keyword arguments for open(). Note: Any 'newline' key will be overridden with the value of '\n'.
        :returns: An open handle to the training set file.
        '''
        kwargs['newline'] = '\n'
        return open(self.qualify_training(name), 'r', *args, **kwargs)

    def load_training(self, name, codecs, block_size=65536, max_records=np.inf):
        '''
        Load (up to) an entire training set into memory at once.
        :param name: The model name.
        :param codecs: A tuple of (AsciiCodec,BytsCodec).
        :param block_size: The amount of data to read at once. Affects I/O performance but probably isn't critical.
        Default is 64K.
        :param max_records: The maximum number of records to load. Default: infinity, which means load everything.
        :returns: A tuple of the training inputs and targets.
        '''
        with self.open_training(name) as file:
            return _do_load_training(file, codecs, block_size, max_records)

    def yield_training(self, name, codecs, batch_size, block_size=65535, max_records=np.inf, loop_mode=False):
        '''
        Yield training samples in batches.
        :param name: The name of the training set.
        :param codecs: A tuple of (AsciiCodec,BytsCodec).
        :param batch_size: The number of records in each batch. The actual size of a batch may be smaller than
        batch_size if there are fewer records in the file, or max_records is smaller than batch_size.
        :param block_size: How many bytes to load from the file at once. This can effect performance, but not the number
        of records returned - more than one block will be read if necessary.
        :param max_records: The maximum number of records to load. Overrides batch_size if max_records is smaller.
        Default value is infinity, which means load up to batch_size or the entire file, whichever is smaller.
        :param loop_mode: If True, the generator loops over the training set indefinitely. Default is False.
        :yields: A tuple of the training inputs and targets.
        '''
        with self.open_training(name) as file:
            batch_num   = 0
            num_records = max_records
            while True:
                # Load the next batch of records.
                count = min(batch_size, num_records)
                batch_num += 1
                Xy = None
                if count:
                    Xy = _do_load_training(file, codecs, block_size, count, line_num=batch_num*count)
                # Break or reset if we reached EOF.
                if not Xy:
                    if not loop_mode:
                        break
                    # mldisasm.training.kfolds_train requires that the generator loop over its data repeatedly. This
                    # breaks Python generator semantics (generators are single-use) but is needed for cross-validation.
                    log.debug('Restarting training file generator')
                    file.seek(0)
                    count       = batch_size
                    batch_num   = 0
                    num_records = max_records
                    continue
                # Check & yield results. Update num_records so we load fewer records next time if
                X, y = Xy
                assert len(X.shape) == 3
                assert len(y.shape) == 3
                assert X.shape[0] == y.shape[0]
                num_records -= int(X.shape[0])
                if num_records < 0:
                    num_records = 0
                yield X, y

    # VALIDATION-SPECIFIC                                                                
    def qualify_validation(self, name):
        '''
        Get the qualified filename of a validation set.
        '''
        return self.qualify(name, FileManager._validation_name)

    def open_validation(self, name):
        '''
        Open validation file.
        '''
        return open(self.qualify_validation(name), 'r')

    def load_validation(self, name, codecs, block_size=65536, max_records=np.inf):
        '''
        Load (up to) an entire validation set into memory at once.
        :param name: The model name.
        :param codecs: A tuple of (AsciiCodec,BytsCodec).
        :param block_size: The amount of data to read at once. Affects I/O performance but probably isn't critical.
        Default is 64K.
        :param max_records: The maximum number of records to load. Default: infinity, which means load everything.
        :returns: A tuple of the training inputs and targets.
        '''
        with self.open_validation(name) as file:
            return _do_load_training(file, codecs, block_size, max_records)

    def yield_validation(self, name, codecs):
        '''
        Yield validation samples one at a time.
        :param name: The model name.
        :param codecs: A tuple of (BytesCodec,AsciiCodec).
        :yields: A single pair of validation inputs and targets.
        '''
        with self.open_validation(name) as file:
            x_codec, y_codec = codecs
            for line in file:
                opcode, disasm = line.split('|')
                yield x_codec.encode(opcode), disasm

    ############################################################################
    # CONSTANTS                                                                #
    ############################################################################
    _log_name          = 'mldisasm.log'     # Log filename.
    _config_name       = 'config.json'      # Config filename.
    _model_name        = 'model.h5'         # Model filename.
    _training_name     = 'training.csv'     # Preprocessed training set filename.
    _validation_name   = 'validation.csv'   # Validation training set filename.
    _tokens_name       = 'tokens.list'      # Token list filename.

def _do_load_training(file, codecs, block_size, max_records, line_num=1):
    '''
    Load up to `max_records` from `file` using blocks of `block_size` bytes.
    :returns: A tuple of the training inputs and labels, or None if there are no records left in the file.
    '''
    with prof('Loaded batch ({} records)', lambda: num_lines):
        num_lines = 0
        x_codec, y_codec = codecs
        # Blocks are likely to end partway through a record, so we read more data than we need and discard any
        # excess records. We save the file position so we can calculate the amount of data actually used and seek to
        # the beginning of the discarded record(s) for the next read.
        file_pos  = file.tell()
        data      = ''
        num_lines = 0
        while data.count('\n') <= max_records:
            block = file.read(block_size)
            if not block:
                break
            data += block
        if not data:
            return None
        # Split on newline and discard records above the maximum.
        lines     = data.split('\n')
        num_lines = min(len(lines), max_records)
        # Process the records and rewind to account for any extra records read.
        X = [None]*num_lines
        y = [None]*num_lines
        len_lines = 0
        for i in range(num_lines):
            line_num += i
            len_lines += len(lines[i]) + 1 # +1 to account for newline stripped by str.split().
            opcode, disasm = lines[i].split('|')
            X[i] = x_codec.encode(opcode)
            y[i] = y_codec.encode(disasm)
        file.seek(file_pos + len_lines)
        return tf.convert_to_tensor(X), tf.convert_to_tensor(y)

