import numpy as np 

import tensorflow as tf

BYTE_MAX = 0xFF
ASCII_MAX = 0x7F

INPUT_SIZE = BYTE_MAX + 1
TARGET_SIZE = ASCII_MAX + 1

def one_hot_bytes(bytes): 
    # one-hot encoding of passed bytes 
    # returns a one-hot tensor with one row per byte and INPUT_SIZE (256) elements per row
    # https://www.wikiwand.com/en/One-hot
    # https://www.tensorflow.org/api_docs/python/tf/one_hot
    return tf.one_hot(list(bytes), depth=INPUT_SIZE)

def one_hot_ascii(str): 
    # one-hot encoding of passed string 
    # returns a one-hot tensor with one row per byte and TARGET_SIZE (128) elements per row
    # https://www.wikiwand.com/en/One-hot
    # https://www.tensorflow.org/api_docs/python/tf/one_hot
    return tf.one_hot([ord(c) for c in str], depth=TARGET_SIZE)

class TrainSet: 
    # class for operating with train set for model 

    def __init__(self, file, x_encoder=one_hot_bytes, y_encoder=one_hot_ascii, shuffled=False) -> None:
        '''
        file: path to file containing train set 
        x_encoder: callable to encode input bytes
        y_encoder: callable to encode target string
        shuffled: whether or not to shuffle examples
        '''

        if isinstance(file, str): 
            file = open(file, 'rb')

        self._x_encoder = x_encoder 
        self._y_encoder = y_encoder
        self._file = file 
        self._randomize = False 
        self._max_seek = 0 
        if shuffled: 
            self.shuffle() 

    def __len__(self): 
        # returns number of samples in train set 
        if self._max_seek > 0: 
            return self._max_seek
        
        pos = self._file.tell() 
        self._file.seek(0)
        self._max_seek = len([_ for _ in self._file])
        self._file.seek(pos) 

        return self._max_seek

    def _seek(self): 
        # seeks to either beginning of file or to random position in file 
        # https://www.tutorialspoint.com/python/file_seek.htm
        if self._randomize: 
            self._file.seek(np.random.randint(0, self._max_seek))
        else: 
            self._file.seek(0)

    def shuffle(self): 
        # returns shuffled samples from train set 
        self._randomize = True 
        self._max_seek = len([_ for _ in self._file])
        self._seek() 

    def __iter__(self): 
        # iterator through the train set 
        self._seek() 
        return self 

    def __next__(self): 
        # returns next item in train set 
        if self._randomize: 
            self._seek()

        ln = next(self._file)

        while ln.startswith('#'): 
            ln = next(self._file)

        elms = ln.split('|')
        if len(elms) != 2:
            raise ValueError('Wrong line format in training file: {}'.format(ln))
        
        opcode = bytes([int(elms[0], 16)])
        X = self._x_encoder(opcode)
        y = self._y_encoder(elms[1])
        return X, y
