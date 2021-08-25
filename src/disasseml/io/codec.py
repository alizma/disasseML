from abc import ABCMeta, abstractmethod

import numpy as np 

import tensorflow as tf
import tensorflow.keras as keras 

from disasseml.constants import * 
from disasseml.util import log 

# https://www.youtube.com/watch?v=UDmJGvM-OUw

class Codec(metaclass=ABCMeta):
    @abstractmethod
    def encode(self, seq):
        pass 

    @abstractmethod
    def decode(self, tensor):
        pass

class AsciiCodec(Codec):
    '''
    Encode ASCII as one-hot vectors, or decode one-hot vectors into ASCII.
    '''
    def __init__(self, seq_len, mask_value):
        self._seq_len    = seq_len
        self._mask_value = mask_value

    def encode(self, seq, as_tensor=True):
        '''
        Encodes the contents of an ASCII string as a one-hot matrix.
        :param seq: The ASCII string.
        :param as_tensor: Whether to encode as a tensor or a list.
        :returns: A one-hot encoded matrix representing the ASCII string.
        '''
        # Create indices and insert start and end tokens if not present.
        indices = list(map(ord, seq))
        if seq[0] != START_TOKEN:
            indices.insert(0, ord(START_TOKEN))
        if seq[-1] != STOP_TOKEN:
            indices.append(ord(STOP_TOKEN))
        # Convert to onehot and pad to seq_len.
        onehot = list(keras.utils.to_categorical(indices, num_classes=ASCII_MAX + 1))
        while len(onehot) < self._seq_len:
            onehot.append([0]*len(onehot[0]))
        onehot = np.asarray(onehot, dtype=np.int32)
        # Convert to tensor and/or return.
        if as_tensor:
            return tf.convert_to_tensor(onehot, dtype=tf.int32)
        return onehot

    def decode(self, onehot):
        '''
        Decode a tensor of token indices into an ASCII string tensor.
        :param onehot: A list of one-hot vectors encoding ASCII characters.
        :returns: The decoded string if onehot is a 2D matrix, or a list of such strings if onehot is 3D.
        '''
        # Check type and convert into NumPy array if needed.
        if isinstance(onehot, tf.Tensor):
            onehot = onehot.eval()
        elif isinstance(onehot, list):
            onehot = np.asarray(onehot)
        elif not isinstance(onehot, np.ndarray):
            raise TypeError('Expected Tensor or ndarray, not {}'.format(type(onehot).__name__))
        # Fail if dimensionality is less than 2.
        if len(onehot.shape) < 2:
            raise ValueError('Expected at least two dimensions, got {}'.format(len(onehot.shape)))
        # Map if dimensionality is greater than 2.
        if len(onehot.shape) > 2:
            return list(map(self.decode, onehot))
        # Decode into a string, filtering out one-hot vectors whose elements are all 0.
        return ''.join(map(
            lambda oh: chr(np.argmax(oh)),
            filter(
                lambda oh: np.max(oh) > 0,
                onehot
            )
        ))

class BytesCodec(Codec):
    '''
    Encodes bytes to one-hot, or decodes one hot into bytes.
    '''
    def __init__(self, seq_len, mask_value):
        '''
        Initialise BytesCodec.
        :param seq_len: The sequence length.
        '''
        self._seq_len    = seq_len
        self._mask_value = mask_value

    def encode(self, bs, as_tensor=True):
        '''
        Encode a bytes to one-hot.
        :param bs: Either a bytes object or a string of hex-encoded bytes.
        :param as_tensor: Whether to return a tensor (True) or a list (False).
        :returns: A matrix of the one-hot encoded bytes.
        '''
        if isinstance(bs, str):
            # Convert to bytes from hex string.
            bs = int(bs, 16).to_bytes(
                len(bs)//2, # Every two chars in hexadecimal is one byte.
                BYTEORDER
            )
        if not isinstance(bs, bytes):
            raise TypeError('Expected bytes, not {}'.format(type(bs).__name__))
        # Convert bytes into integers.
        indices = list(bs)
        if len(indices) > self._seq_len:
            log.warning('Expected {} elements or fewer, got {}'.format(self._seq_len, len(indices)))
        # Encode as one-hot and pad to seq_len.
        onehot = list(keras.utils.to_categorical(indices, num_classes=BYTE_MAX + 1))
        while len(onehot) < self._seq_len:
            onehot.append([0]*len(onehot[0]))
        onehot = np.asarray(onehot, dtype=np.int32)
        if as_tensor:
            return tf.convert_to_tensor(onehot, dtype=tf.int32)
        return onehot

    def decode(self, tensor):
        '''
        Decode a one-hot tensor into a bytes object.
        '''
        if not isinstance(tensor, tf.Tensor):
            raise TypeError('Expected Tensor, not {}'.format(type(tensor).__name__))
        # Compute the argmax of each one-hot vector, filtering out those whose elements are all zero.
        indices = map(
            np.argmax,
            filter(
                lambda oh: np.max(oh) > 0,
                tensor.eval()
            )
        )
        return bytes(indices)

'''
def bytes_to_one_hot(bts): 
    # one-hot encoding of passed bytes 
    # returns a one-hot tensor with one row per byte and INPUT_SIZE (256) elements per row
    # https://www.wikiwand.com/en/One-hot
    # https://www.tensorflow.org/api_docs/python/tf/one_hot
    if not isinstance(bts, bytes): 
        raise TypeError('Expected bytes, received {}'.format(type(bts).__name__))
    return tf.one_hot(list(bts), depth=INPUT_SIZE)

def ascii_to_one_hot(s): 
    # one-hot encoding of passed string 
    # returns a one-hot tensor with one row per byte and TARGET_SIZE (128) elements per row
    # https://www.wikiwand.com/en/One-hot
    # https://www.tensorflow.org/api_docs/python/tf/one_hot
    if not isinstance(s, str):
        raise TypeError('Expected str, received {}'.format(type(s).__name__))
    return tf.one_hot([ord(c) for c in str], depth=TARGET_SIZE)

def one_hot_to_bytes(tens): 
    # decodes passed Tensor tens to bytes 
    if not isinstance(tens, tf.Tensor): 
        raise TypeError('Expected Tensor, received {}'.format(type(tens).__name__))
    if len(tens.shape) != 2: 
        raise ValueError('Expected 2D Tensor, received {}D'.format(len(tens.shape())))
    if tens.shape[1] != TARGET_SIZE:
        raise ValueError('Expected tensor size of dim 2 to be {}, received {}'.format(INPUT_SIZE, tens.shape[1]))

    return bytes([row.find(max(row)) for row in tens])

def one_hot_to_ascii(tens): 
    # decodes passed Tensor tens to ASCII 
    if not isinstance(tens, tf.Tensor): 
        raise TypeError('Expected Tensor, received {}'.format(type(tens).__name__))
    if len(tens.shape) != 2: 
        raise ValueError('Expected 2D Tensor, received {}D'.format(len(tens.shape())))
    if tens.shape[1] != TARGET_SIZE:
        raise ValueError('Expected tensor size of dim 2 to be {}, received {}'.format(INPUT_SIZE, tens.shape[1]))

    return ''.join([chr(row.find(max(row))) for row in tens])
'''