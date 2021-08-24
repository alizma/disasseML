
import tensorflow as tf

BYTE_MAX = 0xFF
ASCII_MAX = 0x7F

INPUT_SIZE = BYTE_MAX + 1
TARGET_SIZE = ASCII_MAX + 1

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