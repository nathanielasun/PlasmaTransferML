import numpy as np
import pandas as pd
import torch
import logging

logging.basicConfig(filename="../logs/PlasmaModel_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def set_seed(seed):
    """
    Set a seed to make randomization deterministic
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For GPU (if you're using one)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    np.random.seed(seed)  # For numpy random numbers
    random.seed(seed)  # For Python's built-in random module
    torch.backends.cudnn.deterministic = True  # Make cuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Ensure consistency

def maxLenFeature(data:'DataFrame'):
    """
    Returns length of longest array in first feature
    Takes norm/raw data not full dataset object
    """
    try:
        feats = [i for i in data]
        array_index = data[feats[0]].apply(len).idxmax()
        array_len = data[feats[0]][array_index]
    except Exception as e:
        logging.error("Failed to return max length: %s", e)
    return array_len

def resizeTensor(tensor, seqLen):
    tensor_size = tensor.size()[0]
    if tensor_size > seqLen:
        #cut longer arrays to seqLen
        tensor = tensor[:seqLen]
    else:
        #pad shorter arrays to seqLen
        tensor = torch.nn.functional.pad(tensor, (0, seqLen - tensor_size))
    return tensor

def tensorData(data:'DataFrame', seqLen=None) -> "Tensor":
    """
    Converts dataframe to tensor
    Takes norm/raw data not full dataset object
    (samples, seqLen, features) i.e. (200, 400, 14)
    By default seqLen will be the length of the longest array in features
    Shorter arrays are padded to match seqLen and longer arrays are shortened
    """
    if seqLen is None:
        seqLen = maxLenFeature(data)
    
    data_tensor = []
    for i in range(data.shape()[0]):
        sample_tensor = []
        for feat in data:
            array_tensor = torch.tensor(data[feat][i])
            array_tensor = resizeTensor(array_tensor, seqLen)
            sample_tensor.append(array_tensor)
        #stack all feature tensors into a 2D sample tensor
        sample_tensor = torch.stack(sample_tensor)
        data_tensor.append(sample_tensor)
    #stack all sample tensors into a 3D tensor
    data_tensor = torch.stack(data_tensor)

    return data_tensor

    