import numpy as np
import pandas as pd
import torch
import logging

logging.basicConfig(filename="./logs/PlasmaModel_logs.txt", level=logging.DEBUG,
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

def maxLenFeature(data:'DataFrame')->int:
    """
    Returns length of longest array in first feature
    Takes norm/raw data not full dataset object
    """
    try:
        feats = [i for i in data]
        array_index = data[feats[0]].apply(len).idxmax()
        array_len = len(data[feats[0]][array_index])
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
    Converts a DataFrame to a tensor with dimensions (samples, seqLen, features).
    Each cell in the DataFrame should be a sequence (e.g., list, array).
    Shorter sequences are padded to match seqLen and longer ones truncated.
    """
    if seqLen is None:
        seqLen = maxLenFeature(data)
    
    sample_tensors = []
    # Iterate over each row (sample) in the DataFrame
    for i in range(len(data)):
        feature_tensors = []
        # Iterate over each feature/column
        for feat in data.columns:
            # Create a tensor from the sequence stored in each cell
            array_tensor = torch.tensor(data[feat].iloc[i], dtype=torch.float32)
            array_tensor = resizeTensor(array_tensor, seqLen)
            feature_tensors.append(array_tensor)
        # Stack feature tensors: shape becomes (features, seqLen)
        sample_tensor = torch.stack(feature_tensors)
        # Transpose so that shape becomes (seqLen, features)
        sample_tensor = sample_tensor.transpose(0, 1)
        sample_tensors.append(sample_tensor)
    
    # Stack all samples: final shape (samples, seqLen, features)
    data_tensor = torch.stack(sample_tensors)
    return data_tensor
