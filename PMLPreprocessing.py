import numpy as np
import pandas as pd
import torch

class PMLPreprocessing:
    def __init__(self):
        
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

    def tensorData(self, data:"pd dataframe") -> "torch tensor":

    def padData(self, data:"pd dataframe") -> "padded pd dataframe":
        """
        