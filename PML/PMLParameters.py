import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import logging

logging.basicConfig(filename="../logs/PlasmaModel_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class PMLParameters:
    def __init__(self, mode='random', ranges:dict, include:dict):
        self.include = include #static features to keep in parameter dicts
        self.mode = mode #grid/random hyperparameter search mode
        self.parameter_list = []
        self.ranges = ranges #dictionary of parameter ranges to perform search on
    
    def makeRandomParameterSet(self, count):
        """
        Makes set "count" number of randomized hyperparameter dicts
        I.e. for count 8, makes 8 random hyperparameter dicts and stores in self.parameter_list
        """
        for i in range(count):
            parameters = self.include
            for parameter in ranges:
                self.parameter_list[parameter] = self.randomizeParameter(self.ranges[parameter])
        logging.info("Randomized %s parameter sets", str(count))

    def makeGriddedParameterSet(self, division):
        """
        Makes set of hyperparameter dicts with each parameter varied by its range/division
        I.e. for two parameters and division=4, 16 parameter dicts will be created, gridded over the 2D hyperparameter space
        All sets appended to self.parameter_list
        """
        
        
    def randomizeParameter(self, parameter):
        """
        Randomize parameter given range list
        i.e. [0.03, 0.05] -> random value between 0.03 and 0.05
        or 'lstm_layers' : [[30, 40], [100, 200], [50, 100]] -> [randInt(30,40), randInt(100, 200), randInt(50, 100)]
        """
        rand_parameter = None
        if type(parameter[0]) == list:
            try:
                rand_parameter = []
                for subrange in parameter:
                    rand_parameter.append(np.random.randInt(subrange[0], subrange[1]))
            except Exception as e:
                logging.error("Failed to randomize parameter: %s", e)   
        else:
            try:
                rand_parameter = np.random.randInt(parameter[0], parameter[1])
            except Exception as e:
                logging.error("Failed to randomize parameter: %s", e)

        return rand_parameter
        
            