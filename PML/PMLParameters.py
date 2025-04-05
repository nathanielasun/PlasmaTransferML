import numpy as np
import itertools
import logging

logging.basicConfig(filename="../logs/PlasmaModel_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class PMLParameters:
    def __init__(self, ranges:dict, include:dict):
        self.include = include #static features to keep in parameter dicts
        self.parameter_list = []
        self.ranges = ranges #dictionary of parameter ranges to perform search on

    def clearParameterList(self):
        """
        Clears parameter list
        """
        self.parameter_list = []

    def getParameterSet(self, amount=None):
        """
        Returns set amount of sets from parameter set in self.parameter_list
        """
        parameter_set = []
        if amount is None:
            amount = len(self.parameter_list)
        for _ in range(amount):
            parameter_set.append(self.parameter_list[0])
            self.parameter_list.remove(self.parameter_list[0])
            
        return parameter_set
    
    def gridStartingList(self, ranges):
        """
        Makes a range starting list for parameter ranges
        I.e. for lr = [0.01, 0.02] and dropout = [[0.1,0.2], [0.2, 0.4]]
        - Creates list [0.01, [0.1, 0.2]]
        """
        starting_list = []
        for parameter_range in ranges:
            if isinstance(ranges[parameter_range][0],list):
                #if range is 2D list (i.e. NN layer range list), find each subrange start
                subrange_list = []
                for subrange in ranges[parameter_range]:
                    subrange_list.append(subrange[0])
                starting_list.append(subrange_list)
            else:
                #if range is 1D list (i.e. not a NN layer range list), append starting range
                starting_list.append(ranges[parameter_range][0])
                
        return starting_list

    def gridIncrementList(self, ranges:dict, division:int):
        """
        Makes a list of increments per parameter range
        Assumes ranges is passed as a dict
        I.e. for lr = [0.01, 0.02] and division = 8, gives [0.00125]
        """
        increment_list = []
        for parameter_range in ranges:
            if isinstance(ranges[parameter_range][0], list):
                #if range is 2D list, ensure each subrange increment is stored
                subrange_inc = []
                for subrange in ranges[parameter_range]:
                    subrange_inc.append(np.abs(subrange[1]-subrange[0])/division)
                increment_list.append(subrange_inc)
            else:
                #otherwise if list is regular range, immediately calculuate division
                increment_list.append(np.abs(ranges[parameter_range][1]-ranges[parameter_range][0])/division)

        return increment_list

    def makeGridParameterSet(self, division, offset=0):
        """
        Makes a gridded set over each parameter
        Forms division^n sets where n = parameter count (including subranges for NN layers)
        I.e. for division = 4 and 4 total parameters (let's say lr and [lstm_layer1, lstm_layer2, lstm_layer3])
                - Creates 256 sets!
        Uses a start list and increment list to perform search
        Offset controls how many increments offset to begin grid search from
        """
        start_list = self.gridStartingList(self.ranges)
        inc_list = self.gridIncrementList(self.ranges, division)
        ranges_per_feat = []
        #append ranges of features to use cartesian product on
        for start, inc in zip(start_list, inc_list):
            feat_range_info = [start + (i + offset)*inc for i in range(division)]
            ranges_per_feat.append(feat_range_info)

        parameter_combinations = list(itertools.product(*ranges_per_feat))
        
        for combination in parameter_combinations:
            try:
                parameter_dict = self.include.copy()
                for value, parameter in zip(combination, [param for param in self.ranges]):
                    parameter_dict[parameter] = value
                self.parameter_list.append(parameter_dict)
            except Exception as e:
                logging.error("Failed to append parameter dict: %s", e)
        
    def makeRandomParameterSet(self, count):
        """
        Makes set "count" number of randomized hyperparameter dicts
        I.e. for count 8, makes 8 random hyperparameter dicts and stores in self.parameter_list
        """
        for _ in range(count):
            try:
                parameter_dict = self.include.copy()
                for parameter in self.ranges:
                    parameter_dict[parameter] = self.randomizeParameter(self.ranges[parameter])
                self.parameter_list.append(parameter_dict)
            except Exception as e:
                logging.error("Failed to create/append randomized parameter dict: %s", e)
        
    def randomizeParameter(self, parameter):
        """
        Randomize parameter given range list
        i.e. [0.03, 0.05] -> random value between 0.03 and 0.05
        or 'lstm_layers' : [[30, 40], [100, 200], [50, 100]] -> [randint(30,40), randint(100, 200), randint(50, 100)]
        """
        rand_parameter = None
        
        if isinstance(parameter[0], list):
            try:
                rand_parameter = []
                for subrange in parameter:
                    if isinstance(subrange[0], int):
                        rand_parameter.append(np.random.randint(subrange[0], subrange[1]))
                    elif isinstance(subrange[0], float):
                        rand_parameter.append(np.random.uniform(subrange[0], subrange[1]))
                    else:
                        logging.error("PML-PARAMETERS - Unsupported range variable type: %s", str(subrange))
            except Exception as e:
                logging.error("Failed to randomize parameter: %s", e)   
                
        elif isinstance(parameter[0], int):
            try:
                rand_parameter = np.random.randint(parameter[0], parameter[1])
            except Exception as e:
                logging.error("Failed to randomize parameter: %s", e)
        elif isinstance(parameter[0], float):
            try:
                rand_parameter = np.random.uniform(parameter[0], parameter[1])
            except Exception as e:
                logging.error("Failed to randomize float parameter: %s", e)
        else:
            logging.error("PML-PARAMETERS - Unsupported range variable type: %s", str(parameter))
        
        return rand_parameter
        
