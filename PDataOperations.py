import logging
import numpy as np
import pandas as pd
import scipy
import PFileManager as PFM

logging.basicConfig(filename="./logs/PlasmaDataset_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class PDataRoutines:

    def dataStats(self, data):
        """
        Returns the mean and standard deviation of inputted array using scipy
        """
        cal_data = np.array([]) #flatten all feature data to a numpy array
        
        for i in data:
            cal_data = np.append(cal_data, i)
            
        data_params = scipy.stats.describe(cal_data)
        data_mean = data_params.mean
        data_sd = np.sqrt(data_params.variance)
        
        return data_mean, data_sd

    def datasetStats(self, dataset):
        """
        Returns the compiled mean and standard deviation of a dataset
        Typically use train dataset for machine learning purposes
        Takes the aggregate sum of all files across a feature
        Stores feature statistics in "stats" in self.data as a tuple (u, o)
        Consider organizing stats by dataset instead of one for train/test/val
        """
        data_stats = {}
        
        logging.info("Dataset size: %s", len(dataset['raw']) - 1)
        for feature in dataset['raw']:
            if feature == 'label':
                continue
            logging.info(feature)
            mean, sd = self.dataStats(dataset['raw'][feature])
            data_stats[feature] = (mean, sd)

        return data_stats

    def labelHDF5Data(self, filelist:list):
        """
        Returns a numpy array of labels (0/1) for the inputted hdf5 file list
        Performs labeling using PFileManager's isDisruptiveHDF5 method
        """
        labels = np.array([])
        try:
            for file in filelist:
                labels = np.append(labels, PFM.isDisruptiveHDF5(file))
        except Exception as e:
            logging.error("Failed to label data: %s", filelist)
        
        return labels

    def normalizeData(self, feature, stats):
        """
        Uses mean and standard deviation of data feature to apply normalization
        stats is a tuple (mean, sd)
        """
        (mean, sd) = stats
        norm_feature = [[(val - mean)/sd for val in feat] for feat in feature]
        return norm_feature

    def normalizeDataset(self, dataset, statset):
        """
        Takes inputted stats dataset for features to normalize dataset
        """
        norm_data = {}
        for feat in dataset['raw']:
            if feat == 'label':
                continue
            norm_feat = pd.DataFrame(self.normalizeData(dataset['raw'][feat], statset[feat]))
            norm_data = pd.concat([norm_data, norm_feat])
            
        pd.concat([norm_data, pd.DataFrame(dataset['raw']['label'])])
        logging.info("Normalized %s", dataset)
        
        return norm_data

    def splitData(self, split, files):
        """
        Routine to split input files into train/test/val data according to split
        If split does not sum to 1, extra data will go to validation
        """
        if np.sum(split) is not 1:
            logging.warn("Set split does not sum to 1!")
        count = len(files)
        train_size = round(count*split[0])
        test_size = round(count*split[1])
        val_size = count - (train_size + test_size)

        np.random.shuffle(files)
        train_files, test_files, val_files = files[:train_size], files[train_size: train_size + test_size], files[train_size+test_size:]
        
        logging.info(f"{train_size} training shots {split[0]*100} %")
        logging.info(f"{test_size} testing shots {split[1]*100} %")
        logging.info(f"{val_size} validation shots {split[2]*100} %")

        return train_files, test_files, val_files