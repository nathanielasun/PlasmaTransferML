import h5py
import inspect
import logging
import numpy as np
import pandas as pd
from PData import PData
import PDataOperations as PDO
import PFileManager as PFM

logging.basicConfig(filename="./logs/PlasmaDataset_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class PlasmaDataset:
    """
    Creates and manages a dataset of tokamak data sourced from HDF5 files
    Organizes into train/test/val datasets and directories and normalizes data
    Builds off of selected features
    """
    def __init__(self, org_directory, h5_source):
        
        self.Dataset = PData(org_directory, h5_source)
        self.org_dir = org_directory #organized data directory
        self.h5_source = h5_source #hdf5 source data directory
        logging.info("Initialized PlasmaDataset with org_dir=%s and h5_source=%s", self.org_dir, self.h5_source)
        
    def initialize(self, reset:bool=True):
        """
        Routine to create an plasma ML train/test/val dataset with set data split and fraction
        Keeping reset true will replace the existing datasets train/test/val
        """
        #Create train/test/val datasets with raw/norm/file data components
        try:  
            self.Dataset.addDataset("train", ['raw', 'norm', 'files'], RESET=reset)
            self.Dataset.addDataset("test", ['raw', 'norm', 'files'], RESET=reset)
            self.Dataset.addDataset("val", ['raw', 'norm', 'files'], RESET=reset)
            self.Dataset.addDataset("stats", ['stats'], RESET=reset)
            logging.info("Successfully created train/test/val datasets")
        except Exception as e:
            logging.error("Failed to create train/test/val datasets: %s", e)
        
    def source(self, data_split:list, features:list = None, data_frac:float = 1):
        """
        Acquires raw data features from file list for train/test/val datasets at data fraction
        i.e. for train data, sources features from hdf5 files in train directory
        Sources features from given feature list. None indicates all hdf5 features by default.
        Note: currently wipes existing data every time used - use only for changing features for training
        """
        #collect hdf5 file locations, assign splits, and fraction data
        try:
            hdf5_files = PFM.sourceHDF5(hdf5_path=self.h5_source)
            train_files, test_files, val_files = PDO.splitData(split=data_split, files=hdf5_files, randomize=True)
            train_files = train_files[:round(data_frac*len(train_files))]
            test_files = test_files[:round(data_frac*len(test_files))]
            val_files = val_files[:round(data_frac*len(val_files))]
            logging.info("Successfully sourced and split HDF5 files")
        except Exception as e:
            logging.error("Failed to source and split HDF5 files: %s", e)
            return
        #label files and create file dataframe
        try:
            train_labels = PDO.labelHDF5Data(train_files)
            test_labels = PDO.labelHDF5Data(test_files)
            val_labels = PDO.labelHDF5Data(val_files)
            file_info = {
                 'train':list(zip(train_files, train_labels)),
                 'test':list(zip(test_files, test_labels)), 
                 'val':list(zip(val_files, val_labels))
            }
        except Exception as e:
            logging.error("Failed to label data: %s", e)
        #update train/test/val datasets with their files (and file labels)
        try:
            self.Dataset.updateDatasetComponent('train', 'files', file_info['train'])
            self.Dataset.updateDatasetComponent('test', 'files', file_info['test'])
            self.Dataset.updateDatasetComponent('val', 'files', file_info['val'])
            logging.info("Successfully assigned HDF5 files to train/test/val")
        except Exception as e:
            logging.error("Failed to assign HDF5 files to train/test/val: %s", e)
            return
        #source raw data into train/test/val
        try:
            train_data = PFM.sourceHDF5Data(file_info['train'], features)
            test_data = PFM.sourceHDF5Data(file_info['test'], features)
            val_data = PFM.sourceHDF5Data(file_info['val'], features)
            logging.info("Successfully sourced train/test/val data")
        except Exception as e:
            logging.error("Failed to source train/test/val data: %s", e)
        #assign train/test/val raw data
        try:
            self.Dataset.updateDatasetComponent('train', 'raw', train_data)
            self.Dataset.updateDatasetComponent('test', 'raw', test_data)
            self.Dataset.updateDatasetComponent('val', 'raw', val_data)
        except Exception as e:
            logging.error("Failed to assign train/test/val/data: %s", e)
        #organize file directories under org_dir
        try:
            PFM.organizeData(files=file_info, org_dir=self.org_dir, label_names=['disruptive', 'nondisruptive'])
        except Exception as e:
            logging.error("Failed to organize train/test/val directories: %s", e)
        
    def calcStats(self):
        """
        Calculates dataset stats from train dataset and stores in "stats" w/ features
        """
        datastats = PDO.datasetStats(self.Dataset.exportDataComponent('train', 'raw'))
        self.Dataset.updateDatasetComponent('stats', 'stats', datastats)

    def normalize(self):
        """
        Performs normalization on train/test/val using data stats
        """
        try:
            stats = self.Dataset.exportDataComponent("stats", "stats")
            train_norm = PDO.normalizeDataset(self.Dataset.exportDataComponent('train', 'raw'), stats)
            test_norm = PDO.normalizeDataset(self.Dataset.exportDataComponent('test', 'raw'), stats)
            val_norm = PDO.normalizeDataset(self.Dataset.exportDataComponent('val', 'raw'), stats)
            logging.info("Successfully normalized train/test/val")
        except Exception as e:
            logging.error("Failed to normalize train/test/val: %s", e)

        try:
            self.Dataset.updateDatasetComponent('train', 'norm', train_norm)
            self.Dataset.updateDatasetComponent('test', 'norm', test_norm)
            self.Dataset.updateDatasetComponent('val', 'norm', val_norm)
            logging.info("Successfully updated normalized train/test/val")
        except Exception as e:
            logging.error("Failed to update normalized train/test/val: %s", e)

    def preview(self, dataset:str=None):
        """
        Shows a preview of train/test/val
        """
        if dataset is None:
            self.Dataset.describeDataset('train')
            self.Dataset.describeDataset('test')
            self.Dataset.describeDataset('val')
            self.Dataset.describeDataset('stats')
        else:
            try:
                self.Dataset.describeDataset(dataset)
            except Exception as e:
                logging.error('Failed to access preview of %s: %s', dataset, e)
    
                
        