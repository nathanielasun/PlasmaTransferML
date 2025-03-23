import pandas as pd
import numpy as np
import logging
import os
import shutil

class PData:
    
    def __init__(self, org_directory, h5_source):
        
        self.data = {} #dictionary for all named datasets | added to by addDataset | all data takes a pandas DataFrame format
        
        self.data_config = {
                            'data_frac'  : 1, #Fraction of dataset to be loaded for dataset creation (default 1 = whole)
                            #'split' : [ 0, 0, 0 ], #data split train/test/val
                            'features' : [] # array of all hdf5 features noted
                           }
        
        self.data_dirs = {
                            'hdf5_files' : [],#data location array for hdf5 files
                            'hdf5_source': "", #root directory of hdf5 unorganized file source
                            'org_dir' : "",  #root directory of organized database
                         }  #dictionary for handling all data directories (esp. train/test/val) added to by setDirectory
        
        os.makedirs(org_directory, exist_ok=True)
        
        self.data_dirs['hdf5_source'] = h5_source #provide a source directory for hdf5 files
        logging.info("Initialized with hdf5 source: %s", h5_source)
        
        self.data_dirs['org_dir'] = org_directory #set the organized data directory
        logging.info("Initialized with organized data directory: %s", org_directory)
        
    def addDataset(self, name:str, components=[], RESET=False):
        """
        Adds a new dataset with named dataframe components as per components
        I.e. 'name' : { 'norm' : pd.DataFrame() , 'raw' : pd.DataFrame()...} where 'norm' and 'raw' are in "components" list
        All data should be stored as a pandas dataframe
        """
        if RESET:
            self.deleteDataset(name)
        try:
            self.data[name] = {}
            for component in components:
                self.data[name][component] = pd.DataFrame()
            self.addDirectory(name)
            logging.info("Added new dataset %s with components %s", name, components)
        except Exception as e:
            logging.error("Failed to add new dataset due to %s", e)

    def addDirectory(self, dir_name:str):
        """
        Routine to initialize subdirectories and add new directories
        Appends a new directory label and location to data_config
        """
        target_dir = os.path.join(self.data_dirs['org_dir'], dir_name)
        try:
            os.makedirs(target_dir, exist_ok=True)
            self.data_dirs[dir_name] = target_dir
        except Exception as e:
            logging.error("Failed to make directory for %s: %s", target_dir, e)
    
    def deleteDataset(self, dataset:str):
        """
        Deletes dataset from self.data and physical location
        """
        try:
            if dataset in self.data:
                del self.data[dataset]
            if dataset in self.data_dirs:
                if os.path.exists(self.data_dirs[dataset]):
                    self.deleteSubdirs(self.data_dirs[dataset])
                del self.data_dirs[dataset]
            logging.info("Removed dataset %s", dataset)
        except Exception as e:
            logging.error("Failed to remove dataset %s: %s", dataset, e)
        
    def deleteSubdirs(self, dir_name:str):
        """
        Deletes all subdirectories within the given root directory.
        """
        for item in os.listdir(dir_name):
            try:
                item_path = os.path.join(dir_name, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Delete directory and its contents
                    logging.info("Deleted directory: %s", item_path)
            except Exception as e:
                logging.error("Failed to remove file %s: %s", item, e)
                
    def exportDataComponent(self, dataset, component):
        """
        Exports the data from a dataset component in pandas dataframe format
        """
        return self.data[dataset][component]
            
    def setDataFrac(self, new_data_frac):
        """
        Routine to set fraction of data to load from data component directories when running experiments and training
        """
        self.data_config['data_frac'] = new_data_frac
        logging.info("Set data fraction to: %s", new_data_frac)
        
    def setOrgDir(self, new_org_dir):
        """
        Routine to update organized data root directory if needed
        """
        self.data_dirs['org_dir'] = new_org_dir
        logging.info("Set new organized data directory to: %s", new_org_dir)

    def updateDatasetComponent(self, dataset:str, component:str, data):
        """
        Appends new data to given dataset raw data keeping set index
        Assumes order of data for feature(s) stays consistent with labeling
        i.e. adds {'featureN' : [[x1, x2, ... xn], ...] } to raw data
        """
        try:
            self.data[dataset][component] = pd.concat([self.data[dataset][component], pd.DataFrame(data)], ignore_index=True)
            logging.info("Updated %s %s data", dataset, component) 
        except Exception as e:
            logging.error("Failed to update %s %s: %s", dataset, component, e)
    
    def setDatasetComponent(self, dataset:str, component:str, data):
        """
        Replaces dataset component with new component
        Assumes dataframe/dictionary format i.e. for stats: {'feature1' : (u1, o1), 'feature2' : (u2, o2), ... }
        I.e. for raw data: {'feature1' : [[x1, x2, ..., xn], [y1,y2, ..., yn], ...], 'feature2'...}
        """
        try:
            self.data[dataset][component] = pd.DataFrame(data)
            logging.info("Updated dataset %s component %s", dataset, component)
        except Exception as e:
            logging.error("Failed to update data: %s", e)

    def resetDatasetComponents(self, dataset:str, components:list=None):
        """
        Resets components listed to blank pandas dataframes in given dataset
        I.e. resets ['norm', 'raw'] in train to blank pandas dataframe
        """
        try:
            for component in components:
                self.data[dataset][component] = pd.DataFrame()
        except Exception as e:
            logging.error("Failed to reset components %s: %s", components, e)
        logging.info("Reset '%s' %s data", dataset, components) 
