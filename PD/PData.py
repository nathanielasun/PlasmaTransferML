import pandas as pd
import numpy as np
import logging
import os
import shutil
import time

logging.basicConfig(filename="./logs/PlasmaDataset_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class PData:
    
    def __init__(self, org_directory):
        
        self.data = {} #dictionary for all named datasets | added to by addDataset | all data takes a pandas DataFrame format
        
        self.data_dirs = {
                            'org_dir' : "",  #root directory of organized database
                         }  #dictionary for handling all data directories (esp. train/test/val) added to by setDirectory
        
        os.makedirs(org_directory, exist_ok=True)        
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
                self.data[name][component] = pd.DataFrame(dtype=object)
            self.addDirectory(name)
            logging.info("Added new dataset %s with components %s", name, components)
        except Exception as e:
            logging.error("Failed to add new dataset due to %s", e)

    def addDirectory(self, dir_name:str):
        """
        Routine to initialize subdirectories and add new directories
        Appends a new directory label and location to data_dirs
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

    def describeDataset(self, dataset:str):
        """
        Gives basic information about specified dataset
        Prints dataset features (if true), lists, components, and prints dataset size
        """
        print(self.data[dataset])
    
    def exportDataComponent(self, dataset, component:str) -> "dataset components dataframe":
        """
        Exports the data from a dataset component in pandas dataframe format
        """
        try:
            export = self.data[dataset][component]
        except Exception as e:
            logging.error("Failed to export %s %s: %s", dataset)
        return self.data[dataset][component]

    def saveDatasetCSV(self, dataset:str, file_name:str, directory:str=None):
        """
        Save dataset components as CSV in their org_directory folder
        """
        #set directory to "dataset" in org_directory by default
        if directory is None:
            directory = self.data_dirs[dataset]
        #remove csv ending if specified by accident
        if "csv" in file_name.split("."):
            file_name = file_name.split(".")[0]
        #export each dataset component as a CSV
        try:
            for component in self.data[dataset]:
                component_csv_name = f"{file_name}-{dataset}-{component}-{time.strftime('%Y-%m-%d--%H-%M-%s')}.csv"
                component_csv_path = os.path.join(directory, component_csv_name)
                self.data[dataset][component].to_csv(component_csv_path, index=False)
        except Exception as e:
            logging.error("Failed to save %s csv: %s", dataset, e)
        
    def setOrgDir(self, new_org_dir):
        """
        Routine to update organized data root directory if needed
        """
        self.data_dirs['org_dir'] = new_org_dir
        logging.info("Set new organized data directory to: %s", new_org_dir)

    def updateDatasetComponent(self, dataset:str, component:str, data:dict):
        """
        Appends new data to given dataset component keeping set index
        Assumes order of data for feature(s) stays consistent with labeling
        i.e. adds {'feat1' : [[x1, x2, ... xn]], 'feat2':[[x1,x2,...xn]],... } to raw data
        Assumes same features in input as in dataset
        """
        try:
            self.data[dataset][component] = pd.concat([self.data[dataset][component], pd.DataFrame(data, dtype=object)], ignore_index=True)
            logging.info("Updated %s %s data", dataset, component) 
        except Exception as e:
            logging.error("Failed to update %s %s: %s", dataset, component, e)
    
    def setDatasetComponent(self, dataset:str, component:str, data):
        """
        Sets dataset component with new component
        Assumes dataframe/dictionary format i.e. for stats: {'feature1' : (u1, o1), 'feature2' : (u2, o2), ... }
        I.e. for raw data: {'feature1' : [[x1, x2, ..., xn], [y1,y2, ..., yn], ...], 'feature2'...}
        """
        try:
            self.data[dataset][component] = pd.DataFrame(data, dtype=object)
            logging.info("Reset dataset %s component %s", dataset, component)
        except Exception as e:
            logging.error("Failed to update data: %s", e)

    def resetDatasetComponents(self, dataset:str, components:list=None):
        """
        Resets components listed to blank pandas dataframes in given dataset
        I.e. resets ['norm', 'raw'] in train to blank pandas dataframe
        """
        try:
            for component in components:
                self.data[dataset][component] = pd.DataFrame(dtype=object)
        except Exception as e:
            logging.error("Failed to reset components %s: %s", components, e)
        logging.info("Reset '%s' %s data", dataset, components) 
