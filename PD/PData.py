import gc
import logging
import numpy as np
import pandas as pd
import os
import shutil
import time
from PD import PFileManager as PFM

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
    
    def dataNumpyToList(self, data):
        """
        Converts data in component columnn from numpy array to list
        """
        converted_data = []
        try:
            for row in data:
                if isinstance(row, list):
                    if isinstance(row[0], object):
                        converted_row = [float(np_data) for np_data in row]
                        converted_data.append(converted_row)
                    else:
                        converted_data.append(row)
                else:
                    converted_data.append(row)
        except Exception as e:
            logging.error("Failed to convert to list: %s", e)
        return converted_data
    
    def datasetNumpyToList(self, dataset):
        """
        Converts all dataset features to lists
        """
        list_component = pd.DataFrame([])
        for feature in dataset:
            list_component = pd.concat([
                list_component,
                pd.DataFrame({feature: self.dataNumpyToList(dataset[feature])})
            ])
        return list_component
    
    def deleteDataset(self, dataset:str):
        """
        Deletes dataset from self.data and physical location
        """
        try:
            if dataset in self.data:
                del self.data[dataset]
            if dataset in self.data_dirs:
                if os.path.exists(self.data_dirs[dataset]):
                    PFM.deleteSubdirs(self.data_dirs[dataset])
                del self.data_dirs[dataset]
            gc.collect()
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

    def saveDatasetCSV(self, dataset: str, file_name: str = None, directory: str = None) -> dict:
        """
        Save dataset components as CSV in their org_directory folder.
        
        Args:
            dataset (str): The dataset key (used in self.data and self.data_dirs).
            file_name (str, optional): Base file name (without .csv). If None, appends timestamp.
            directory (str, optional): Directory to save files. Defaults to self.data_dirs[dataset].
        
        Returns:
            dict: A dictionary mapping each dataset component to the full path of its saved CSV file.
        """
        
        # Default directory: "dataset" subfolder in self.data_dirs
        if directory is None:
            directory = self.data_dirs[dataset]
        
        # Strip ".csv" if it's accidentally in file_name
        if file_name and "csv" in file_name.lower():
            file_name = file_name.split(".")[0]
        
        # Build a dictionary to track saved paths
        saved_paths = {}

        try:
            for component in self.data[dataset]:
                # Use timestamp if no file_name is given; note '%S' for seconds (uppercase S)
                time_part = time.strftime('%Y-%m-%d--%H-%M-%S') if file_name is None else file_name
                
                # Construct the CSV filename
                component_csv_name = f"{dataset}-{component}-{time_part}.csv"
                component_csv_path = os.path.join(directory, component_csv_name)
                
                #converts all dataset component features to lists (from numpy arrays)
                export_component = self.datasetNumpyToList(self.data[dataset][component])

                # Write CSV with explicit comma separator and UTF-8 BOM (helps Excel on Windows)
                self.data[dataset][component].to_csv(
                    component_csv_path,
                    index=False,
                    sep=',',
                    encoding='utf-8-sig'
                )
            
                # Save the path in our tracking dict
                saved_paths[component] = component_csv_path

        except Exception as e:
            logging.error("Failed to save %s CSV: %s", dataset, e)
            # Optionally re-raise or return an empty dict
            # raise e
        
        return saved_paths

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
