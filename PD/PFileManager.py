import h5py
import os
import numpy as np
import pandas as pd
import logging
import shutil

logging.basicConfig(filename="./logs/PlasmaDataset_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def isDisruptiveHDF5(hdf5_loc:str) -> bool:
    """
    Defines a routine to classify HDF5 file as disruptive/non-disruptive using metadata
    1 - disruptive
    """
    with h5py.File(hdf5_loc) as f:
        label = f['meta/IsDisrupt'][()]
        
    return 1 if label else 0
    
def deleteSubdirs(dir_name:str):
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
            
def getHDF5Feats(file_dir:str) -> dict:
    """
    Returns a dictionary of HDF5 features by feature type
    I.e. {'data' : [features], 'meta' : [features]}
    """
    feat_dict = {}
    with h5py.File(file_dir, 'r') as f:
        for feat_type in f:
            try:
                feat_list = [str(feat) for feat in feat_type]
            except Exception as e:
                logging.error(f"Failed to create feature list for %s: %s", feat_type, e)
            feat_dict[feat_type] = feat_list
    return feat_dict
    
def organizeData(files:dict, org_dir:str, label_names:list=[0, 1]):
    """
    Pulls files from a dict with lists with zipped files and labels into a directory with labels.
    Each dataset filelist header (i.e. "train/test/val") will be used as the subdirectory folder name.
    If labels have names (e.g. nondisruptive/disruptive), these should be noted in "label_names",
    otherwise 0/1 are used by default.
    Each file within a dataset filelist is a tuple (filepath, label).
    By default, creates dataset subdirectories for binary labels: each of train/test/val will have disruptive/nondisruptive folders.
    """
    
    for fileset in files:
        subdir = os.path.join(org_dir, fileset)
        # Ensure label directory names are strings
        label0_dir = os.path.join(subdir, str(label_names[0]))
        label1_dir = os.path.join(subdir, str(label_names[1]))
        
        # Create subdirectories
        os.makedirs(label0_dir, exist_ok=True)
        os.makedirs(label1_dir, exist_ok=True)
 
        for file in files[fileset]:
            file_name = file[0]
            file_label = file[1]
            try:
                # Extract just the base name without extension
                base_name, _ = os.path.splitext(os.path.basename(file_name))
                # Build new file name with .hdf5 extension
                new_file_name = base_name + '.hdf5'
                # Choose destination directory based on label (truthy value selects label1_dir)
                dest_dir = label1_dir if file_label else label0_dir
                new_file_path = os.path.join(dest_dir, new_file_name)
                shutil.copy(file_name, new_file_path)
            except Exception as e:
                logging.error("Skipping file %s: %s", file_name, e)
        logging.info("Exported %d hdf5 files to %s", len(files[fileset]), subdir)

def sourceHDF5(hdf5_path:str) -> list:
    """
    Routine to source hdf5 file names for the tokamak dataset
    """
    #hdf5 files are found within a set of folders in the primary directory
    hdf5_filelist = []
    for root, dirs, files in os.walk(hdf5_path):
        for file in files:
            try:
                if file.lower() != ".ds_store" and file.lower().endswith(".hdf5"):
                    hdf5_filelist.append(os.path.join(root, file))
            except Exception as e:
                logging.error("Failed to append file %s: %s", file, e)

    logging.info("HDF5 files sourced from %s", hdf5_path)
    logging.info("Preview: %s", hdf5_filelist[:5])
    return hdf5_filelist

def sourceHDF5Data(file_list:list, features:list = None) -> "feature pandas dataframe":
    """
    Sources features from an HDF5 file and outputs a 2D pandas dataframe with features and multifile feature data
    By default (None) sources all features from file
    """
    feat_data = pd.DataFrame(dtype=object)
    #check if the file_list is/is not a tuple and source only files from the tuple if so
        
    for file_dir in file_list:
        feat_list = pd.DataFrame(dtype=object)
        with h5py.File(file_dir, 'r') as hdf5_file:
            for feature in features if features else hdf5_file['data']:
                try:
                    feat = np.array(hdf5_file[f'data/{feature}'])
                    feat_list[feature] = [feat]
                except Exception as e:
                    logging.error("Failed to add feature: %s", e)
        feat_data = pd.concat([feat_data, pd.DataFrame(feat_list, dtype=object)], ignore_index=True)
    logging.info("%s data sourced from HDF5s", features if features else "Features")
    return feat_data
