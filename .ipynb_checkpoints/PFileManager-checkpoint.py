import h5py
import os
import numpy as np
import pandas as pd
import logging
import shutil

logging.basicConfig(filename="./logs/PlasmaDataset_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def isDisruptiveHDF5(hdf5_loc):
    """
    Defines a routine to classify HDF5 file as disruptive/non-disruptive using metadata
    1 - disruptive
    """
    with h5py.File(hdf5_loc) as f:
        label = f['meta/IsDisrupt'][()]
        
    return 1 if label else 0

def organizeData(files, org_dir, label_names=[0, 1]):
    """
    Pulls files from a pandas dataframe sorted into dataset filelists into a directory with labels
    Each dataset filelist header i.e. "train/test/val" will be used as the subdirectory folder name
       If labels have names, i.e. nondisruptive/disruptive, these should be noted in "label_names" otherwise 0/1 by default
    Each file within a dataset filelist is a tuple (filepath, label)
    By default creates dataset subdirectories for binary labels - i.e. train/test/val each have disruptive/nondisruptive folders
    """
    
    for fileset in files:
        subdir = os.path.join(org_dir, fileset)
        label0, label1 = os.path.join(subdir, label_names[0]), os.path.join(subdir, label_names[1])
        os.mkdir(subdir, exist_ok=True)
        os.mkdir(label0, exist_ok=True)
        os.mkdir(label1, exist_ok=True)
 
    for file_tuple in files[fileset]:
        file, label = file_tuple[0], file_tuple[1]
        try:   
            file_path = os.path.splitext(os.path.basename(file))[0]
            new_file_path = os.path.join(label1, file) if label else os.path.join(label0, file)
            shutil.copy(file, new_file_path + '.hdf5')
        except Exception as e:
            logging.error("Skipping file %s: %s", file, e)
    
    logging.info(f"Exported {len(files[fileset])} hdf5 files to %s", data_dir)

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

    for file_dir in file_list:
        feat_list = pd.DataFrame(dtype=object)
        with h5py.File(file_dir, 'r') as hdf5_file:
            for feature in features if features else hdf5_file['data']:
                try:
                    feat = np.array(hdf5_file[f'data/{feature}'])
                    feat_list[feature] = [feat]
                except Exception as e:
                    logging.error(f"Failed to add feature due to {e}")
        feat_data = pd.concat([feat_data, pd.DataFrame(feat_list, dtype=object)], ignore_index=True)
        
    return feat_data
