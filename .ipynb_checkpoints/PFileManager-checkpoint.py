import h5py
import os
import numpy as np
import logging
import shutil

logging.basicConfig(filename="./logs/PlasmaDataset_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class PFileManager:
    
    def isDisruptiveHDF5(self, hdf5_loc):
        """
        Defines a routine to classify HDF5 file as disruptive/non-disruptive using metadata
        1 - disruptive
        """
        with h5py.File(hdf5_loc) as f:
            label = f['meta/IsDisrupt'][()]
            
        return 1 if label else 0
    
    def organizeData(self, files, org_dir, label_names=[0, 1]):
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
    
    def sourceHDF5(self, hdf5_path):
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
    
    def sourceHDF5Data(self, file_dir, features = None):
        """
        Sources features from an HDF5 file and outputs 2D feature list
        By default (None) sources all features from file
        """
        feat_list = []
        feat_names = []
        with h5py.File(file_dir, 'r') as hdf5_file:
            for feature in features if features else hdf5_file['data']:
                try:
                    feat = np.array(hdf5_file[f'data/{feature}'])
                    feat_list.append(feat)
                    feat_names.append(feature)
                except Exception as e:
                    logging.error(f"Failed to add feature due to {e}")
                    
        return feat_list, feat_names