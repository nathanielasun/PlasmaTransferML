import h5py
import inspect
import logging
import PData
import PDataOperations
import PFileManager as PFM

logging.basicConfig(filename="./logs/PlasmaDataset_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class PlasmaDataset:
    
    def __init__(self, org_directory, h5_source):

        Dataset = new PData(org_directory, h5_source)
    
    def createMLDataStructure(self, data_split, reset=True):
        """
        Routine to generate train/test/val data structure with respective disruptive/nondisruptive folders
        Keeping reset true will replace the existing datasets train/test/val
        """
        Dataset.addDataset("train", RESET=reset)
        Dataset.addDataset("test", RESET=reset)
        Dataset.addDataset("val", RESET=reset)
        
        logging.info("Created train/test/val dirs at: \n%s\n%s\n%s", train_dir, test_dir, val_dir)

    def sourceDataset(self, dataset, features = None, labeled=True, reset=True):
        """
        Acquires raw data features from file list for given dataset at data fraction
        i.e. for train data, sources features from hdf5 files in train directory
        Sources features from given feature list. None indicates all hdf5 features by default.
        Note: reset will completely wipe the existing dataset - used by default for feature tuning
        """
        data_frac = self.data_config['data_frac']
        
        if reset:
            self.wipeDataset(dataset)
            
        feat_list = [] #list to take specified file features
        #initialize labels within x dataset dictionary
        self.data[dataset]['raw']['label'] = []
        files = self.data[dataset]['files']
        for hdf5_file in files[:round(len(files)*data_frac)]:
            feat_list, feat_names = self.sourceHDF5Data(hdf5_file, features)
            for i in range(len(feat_names)):
                if feat_names[i] in self.data[dataset]['raw']:
                    self.data[dataset]['raw'][feat_names[i]].append([feat_list[i]])
                else:
                    self.data[dataset]['raw'][feat_names[i]] = [feat_list[i]]
            if labeled:#append the label information if the hdf5/datafile contains it
                self.data[dataset]['raw']['label'].append(self.isDisruptiveHDF5(hdf5_file))
                
        logging.info(f"Constructed dataset '{dataset}' with {len(self.data[dataset]['raw']['label'])} files and {len(self.data[dataset]['raw']) - 1 if labeled else 0 } features")
    
        