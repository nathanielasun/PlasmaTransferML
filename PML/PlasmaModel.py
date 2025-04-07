import ast
import json
import logging
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PML.LSTM_Linear import LSTM_Linear
from PML.PMLParameters import PMLParameters
import PML.PMLPreprocessing as PMLPreprocessing

logging.basicConfig(filename="./logs/PlasmaModel_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

#---------------------------------------------------------------
"""
Class to handle model creation, training, and evaluation
Makes an model according to input parameter dict (can be received from randomizer)
"""
class PlasmaModel:
    def __init__(self, model_save_dir, json_save_file, static_parameters):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.model = None
        self.model_dir = model_save_dir
        self.json_save_file = json_save_file
        self.model_metrics = {
            'test_loss' : None,
            'test_tp'   : None,
            'test_tn'   : None,
            'test_acc'  : None
        }
        self.dataloaders = {}
        self.params = static_parameters
        self.param_list = []

    def makeHyperparameterSet(self, static_params:dict, param_ranges:dict, count:int, mode:str):
        """
        Uses PMLParameters to generate hyperparameter queue set
        """
        param_searcher = PMLParameters(ranges=param_ranges, include=static_params)
        if mode.lower() == 'grid':
            param_searcher.makeGridParameterSet(count)
            self.param_list = param_searcher.getParameterSet()
        elif mode.lower() == 'random':
            param_searcher.makeRandomParameterSet(count)
            self.param_list = param_searcher.getParameterSet()
        else:
            logging.error("Incorrect mode %s specified", mode)

    def runModelSearch(self, limit:int=None):
        """
        Make and train a model for each hyperparameter set
        Makes a model for full parameter set list unless "limit" is specified
        """
        for parameter_set in self.param_list if limit is None else self.param_list[:limit]:
            print(parameter_set)
            self.params = parameter_set
            self.makeLSTM(parameter_set)
            self.trainModel(self.dataloaders['train_loader'], self.dataloaders['val_loader'])
            self.testModel(self.dataloaders['test_loader'])
            model_name = f"LSTM{self.params['lstm_layers']}_linear{self.params['linear_layers']}_lr={self.params['lr']}.pth"
            self.exportModel(self.model_dir, name=model_name)
            self.exportModelMetrics(parameters=parameter_set, metrics=self.model_metrics, name=model_name)
    
    def prepareData(self, data_loc:dict):
        """
        Source norm and label data from CSV files
        Convert from dataframe to tensor
        Move all data to specified device
        Make dataloaders for all datasets
        """
        #source to PD dataframe
        try:
            header_cols = pd.read_csv(data_loc['val_norm'], nrows=0).columns
            label_header = pd.read_csv(data_loc['val_labels'], nrows=0).columns
            train_norm_df = pd.read_csv(data_loc['train_norm'], converters={col : self.string_to_list for col in header_cols})
            train_labels_df = pd.read_csv(data_loc['train_labels'])
            test_norm_df = pd.read_csv(data_loc['test_norm'], converters={col : self.string_to_list for col in header_cols})
            test_labels_df = pd.read_csv(data_loc['test_labels'])
            val_norm_df = pd.read_csv(data_loc['val_norm'], converters={col : self.string_to_list for col in header_cols})
            val_labels_df = pd.read_csv(data_loc['val_labels'])
        except Exception as e:
            logging.error("Failed to source PD dataframes: %s", e)
        #process dataframes and labels to tensors
        try:
            train_norm_tensor = PMLPreprocessing.tensorData(train_norm_df)
            test_norm_tensor = PMLPreprocessing.tensorData(test_norm_df)
            val_norm_tensor = PMLPreprocessing.tensorData(val_norm_df)
            train_labels_tensor = torch.tensor(train_labels_df[label_header[0]], dtype=torch.float32)
            test_labels_tensor = torch.tensor(test_labels_df[label_header[0]], dtype=torch.float32)
            val_labels_tensor = torch.tensor(val_labels_df[label_header[0]], dtype=torch.float32)
        except Exception as e:
            logging.error("Failed to convert DFs to tensor: %s", e)
        #move tensor data to GPU
        try:
            train_norm_tensor=train_norm_tensor.to(self.device)
            test_norm_tensor=test_norm_tensor.to(self.device)
            val_norm_tensor=val_norm_tensor.to(self.device)
            train_labels_tensor=train_labels_tensor.to(self.device)
            test_labels_tensor=test_labels_tensor.to(self.device)
            val_labels_tensor=val_labels_tensor.to(self.device)
        except Exception as e:
            logging.error("Failed to move tensors to %s: %s", str(self.device), e)
        #convert train/test/val data and label tensors to TensorDataset objects
        train_td = TensorDataset(train_norm_tensor, train_labels_tensor)
        test_td = TensorDataset(test_norm_tensor, test_labels_tensor)
        val_td = TensorDataset(val_norm_tensor, val_labels_tensor)
        #use TensorDataset objects to store dataloaders
        self.dataloaders['train_loader'] = self.makeDataLoader(train_td)
        self.dataloaders['test_loader'] = self.makeDataLoader(test_td)
        self.dataloaders['val_loader'] = self.makeDataLoader(val_td)

    #-----------Routines to aid in data interpretation-----------

    def string_to_list(self, s):
        # Use ast.literal_eval for conversion from string to list
        lst = ast.literal_eval(s)
        return [np.float32(item) for item in lst]
        
    #------------Routines to aid in model management-------------

    def accuracy(self, outputs, targets):
        # Count total correct predictions
        correct = sum(abs(o - t) < 0.5 for o, t in zip(outputs, targets))
        
        # Predicted positives vs. negatives
        p = sum(o >= 0.5 for o in outputs)
        n = len(outputs) - p
        
        # Correct positives vs. correct negatives
        tp = sum((abs(o - t) < 0.5) and (o >= 0.5) for o, t in zip(outputs, targets))
        tn = sum((abs(o - t) < 0.5) and (o < 0.5)  for o, t in zip(outputs, targets))
        
        acc = correct / len(outputs)
        return (tp / p if p else 0.0), (tn / n if n else 0.0), acc
    
    def exportModel(self, loc, name):
        try: 
            exp_loc = os.path.join(loc, name)
            torch.save(self.model.state_dict(), exp_loc)
        except Exception as e:
            logging.error("Failed to save model: %s", e)
            
    def exportModelMetrics(self, name, parameters, metrics, JSON_file=None):
        """
        Exports JSON of model performance
        Uses model directory to save JSON file
        Records TP, TN, total pos, total neg, LOSS evaluation, and parameters
        """
        if JSON_file is None:
            JSON_file = self.json_save_file
    
        # Define the list of parameters you want to include in the JSON file
        JSON_params = ['lr', 'lstm_layers', 'linear_layers', 'dropout_layers']
        
        # Create a dictionary of the selected parameters and their values
        #JSON_dict = {key: parameters[key] for key in JSON_params if key in parameters}
    
        # Prepare the data to be exported
        JSON_data = {
            'name': name,
            #'hyperparameters': JSON_dict,  # Changed to use the filtered dictionary
            'metrics': metrics
        }
    
        try:
            # Read existing JSON data if the file exists
            try:
                with open(JSON_file, 'r') as json_file:
                    existing_json_data = [json.load(json_file)]
            except FileNotFoundError:
                # If the file doesn't exist, initialize an empty list
                existing_json_data = []
    
            # Append the new data to the existing data
            existing_json_data.append(JSON_data)
    
            # Write the updated data back to the JSON file
            with open(JSON_file, 'w') as json_file:
                json.dump(existing_json_data, json_file, indent=4)  # indent=4 is for pretty formatting
    
        except Exception as e:
            logging.error("Failed to export JSON for %s: %s", name, e)
        
    def makeLSTM(self, parameters):
        """
        Makes an LSTM model using LSTM_Linear and parameters
        Moves model onto device and assigns model to class instance variable
        """
        try:
            self.model = LSTM_Linear(parameters)
            self.model.to(self.device)
        except Exception as e:
            logging.error("Failed to create LSTM_Linear model: %s", e)
    
    def testModel(self, test_loader):
        """
        Tests current model with test_loader data
        """
        try:    
            criterion = self.params['criterion']        
            self.model.eval()
            test_loss = 0.0
            tp, tn, acc = 0,0,0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    tp_new, tn_new, acc_new = self.accuracy(outputs, targets)
                    tp += tp_new
                    tn += tn_new
                    acc += acc_new
                    test_loss += loss.item() * inputs.size(0)
            self.model_metrics['test_tp'] = float(tp/len(test_loader.dataset))
            self.model_metrics['test_tn'] = float(tn/len(test_loader.dataset))
            self.model_metrics['test_acc'] = float(acc/len(test_loader.dataset))
        except Exception as e:
            logging.error("Failed to initialize and test model: %s", e)
        print(f"Model loss: {test_loss:.4f}")
        self.model_metrics['test_loss'] = test_loss
        return test_loss

    def trainModel(self, train_loader, val_loader):
        """
        Train and validate model using train/val dataloaders
        """
        try:
            optimizer = self.params['optimizer'](self.model.parameters(), lr=self.params['lr'])
            criterion = self.params['criterion']
            self.model.apply(self.initWeights)
        except Exception as e:
            logging.error("Failed to initialize optimizer/criterion: %s", e)
        
        try:
            for epoch in range(self.params['epochs']):
                self.model.train()
                running_loss = 0.0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()        
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()       
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                
                epoch_train_loss = running_loss / len(train_loader.dataset)
                print(f"Epoch [{epoch+1}/{self.params['epochs']}], Training Loss: {epoch_train_loss:.4f}")

                # Validation Phase
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                epoch_val_loss = val_loss / len(val_loader.dataset)
                print(f"Epoch [{epoch+1}/{self.params['epochs']}], Validation Loss: {epoch_val_loss:.4f}")
        except Exception as e:
            logging.error("Failed to train model: %s", e)

    #-------------Routines to aid in model creation--------------
    def initWeights(self, m, zero_bias=False):
        """
        Initializes model weights to specified weighting
        """
        try:
            # Check if the module has a weight attribute and it's not None.
            if hasattr(m, 'weight') and m.weight is not None:
                # Apply initialization to the weight tensor.
                self.params['init'](m.weight)
            if zero_bias:    
                if m.bias is not None:
                    m.bias.data.zero_()
        except Exception as e:
            logging.error("Failed to initialize weights: %s", e)
            
    def makeDataLoader(self, data:"tensor dataset")->"torch dataloader":
        """
        Creates pytorch dataloader for data according to specs
        """
        try:
            dataloader = DataLoader(data, batch_size=self.params['batch_size'], shuffle=True)
        except Exception as e:
            logging.error("Failed to create dataloader: %s", e)
            return None
        return dataloader
