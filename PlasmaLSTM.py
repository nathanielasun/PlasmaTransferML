import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PlasmaLSTM(nn.Module):
    """
    Creates and manages a LSTM model for disruption prediction
    Assumes a LSTM -> Linear -> Output (one-hot) structure
    Data is imported as a tensor, not a dataframe
    parameters is a list including 
    """
    def __init__(self, parameters:"parameter dict"):
        super(PlasmaLSTM, self).__init__()
        self.parameters = parameters
        self.model = nn.ModuleDict({})
        
        #initialize default layer info
        if 'input_size' not in self.parameters:
            self.parameters['input_size'] = None
        if 'lstm_layers' not in self.parameters:
            self.parameters['lstm_layers'] = [ 396, 101 ] #creates LSTM with given cell count for layer in layers variable
        if 'linear_layers' not in self.parameters:
            self.parameters['linear_layers'] = [ 300 ] #adds linear layer with neuron count per layer in list
            
        #initialize default activators
        if 'lstm_activation' not in self.parameters:
            self.parameters['lstm_activation'] = F.tanh()
        if 'linear_activation' not in self.parameters:
            self.parameters['linear_activation'] = F.relu()
        if 'output_activation' not in self.parameters:
            self.parameters['output_activation'] = F.sigmoid()
        if 'optimizer' not in self.parameters:
            self.parameters['optimizer'] = 'adam'

        #initialize default hyperparameters
        if 'lr' not in self.parameters:
            self.parameters['lr'] = 0.003
        if 'init_weight' not in self.parameters:
            self.parameters['init_weight'] = 'glorot'
        """
        Constructs LSTM layers using the list 'lstm_layers' containing hidden size parameters for each layer
        Constructs linear layers using the list 'linear_layers' containing neuron counts for each layer
        """
        input_size = self.parameters['input_size']
        #add LSTM layer(s) to model
        self.model['lstm'] = nn.ModuleList([])
        for hidden_size in self.parameters['lstm_layers']
            self.model['lstm'].append(nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                batch_first=True)
            )
            input_size = hidden_size

        #add linear layer(s) to model from 'linear_layers'
        self.model['linear'] = nn.ModuleList([])
        in_features = input_size
        for i in range(len(self.parameters['linear_layers']))
            self.model['linear'].append(
            nn.Linear(
                in_features=in_features,
                out_features=self.parameters['linear_layers'][i]
            )
            out_features = in_features
        #create a one-hot output layer
        self.model['output'] = nn.ModuleList([])
        self.model['output'].append(nn.Linear(in_features = out_features, out_features = 1))
        
    def forward(self, X):
        """
        Pass data through model layers
        """
        for lstm in self.model['lstm']:
            X = lstm(X)
            X = self.parameters['lstm_activation'](X)
        for linear in self.model['linear']:
            X = linear(X)
            X = self.parameters['linear_activation'](X)
        X = self.model['output'](X)

        return X

#------------------------------------------------------------------------------------------------
"""
Routines to handle model creation, training, and evaluation
"""
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
model = None
params = None

def makeLSTM(parameters:"paramter dict"):
    model = PlasmaLSTM(parameters)
    model.to(device)
    params = parameters
    
def trainLSTM(train_data, val_data):
    optimizer = optim.Adam(
        
def runLSTM(data):

def testLSTM(test_data):

        
        





