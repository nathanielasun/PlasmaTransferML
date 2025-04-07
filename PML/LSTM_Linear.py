import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Linear(nn.Module):
    """
    Creates and manages a LSTM model for disruption prediction
    Assumes a LSTM -> Linear -> Output (one-hot) structure
    Data is imported as a tensor, not a dataframe
    parameters is a list including 
    """
    def __init__(self, params:"parameter dict"):
        super(LSTM_Linear, self).__init__()
        self.params = params
        self.model = nn.ModuleDict({})
        
        #initialize default layer info
        if 'input_size' not in self.params:
            self.params['input_size'] = None
        if 'lstm_layers' not in self.params:
            self.params['lstm_layers'] = [ 396, 101 ] #creates LSTM with given cell count for layer in layers variable
        if 'linear_layers' not in self.params:
            self.params['linear_layers'] = [ 300 ] #adds linear layer with neuron count per layer in list
        if 'dropout_layers' not in self.params:
            self.params['dropout_layers'] = [] #adds no default dropout layers - list takes form [p1, p2, ...] where p is dropout prob
            
        #initialize default activation functions
        if 'lstm_activation' not in self.params:
            self.params['lstm_activation'] = F.tanh
        if 'linear_activation' not in self.params:
            self.params['linear_activation'] = F.relu
        if 'output_activation' not in self.params:
            self.params['output_activation'] = F.sigmoid
            
        """
        Constructs LSTM layers using the list 'lstm_layers' containing hidden size parameters for each layer
        Constructs linear layers using the list 'linear_layers' containing neuron counts for each layer
        """
        input_size = self.params['input_size']
        #add LSTM layer(s) to model
        self.model['lstm'] = nn.ModuleList([])
        for hidden_size in self.params['lstm_layers']:
            self.model['lstm'].append(nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                batch_first=True)
            )
            input_size = hidden_size

        #add linear layer(s) to model from 'linear_layers'
        self.model['linear'] = nn.ModuleList([])
        in_features = input_size
        for i in range(len(self.params['linear_layers'])):
            self.model['linear'].append(
            nn.Linear(
                in_features=in_features,
                out_features=self.params['linear_layers'][i]
            ))
            in_features = self.params['linear_layers'][i]
            #implement dropout layers after each linear layer if dropout layers
            if i < len(self.params['dropout_layers']):
                self.model['linear'].append(
                nn.Dropout(
                    p = self.params['dropout_layers'][i]
                ))
        #create single neuron output layer
        self.model['output'] = nn.ModuleList([])
        self.model['output'].append(nn.Linear(in_features = in_features, out_features = 1))
        
    def forward(self, X):
        """
        Pass data through model layers
        """
        #pass through LSTM layers
        for lstm in self.model['lstm']:
            X,_ = lstm(X)
            X = self.params['lstm_activation'](X)
        X = X.mean(dim=1)  # mean across sequence length dimension
        #pass through linear and dropout layers
        for linear in self.model['linear']:
            X = linear(X)
            if isinstance(linear, nn.Linear):
                X = self.params['linear_activation'](X)
        #pass through output layer
        for layer in self.model['output']:
            X = layer(X)
        #X = self.params['output_activation'](X)
        X = torch.flatten(X)
        return X