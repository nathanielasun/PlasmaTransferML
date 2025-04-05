import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from PML.LSTM_Linear import LSTM_Linear
from PML.PMLParameters import PMLParameters

logging.basicConfig(filename="../logs/PlasmaModel_logs.txt", level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

#---------------------------------------------------------------
"""
Class to handle model creation, training, and evaluation
Makes an model according to input parameter dict (can be received from randomizer)
"""
class PlasmaModel:
    def __init__(self, parameters):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.model = None
        self.params = parameters

    def exportModel(self, loc):
        try: 
            name = f"LSTM{self.params['lstm_layers']}_linear{self.params['linear_layers']}_lr={self.params['lr']}.pth"
            exp_loc = os.path.join(loc, name)
            torch.save(model, exp_loc)
        except Exception as e:
            logging.error("Failed to save model: %s", e)
    
    def makeLSTM(self):
        """
        Makes an LSTM model using LSTM_Linear and parameters
        Moves model onto device and assigns model to class instance variable
        """
        try:
            model = LSTM_Linear(self.params)
            model.to(self.device)
            logging.info("Created PlasmaLSTM model")
    
        except Exception as e:
            logging.error("Failed to create PlasmaLSTM model: %s", e)
        
    def trainModel(self, train_data, val_data):
        """
        """
        try:
            optimizer = params['optimizer'](model.paramters(), lr=params['lr'])
            criterion = params['criterion']
            model.apply(initWeights)
            train_loader = self.makeDataLoader(train_data).to(self.device)
            val_loader = self.makeDataLoader(val_data).to(self.device)
        except Exception as e:
            logging.error("Failed to initialize optimizer/criterion/loader: %s", e)
        
        try:
            for epoch in range(params['epochs']):
                model.train()
                running_loss = 0.0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()           
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()       
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                
                epoch_train_loss = running_loss / len(train_loader.dataset)
                print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}")
                
                # Validation Phase
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                epoch_val_loss = val_loss / len(val_loader.dataset)
                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_val_loss:.4f}")
        except Exception as e:
            logging.error("Failed to train model: %s", e)
            
    def testModel(self, test_data):
        try:    
            criterion = self.params['criterion']        
            test_loader = self.makeDataLoader(test_data).to(self.device)
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item() * inputs.size()
        except Exception as e:
            logging.error("Failed to initialize and test model: %s", e)
        print(f"Model loss: {test_loss:.4f}")
        return test_loss
        
    #---------Routines to aid in model creation----------
    def makeDataLoader(self, data:"tensor dataset")->"torch dataloader":
        """
        Creates pytorch dataloader for data according to specs
        """
        try:
            dataloader = DataLoader(data, params['batch_size'], shuffle=True)
        except Exception as e:
            logging.error("Failed to create dataloader: %s", e)
    
    def initWeights(m, zero_bias=False):
        try:
            # Check if the module has a weight attribute and it's not None.
            if hasattr(m, 'weight') and m.weight is not None:
                # Apply Xavier uniform initialization to the weight tensor.
                params['init'](m.weights)
            if zero_bias:    
                if m.bias is not None:
                    m.bias.data.zero_()
        except Exception as e:
            logging.error("Failed to initialize weights: %s", e)
    #--------data importing routines--------
    def importData(self, data:"
    