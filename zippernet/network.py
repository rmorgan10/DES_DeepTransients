"""
Neural network for DeepTransient Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, config_dict: dict):
        super(CNN, self).__init__()

        self.config_dict = config_dict

        for layer, kwargs in config_dict.items():
            if layer.startswith("conv"):
                obj = nn.Conv2d
            elif layer.startswith("dropout"):
                obj = nn.Dropout2d
            elif layer.startswith("fc"):
                obj = nn.Linear

            setattr(self, layer, obj(**kwargs))     
        
    def forward(self, x):

        for layer, kwargs in self.config_dict:
            if (layer.startswith("conv") or 
                layer.startswith("dropout") or 
                layer.startswith("fc")):
                x = getattr(self, layer)(x)
            
            elif layer.startswith("relu"):
                x = F.relu(x)

            elif layer.startswith("maxpool"):
                x = F.max_pool2d(x, **kwargs)

            elif layer.startswith("flatten"):
                x = torch.flatten(x, 1)

            elif layer.startswith("softmax"):
                x = F.log_softmax(x, dim=1)

        return x

    
class RNN(nn.Module):
    def __init__(self, config_dict: dict):
        super(RNN, self).__init__()

        self.config_dict = config_dict

        for layer, kwargs in config_dict.items():
            if layer.startswith("lstm"):
                obj = nn.LSTM
            elif layer.startswith("fc"):
                obj = nn.Linear

            setattr(self, layer, obj(**kwargs))

    def forward(self, x):

        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        hn_hc_tuple = None  # None represents zero initial hidden state
        fc_flag = False
        for layer in self.config_dict:
            if layer.startswith("lstm"):
                x, hn_hc_tuple = getattr(self, layer)(x, hn_hc_tuple)   

            if layer.startswith("fc"):
                if not fc_flag:
                    x = x[:, -1, :]  # Choose output at the last time step.
                    fc_flag = True
                x = getattr(self, layer)(x)

        return x

    
class ZipperNN(nn.Module):
    def __init__(self, config_dict: dict):
        super(ZipperNN, self).__init__()

        self.config_dict = config_dict
        
        #Network Components
        self.cnn = CNN(config_dict['CNN'])
        self.rnn = RNN(config_dict['RNN'])

        for layer, kwargs in self.config_dict.items():
            if layer.startswith("fc"):
                setattr(self, layer, nn.Linear(**kwargs))
        
    def forward(self, x, y):
        rnn_output = self.rnn(x)
        cnn_output = self.cnn(y)

        full_output = torch.cat((rnn_output, cnn_output), dim=1)
        
        for layer in self.config_dict:
            if layer.startswith("fc"):
                full_output = getattr(layer)(full_output)
            elif layer.startswith("relu"):
                full_output = F.relu(full_output)
            elif layer.startswith("softmax"):
                full_output = F.log_softmax(full_output, dim=1)
        
        return full_output
