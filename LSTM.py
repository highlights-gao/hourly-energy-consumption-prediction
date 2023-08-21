# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:47:53 2023

@author: gaoji
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 09:51:22 2023

@author: gaoji
"""


import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd

    
class  TraingingDatasetAir(Dataset):
    
    def __init__(self):
        
        # load data
        
        # training input
        path_x = '.\Data_A\X_train_air.csv'
        path_y = '.\Data_A\y_train_air.csv'
        
        x = pd.read_csv(path_x)
        y = pd.read_csv(path_y)
        drop_x = ['Unnamed: 0','week','小时','wind_direction','风向角度(°)']
        drop_y = ['Unnamed: 0']
        x = x.drop(drop_x,axis=1)
        y = y.drop(drop_y,axis=1)

        self.x = torch.from_numpy(x.values)
        self.y = torch.from_numpy(y.values)
        self.num_samples = x.shape[0]
               
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.num_samples


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

dataset_air = TraingingDatasetAir()

dataloader = DataLoader(dataset=dataset_air,batch_size=24, shuffle=True,num_workers=2)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0)) # output(5, 30, 64)
        pred = self.linear(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred

