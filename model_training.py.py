# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:50:38 2023

@author: Gao,Jianguang

"""

import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd


def main():
    
    
    dataset_air = TraingingDatasetAir()

    dataloader = DataLoader(dataset=dataset_air,batch_size=24, shuffle=True,num_workers=2)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    
    model = NeuralNetwork().to(device)
    #print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer,device)
        #test(test_dataloader, model, loss_fn,device)
    print("Done!")
    
    
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    

    
class  TraingingDatasetAir(Dataset):
    
    def __init__(self):
        
        # load data
        
        # training input
        path_x = '.\Data_A\X_train_air.csv'
        path_y = '.\Data_A\y_train_air.csv'
        
        x = pd.read_csv(path_x)
        y = pd.read_csv(path_y)
        drop_x = ['Unnamed: 0','year','month','day','week','hour','小时','wind_direction','风向角度(°)']
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
    


# Define classic nn model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11, 512).double(),
            nn.ReLU(),
            nn.Linear(512, 512).double(),
            nn.ReLU(),
            nn.Linear(512, 115).double(),
            nn.ReLU(),
            nn.Linear(115, 512).double(),
            nn.ReLU(),
            nn.Linear(512, 168).double()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



def train(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
def test(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  
    

if __name__ == '__main__' :
    main()
