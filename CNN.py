# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:34:10 2024

@author: Lenovo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch import tensor
from get_data.get_data import get_data_Alu, GetSimulateData
from torch.utils.data import DataLoader, TensorDataset

path_simulate47 = 'E:/code/7new_class_learning/Data/TrainDataset4_7/TrainDataset' #总共5个类别
path_streaming_simulate47 = 'E:/code/7new_class_learning/Data/StreamingTimeSeries4_7/StreamingTimeSeriesA0.xlsx'

class MultiLayerCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiLayerCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
def training():
    X_train,y_train,X_test,y_test,online,scaler = GetSimulateData(path_simulate47,path_streaming_simulate47).main()
    if not isinstance(X_train, torch.Tensor):
        X_train = tensor(X_train, dtype=torch.float).contiguous()
    if not isinstance(y_train, torch.Tensor):
        y_train = tensor(y_train, dtype=torch.long).contiguous()
    if not isinstance(X_test, torch.Tensor):
        X_test = tensor(X_test, dtype=torch.float).contiguous()
    if not isinstance(y_test, torch.Tensor):
        y_test = tensor(y_test, dtype=torch.long).contiguous()

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=False, drop_last=False)
    
    #print(X_train.shape[-1])
    model = MultiLayerCNN(X_train.shape[-1],5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        for j, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1) 
        print(predicted,y_test)
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)
       
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
    return

training()

