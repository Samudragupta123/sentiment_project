import os
print(os.listdir())
print(os.listdir('data'))
import pandas as pd
import numpy as np
dataset=pd.read_csv('data/raw_dataset.csv')
dataset.head()
#dropping redundant columns
dataset=dataset.drop(dataset.columns[[0,6,7,8,9,11,12,13,14]], axis=1)
dataset.head()
datasetfinal=dataset.fillna(0)
datasetfinal.head()
X=dataset.iloc[:,1:5].values
y=dataset.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
import torch
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        # Get one sample from the numpy arrays
        sample_features = self.x_data[idx]
        sample_label = self.y_data[idx]
train_dataset= MyDataset(X_train, y_train)
test_dataset= MyDataset(X_test, y_test)
from torch.utils.data import DataLoader

# Training loader: Shuffling is True
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=1000, 
    shuffle=True  
)

# Testing/Validation loader: Shuffling is usually False
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=100, 
    shuffle=False
)