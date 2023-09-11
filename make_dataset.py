import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler    


class FloodDataset(torch.utils.data.Dataset):
    '''
  Prepare the flood dataset for regression
  '''

    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float()
    

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]