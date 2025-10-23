import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class DiabetesDataset(torch.nn.Module):
    def __init__(self,file_path):
        xy=np.loadtxt(file_path,delimiter=',',dtype=np.float64)
        self.x_data=torch.from_numpy(xy[:,:-1])


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


for epoch in range(1,101):
    for i,data in enumerate(train_loader,0):
        #1. Prepare data
        inputs,label=data