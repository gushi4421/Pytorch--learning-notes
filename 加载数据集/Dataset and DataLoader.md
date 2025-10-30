
## [Total Code](Dataset%20and%20DataLoader.py)
``` python

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


# 1.Prepare Dataset
class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
        self.len=xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset("diabetes.csv.gz")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


# 2.Create Model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# 3.Construct loss and optimizer
criterion = torch.nn.Sigmoid()  # Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4.Training Cycle
for epoch in range(1, 101):
    for i, data in enumerate(train_loader, 0):
        # 1. Prepare data
        inputs, labels = data
        # 2. Forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()

```


### 1. Prepare Dataset
``` python
class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset("diabetes.csv.gz")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
```
__This part is different from [Multiple Dimension Input](../处理多维特征的输入/Multiple%20Dimension%20Input.py).We take it as a class.__
>___Dataset___
>>__Dataset is a class,which is an abstract class.We can define out class inherited from this class.__  
>>__DatabetesDataset is inherited from abstract class Dataset.__

>___DataLoader___
>>__DataLoader is a class to help us loading data in PyTorch.__

### 2. Design Dataset
``` python
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
```


### 3. Construct Loss and Optimizer
``` python
criterion = torch.nn.BCELoss()  # Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```


### 4. Training Cycle
``` python
for epoch in range(1, 101):
    for i, data in enumerate(train_loader, 0):
        # 1. Prepare data
        inputs, labels = data
        # 2. Forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()
```
