## [Total Code](Multiple%20Dimension%20Input.py)

``` python
import torch
import numpy as np

xy = np.loadtxt("diabetes.csv.gz", delimiter=",", dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


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

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, 101):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    # Backward
    optimizer.zero_grad()
    loss.backward()
    # Update
    optimizer.step()
```

### 1. Prepare Dataset
``` python
xy = np.loadtxt("diabetes.csv.gz", delimiter=",", dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
```

#### Explain:
>___xy = np.loadtxt("diabetes.csv.gz", delimiter=",", dtype=np.float32)___
>>__Read data from "diabetes.csv.gz",using "," to split data.
Type is np.float32.__

>___x_data = torch.from_numpy(xy[:, :-1])___
>>__Get data from the form -- numpy.
>>The meaning of ":" is geting the whole row or column.
>>The meaning of "-1" is geting all rows or columns from the last second row or columns.__

### 2. Build Model
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
>___self.linear = torch torch.nn.Linear(x,y)___
>>__The features turn x to y.__
>>>_In this example,we turn 8 features to 1._
>>>_Finally,we get the target._

>___self.sigmoid = torch.nn.Sigmoid()___
>>__We get sigmoid function from torch.nn.Sigmoid(),which is a class,different from 'F.sigmoid' in [Logistic.py](../逻辑斯蒂回归/Logistic.py) , which is a function.__
### 3. Construct Loss and Optimizer
``` python
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```


### 4. Training cycle
``` python
for epoch in range(1, 101):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    # Backward
    optimizer.zero_grad()
    loss.backward()
    # Update
    optimizer.step()
```