## PyTorch--learning notes

#### 1.Prepare dataset
##### we will share this later
#### 2.Design model using Class
##### inherit from nn.Module
#### 3.Construct loss and optimizer
##### using PyTorch API
#### 4.Training cycle
##### forward,backward,update


__In PyTorch,the computational graph is in mini-batch fashion,so X and T are 3*1 Tensors.__

#### To be honest,this my first time to use torch to design my code.Therefore,I record many details.

![alt text](Linear.png)

## [Total code](PyTorch%20Fashion.py) : 
``` python
import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):  # nn == neural network
    def __init__(self):  # Constructor function
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(1, 101):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print("Epoch:", epoch, "loss=", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # update parameters

print("w=", model.linear.weight.item())
print("b=", model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print("Predict (after training)", 4, y_test.item())
```

### 1. Prepare Dataset
``` python
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
```
__Just some test data.__
### 2. Design Model
``` python
class LinearModel(torch.nn.Module): # nn neural network
```
>___Our model class should be inherited from nn.Module,which is Base class for all neural network modules.___


>__Member methods _\_\_init\_\_()_ and _forward()_ have to be implemented.__

``` python
def __init__(self):
        super(LinearModel, self).__init__() 
        self.linear = torch.nn.Linear(1, 1)
```

>__for super:__
>>__just do it__

>__Class nn.Linear contain two member Tensors:_weight_ and bias__

``` python
self.linear=torch.nn.Linear(1,1)
```
>__Applies a linear transformation to the incoming data : $y=Ax+b$__

>__class torch.nn.Linear(in_features,out_features,bias=True)__
>>__Parameters:__    
>>>___in_features - size of each input sample___
               ___out_features - size of each output sample___
               ___bias - if set False,the layer will not learn an additive bias___

``` python
def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
```
>__Class ___<u>nn.Linear___</u> has implemented the magic method ___\_\_call\_\____,which enable the instance of the class can be called just like a function.Normally the ___forward()___ will be called.__

``` python
model = LinearModel()
```
>__Create a instance of LinearModel__


### 3. Construct Loss and Optimizer
``` python
criterion = torch.nn.MSELoss(size_average=False)
```
>##### [^1] class torch.nn.MSELoss(size_average=True,reduce=True)
>>__Create a criterion that measures the mean squared error between input x and target y.__
>>__The loss can be described as:__
>>>__$l(x,y)=L=\{l_1,\dots,l_N\}{^T}, l_n=(x_n-y_n){^2}$__
>>>__where N is the batch size.__



``` python
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
```

> __class torch.optim.SGD(params,lr=\<object object\>,momentum=0,dampening=0,weight_decay=0,nesterov=False)__
>>__Implements stochastic gradient descent (optionally with momentum).__

>__Parameters:__ 
>>__params(iterable)-iterable of parameters to optimize or dicts defining parameter groups__

>>__lr(float)-learning rate__


### 4. Training Cycle

``` python 
for epoch in range(1, 101):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print("Epoch:", epoch, "loss=", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

y_pred = model(x_data)[^2]
loss = criterion(y_pred, y_data)[^3]
print("Epoch:", epoch, "loss=", loss.item())
optimizer.zero_grad()[^4]
loss.backward()[^5]
optimizer.step()[^6]




[^1]:Also inherit from nn.Module
[^2]:Forwar:Predict
[^3]:Forward:Loss
[^4]:The grad computed by .backward() will be accumulated.
So before backward,remeber set the grad to ZERO!
[^5]:Backward:Autograd
[^6]:Update
