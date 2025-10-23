
## [Total Code](Dataset%20and%20DataLoader.py)
``` python

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DiabetesDataset(torch.nn.Module):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

```

``` python
class DiabetesDataset(torch.nn.Module):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

```
>DataLoader is a class to help us loading data in Pytorch
>>DiabetesDataset is inherited from abstract class Dataset

The expression,dataset[index],will call this magic function.