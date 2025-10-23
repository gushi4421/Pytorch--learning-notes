### Total Code:
``` python
import torch.nn.functional as F
import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])


class LogisticRegressionModel(torch.nn.Module):
    def __init___(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1, 1001):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print("Epoch:", epoch, "loss=", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # update parameters


```