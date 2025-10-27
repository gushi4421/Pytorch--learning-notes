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

print(f"w={model.linear.weight},b={model.linear.bias}")

criterion = torch.nn.MSELoss(size_average=False)  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器


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
