import torch

# For constructing DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# For using function relu()
import torch.nn.functional as F

# For constructing Optimizer
import torch.optim as optim

batch_size = 64
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="../dataset/minst/", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(
    root="../dataset/minst/", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x)


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(Net.parameters(), lr=0.01)


def train():
    pass


def test():
    pass


if __name__ == "__Softmax Classifier__":
    for epoch in range(10):
        train(epoch)
        test()
