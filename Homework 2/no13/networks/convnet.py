import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # input channel 3, output channel 6, kernel 5*5
        self.pool = nn.MaxPool2d(2, 2)  # size of the maxpool
        self.conv2 = nn.Conv2d(6, 16, 5)  # input channel 6, output channel 16, kernel 5*5
        self.fc1 = nn.Linear(256, 120) # input = 16 channel picture size 5*5, output 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x