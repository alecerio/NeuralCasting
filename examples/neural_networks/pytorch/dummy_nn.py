import torch
import torch.nn as nn

class DummyNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(DummyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        print("FC1: ", x)
        x = self.relu(x)
        print("RELU: ", x)
        x = self.fc2(x)
        print("FC2: ", x)
        return x


class DummyNN2(nn.Module):
    def __init__(self, input_size=2, output_size=3):
        super(DummyNN2, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

class DummyNN3(nn.Module):
    def __init__(self, input_size=2, output_size=3):
        super(DummyNN3, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        print(x)
        x = self.fc1(x)
        print(x)
        x = self.sigmoid(x)
        print(x)
        return x