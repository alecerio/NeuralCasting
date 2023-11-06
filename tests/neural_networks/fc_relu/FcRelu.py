import torch.nn as nn

class FcRelu(nn.Module):
    def __init__(self, input_size=2, output_size=3):
        super(FcRelu, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x