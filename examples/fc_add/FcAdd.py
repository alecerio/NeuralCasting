import torch
import torch.nn as nn

class FcAdd(nn.Module):
    def __init__(self, input_size=2, output_size=2):
        super(FcAdd, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x_input = x
        x = self.fc(x)
        x = x + x_input
        return x