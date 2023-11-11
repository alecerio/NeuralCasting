import torch.nn as nn

class FcTanh(nn.Module):
    def __init__(self, input_size=2, output_size=3):
        super(FcTanh, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.tanh(x)
        return x