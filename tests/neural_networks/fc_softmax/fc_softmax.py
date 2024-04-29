import torch.nn as nn

class FcSoftmax(nn.Module):
    def __init__(self, input_size=2, output_size=3):
        super(FcSoftmax, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x
