import torch.nn as nn

class FcSigmoid(nn.Module):
    def __init__(self, input_size=2, output_size=3):
        super(FcSigmoid, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x