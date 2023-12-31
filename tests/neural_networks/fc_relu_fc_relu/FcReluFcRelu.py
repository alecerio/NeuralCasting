import torch.nn as nn

class FcReluFcRelu(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=3):
        super(FcReluFcRelu, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu2 = nn.ReLU()

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x