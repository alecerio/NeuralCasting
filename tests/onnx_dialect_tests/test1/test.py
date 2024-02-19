import os
from neural_cast.frontend.common.common import CompilerConfig
import yaml
import torch
from neural_cast.compiler import run
import torch.nn as nn

class FcAdd(nn.Module):
    def __init__(self, input_size=2, output_size=3):
        super(FcAdd, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x_input = x
        x = self.fc(x)
        x = x + x_input
        return x

# setup
curr_file = os.path.abspath(__file__)
curr_path = os.path.dirname(curr_file)
with open(curr_path + '/../config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
CompilerConfig(config)

# init config file
name : str = CompilerConfig()['name']
output_path : str = CompilerConfig()['output_path']
test_path : str = CompilerConfig()['test_path'] + 'neural_networks/fc_add/'
temp_path : str = CompilerConfig()['temp_path']
params_path = test_path + 'params.pth'

# create pytorch model
model = FcAdd(2, 2)
dummy_input = torch.randn(1, 2)
params = torch.load(params_path)
model.load_state_dict(params)

# run compiler
run(CompilerConfig(), framework='pytorch', model=model, dummy_input=dummy_input, params=params)