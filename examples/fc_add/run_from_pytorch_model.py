import os
import yaml
import torch
from neural_cast.compiler import run
from FcAdd import FcAdd

# load configuration
curr_file = os.path.abspath(__file__)
curr_path = os.path.dirname(curr_file)   
with open(curr_path + '/../../config/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# create model
model = FcAdd()

# create dummy input
dummy_input = torch.randn(1, 2)

# upload model weights
params = torch.load(curr_path + '/weights.pth')

# run compiler
run(config, framework='pytorch', model=model, dummy_input=dummy_input, params=params)

# check generated code in the output folder defined in config.yaml