import os
import yaml
import torch
from neural_cast.compiler import run

# load configuration
curr_file = os.path.abspath(__file__)
curr_path = os.path.dirname(curr_file)   
with open(curr_path + '/../../config/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# run compiler
run(config, framework='onnx', path=curr_path + '/model.onnx')

# check generated code in the output folder defined in config.yaml