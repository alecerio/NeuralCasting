import torch
import neural_cast as cmp
from neural_cast.compiler import run
from neural_cast.frontend.common.common import CompilerConfig
import os
import yaml
import sys

SIZE = int(sys.argv[1])

curr_file = os.path.abspath(__file__)
curr_path = os.path.dirname(curr_file)
with open(curr_path + '/../../../config/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
CompilerConfig(config)

# init config file
test_path : str = CompilerConfig()['repo'] + 'experiments/gru/neuralcasting/'
path_onnx = test_path + 'M' + str(SIZE) + '.onnx'


# run compiler
run(CompilerConfig(), framework='onnx', path=path_onnx)