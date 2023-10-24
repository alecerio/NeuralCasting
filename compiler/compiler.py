import hydra
import numpy as np
import torch.nn as nn
from compiler.frontend.torch2onnx.torch2onnx import torch2onnx
from compiler.frontend.parser.parser.parser import parse
from compiler.frontend.parser.node.node import Node

def run(config, model, dummy_input):
    print("run compiler ...")

    # lower from python framework to onnx
    fr = config.framework
    if(fr.framework_name == 'pytorch'):
        temp_path : str = str(fr.temp_path)
        name : str = str(fr.name)
        path = temp_path + name + '.onnx'
        verbose : bool = bool(fr.verbose)
        torch2onnx(model, dummy_input, path, verbose)
    else:
        raise Exception("Error: unexpected framework")
    
    # parse onnx
    nodes : list[Node] = parse(config)

    # create dag
    