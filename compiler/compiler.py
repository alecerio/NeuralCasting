import hydra
import numpy as np
import torch.nn as nn
from compiler.frontend.torch2onnx import torch2onnx

def run(config, model, dummy_input):
    print("run compiler ...")
    fr = config.framework
    if(fr.framework_name == 'pytorch'):
        temp_path : str = str(fr.temp_path)
        name : str = str(fr.name)
        path = temp_path + name + '.onnx'
        verbose : bool = bool(fr.verbose)
        torch2onnx(model, dummy_input, path, verbose)
    