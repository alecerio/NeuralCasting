import torch
import torch.nn as nn
import numpy as np

def torch2onnx(model : nn.Module, dummy_input : np.ndarray, onnx_file_path : str, verbose : bool):
    print("run torch2onnx ...")
    torch.onnx.export(model, dummy_input, onnx_file_path, verbose)
