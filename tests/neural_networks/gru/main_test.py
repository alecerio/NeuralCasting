import torch
import neural_cast as cmp
from neural_cast.compiler import run
from neural_cast.frontend.common.common import CompilerConfig
import subprocess
import unittest
import onnxruntime as ort
import numpy as np
import os
import yaml
from tests.common.common import inference_onnx_runtime, create_main_c, read_inferred_output_shape, read_output_c, print_test_header, compare_shape, compare_results

class TestGRU(unittest.TestCase):
    def test_00(self):

        print_test_header("GRU 00 ONNX", 4)

        # init config file
        name : str = CompilerConfig()['name']
        output_path : str = CompilerConfig()['output_path']
        test_path : str = CompilerConfig()['test_path'] + 'neural_networks/gru/'
        temp_path : str = CompilerConfig()['temp_path']
        path_onnx = test_path + 'gru.onnx'

        # inference onnxruntime
        #x_data =  np.ones((1, 3), dtype=np.float32)
        #hidden_data =  np.zeros((1, 4), dtype=np.float32)
        input_size = 10
        hidden_size = 20
        batch_size = 1
        sequence_length = 5
        x_data = np.random.rand(batch_size, sequence_length, input_size).astype(np.float32)
        initial_hidden_state = np.random.rand(1, batch_size, hidden_size).astype(np.float32)
        input_data = [x_data, initial_hidden_state]
        [outputs_onnx, outputs_shape_onnx] = inference_onnx_runtime(path_onnx, input_data)
        output_onnx = outputs_onnx[0]
        output_shape_onnx = outputs_shape_onnx[0]

        # run compiler
        run(CompilerConfig(), framework='onnx', path=path_onnx)

        # create test main.c
        create_main_c(test_path, output_path, name, main_name='main.c')
        
        # run command
        #try:
        #    subprocess.run(["bash", test_path + "build.sh", name, output_path], check=True)
        #except subprocess.CalledProcessError as e:
        #    print(f"Error: {e}")

        # read inferred output shape
        #output_shape_c = read_inferred_output_shape(temp_path)
        
        # read c output
        #output_c = read_output_c(output_path)

        # compare shape
        #compare_shape(self, output_shape_onnx, output_shape_c, "ONNX", "C")

        # compare results
        #compare_results(self, output_onnx, output_c, "ONNX", "C", 1e-6)