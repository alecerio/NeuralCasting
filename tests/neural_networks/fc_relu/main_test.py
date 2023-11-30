from .FcRelu import FcRelu
import torch
from neural_cast.compiler import run
from neural_cast.frontend.common.common import CompilerConfig
import subprocess
import unittest
import numpy as np
from tests.common.common import inference_pytorch, inference_onnx_runtime, create_main_c, read_inferred_output_shape, read_output_c, print_test_header, compare_shape, compare_results


class TestFcRelu(unittest.TestCase):
    def test_00(self):

        print_test_header("FULLY_CONNECTED-RELU PYTORCH", 4)

        # init config file
        name : str = CompilerConfig()['name']
        output_path : str = CompilerConfig()['output_path']
        test_path : str = CompilerConfig()['test_path'] + 'neural_networks/fc_relu/'
        temp_path : str = CompilerConfig()['temp_path']
        params_path = test_path + 'params.pth'

        # create pytorch model
        model = FcRelu(2, 3)
        dummy_input = torch.randn(1, 2)
        params = torch.load(params_path)
        model.load_state_dict(params)

        # inference pytorch
        input0 = torch.ones(1, 2)
        [output_python, output_shape_python] = inference_pytorch(model, input0)

        # run compiler
        run(CompilerConfig(), framework='pytorch', model=model, dummy_input=dummy_input, params=params)

        # create test main.c
        create_main_c(test_path, output_path, name)

        # run command
        try:
            subprocess.run(["bash", test_path + "build.sh", name, output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

        # read inferred output shape
        output_shape_c = read_inferred_output_shape(temp_path)

        # read c output
        output_c = read_output_c(output_path)

        # compare shape
        compare_shape(self, output_shape_python, output_shape_c, "PyTorch", "C")

        # compare results
        compare_results(self, output_python, output_c, "PyTorch", "C", 1e-6)
    
    def test_01(self):

        print_test_header("FULLY_CONNECTED-RELU ONNX", 4)

        # init config file
        name : str = CompilerConfig()['name']
        output_path : str = CompilerConfig()['output_path']
        test_path : str = CompilerConfig()['test_path'] + 'neural_networks/fc_relu/'
        temp_path : str = CompilerConfig()['temp_path']
        path_onnx = test_path + 'FcRelu.onnx'

        # inference onnxruntime
        input_data = np.ones((1, 2), dtype=np.float32)
        [outputs_onnx, outputs_shape_onnx] = inference_onnx_runtime(path_onnx, [input_data])
        output_onnx = outputs_onnx[0]
        output_shape_onnx = outputs_shape_onnx[0]

        # run compiler
        run(CompilerConfig(), framework='onnx', path=path_onnx)

        # create test main.c
        create_main_c(test_path, output_path, name)

        # run command
        try:
            subprocess.run(["bash", test_path + "build.sh", name, output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        
        # read inferred output shape
        output_shape_c = read_inferred_output_shape(temp_path)

        # read c output
        output_c = read_output_c(output_path)

        # compare shape
        compare_shape(self, output_shape_onnx, output_shape_c, "ONNX", "C")

        # compare results
        compare_results(self, output_onnx, output_c, "ONNX", "C", 1e-6)