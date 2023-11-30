import torch
import neural_cast as cmp
from neural_cast.compiler import run
from neural_cast.frontend.common.common import CompilerConfig
import subprocess
import unittest
import numpy as np
from tests.common.common import inference_onnx_runtime, create_main_c, read_inferred_output_shape, read_output_c, print_test_header

class TestConstant(unittest.TestCase):
    def test_00(self):

        print_test_header("CONSTANT ONNX", 4)

        # init config file
        name : str = CompilerConfig()['name']
        output_path : str = CompilerConfig()['output_path']
        test_path : str = CompilerConfig()['test_path'] + 'neural_networks/constant/'
        temp_path : str = CompilerConfig()['temp_path']
        path_onnx = test_path + 'constant.onnx'

        # inference onnxruntime
        input_data = np.ones((4, 1), dtype=np.float32)
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
        print("Output shape onnx: ", output_shape_onnx)
        print("Output shape C: ", output_shape_c)
        self.assertEqual(len(output_shape_onnx), len(output_shape_c))
        for i in range(len(output_shape_onnx)):
            self.assertEquals(output_shape_onnx[i], output_shape_c[i])

        # compare results
        print("Output onnx: ", output_onnx)
        print("Output C: ", output_c)
        self.assertEqual(len(output_onnx), len(output_c))
        N : int = len(output_onnx)
        for i in range(N):
            self.assertAlmostEqual(output_onnx[i], output_c[i], delta=1e-6)