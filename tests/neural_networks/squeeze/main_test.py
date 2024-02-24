from neural_cast.compiler import run
from neural_cast.frontend.common.common import CompilerConfig
import subprocess
import unittest
import numpy as np
from tests.common.common import inference_onnx_runtime, create_main_c, read_inferred_output_shape, read_output_c, print_test_header, compare_shape, compare_results

class TestSqueeze(unittest.TestCase):
    def test_00(self):

        print_test_header("SQUEEZE ONNX", 4)

        # init config file
        name : str = CompilerConfig()['name']
        output_path : str = CompilerConfig()['output_path']
        test_path : str = CompilerConfig()['test_path'] + 'neural_networks/squeeze/'
        temp_path : str = CompilerConfig()['temp_path']
        path_onnx = test_path + 'squeeze_model.onnx'

        # inference onnxruntime
        input_data = np.array(
            [[
                [[1, 2, 3, 4]],
                [[5, 6, 7, 8]],
                [[9, 10, 11, 12]]
            ]], dtype=np.float32)

        #input_data = np.zeros([1, 3, 1, 4], dtype=np.float32)
        [outputs_onnx, outputs_shape_onnx] = inference_onnx_runtime(path_onnx, [input_data])
        output_onnx = outputs_onnx[0]
        output_shape_onnx = outputs_shape_onnx[0]

        print(output_onnx)
        print(output_shape_onnx)

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
        print("Output ONNX")
        print(output_onnx)
        print("Output C")
        print(output_c)
        compare_results(self, output_onnx, output_c, "ONNX", "C", 1e-6)