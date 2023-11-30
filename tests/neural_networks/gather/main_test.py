from neural_cast.compiler import run
from neural_cast.frontend.common.common import CompilerConfig
import subprocess
import unittest
import numpy as np
from tests.common.common import inference_onnx_runtime, create_main_c, read_inferred_output_shape, read_output_c, print_test_header, compare_shape, compare_results

class TestGather(unittest.TestCase):
    def test_00(self):

        print_test_header("GATHER AXIS 1 ONNX", 4)

        # init config file
        name : str = CompilerConfig()['name']
        output_path : str = CompilerConfig()['output_path']
        test_path : str = CompilerConfig()['test_path'] + 'neural_networks/gather/'
        temp_path : str = CompilerConfig()['temp_path']
        path_onnx = test_path + 'gather_axis1.onnx'

        # inference onnxruntime
        values =  np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ], dtype=np.float32)
        indices = np.array([
            [0, 2],
            [0, 2],
            [0, 2],
        ], dtype=np.int64)
        input_data = [values, indices]
        [outputs_onnx, outputs_shape_onnx] = inference_onnx_runtime(path_onnx, input_data)
        output_onnx = outputs_onnx[0]
        output_shape_onnx = outputs_shape_onnx[0]

        # run compiler
        run(CompilerConfig(), framework='onnx', path=path_onnx)

        # create test main.c
        create_main_c(test_path, output_path, name, main_name='main_axis1.c')

        # run command
        try:
            subprocess.run(["bash", test_path + "build_axis1.sh", name, output_path], check=True)
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
    
    def test_01(self):

        print_test_header("GATHER AXIS 0 ONNX", 4)

        # init config file
        name : str = CompilerConfig()['name']
        output_path : str = CompilerConfig()['output_path']
        test_path : str = CompilerConfig()['test_path'] + 'neural_networks/gather/'
        temp_path : str = CompilerConfig()['temp_path']
        path_onnx = test_path + 'gather_axis0.onnx'

        # inference onnxruntime
        values =  np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ], dtype=np.float32)
        indices = np.array([
            [0, 2],
            [0, 2],
            [0, 2],
        ], dtype=np.int64)
        input_data = [values, indices]
        [outputs_onnx, outputs_shape_onnx] = inference_onnx_runtime(path_onnx, input_data)
        output_onnx = outputs_onnx[0]
        output_shape_onnx = outputs_shape_onnx[0]

        # run compiler
        run(CompilerConfig(), framework='onnx', path=path_onnx)

        # create test main.c
        create_main_c(test_path, output_path, name, main_name='main_axis0.c')

        # run command
        try:
            subprocess.run(["bash", test_path + "build_axis0.sh", name, output_path], check=True)
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