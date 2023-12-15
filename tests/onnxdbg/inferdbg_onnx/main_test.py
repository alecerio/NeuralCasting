import unittest
from neural_cast.frontend.common.common import CompilerConfig
from tests.common.common import print_test_header, inference_onnx_runtime, compare_shape, compare_results
import os
import yaml
import onnx
import pickle
import numpy as np
from onnxdbg.onnxdbg import onnxdbg

class TestInferDbgOnnx(unittest.TestCase):
    def test_00(self):
        print_test_header("INFERDBG ONNX", 4)

        # init config file
        onnx_name : str = 'gru_reimplemented_1'
        test_path : str = CompilerConfig()['test_path'] + 'onnxdbg/inferdbg_onnx/'
        output_path : str = test_path + '/output/'
        temp_path : str = CompilerConfig()['temp_path']

        # input data
        model = onnx.load(test_path + onnx_name + '.onnx')
        input_names = [input.name for input in model.graph.input]
        x_data =  np.ones((1, 3), dtype=np.float32)
        hidden_data =  np.zeros((1, 4), dtype=np.float32)
        input_data = {
            input_names[0]: x_data,
            input_names[1]: hidden_data
        }

        # run command inferdbg
        onnxdbg("inferdbg", srcp=test_path, dstp=output_path, mdl=onnx_name, input=input_data)

        with open(output_path + 'output.pkl', 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        expected_output_31 = [np.array([-0.5007976 , -0.41972914,  0.13484477, -0.40547925], dtype=np.float32)]
        actual_output_31 = data['31']

        with open(output_path + 'output_shape.pkl', 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        expected_output_shape_31 = [np.array([1.0, 4.0], dtype=np.float32)]
        actual_output_shape_31 = data['31']

        # compare results
        expected_output_31 = [arr.tolist() for arr in expected_output_31]
        actual_output_31 = [arr.tolist() for arr in actual_output_31]
        compare_results(self, expected_output_31, actual_output_31, "EXPECTED OUTPUT", "ACTUAL OUTPUT", 1e-6)

        # compare shapes
        expected_output_shape_31 = [arr.tolist() for arr in expected_output_shape_31]
        expected_output_shape_31 = expected_output_shape_31[0]
        actual_output_shape_31 = list(actual_output_shape_31[0])
        compare_results(self, expected_output_shape_31, actual_output_shape_31, "EXPECTED OUTPUT SHAPE", "ACTUAL OUTPUT SHAPE", 1e-6)

def run_tests():
    curr_file = os.path.abspath(__file__)
    curr_path = os.path.dirname(curr_file)
    with open(curr_path + '/../../../config/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    CompilerConfig(config)
    unittest.main()

if __name__ == "__main__":
    run_tests()