import unittest
import onnxruntime as ort
from neural_cast.frontend.common.common import CompilerConfig
from tests.common.common import print_test_header, inference_onnx_runtime, compare_shape, compare_results
import os
import yaml
import numpy as np
from onnxdbg.onnxdbg import onnxdbg

class TestSubgraphOnnx(unittest.TestCase):
    def test_00(self):
        print_test_header("SUBGRAPH ONNX", 4)

        # init config file
        test_path : str = CompilerConfig()['test_path'] + 'onnxdbg/subgraph_onnx/'
        temp_path : str = CompilerConfig()['temp_path']
        path_onnx = test_path + 'gru_reimplemented_1.onnx'
        path_onnx_dest = temp_path + 'gru_reimplemented_1.onnx'

        # expected results
        expected_shape = [1, 4]
        expected_output = [-0.90645695, -0.7594772, 0.23172721, -0.7083346]

        # re-create onnx with onnxdbg
        onnxdbg("subgr", src=path_onnx, dst=path_onnx_dest, mdl='sub_model', out='/Tanh')

        # inference onnxruntime subgraph
        x_data =  np.ones((1, 3), dtype=np.float32)
        hidden_data =  np.zeros((1, 4), dtype=np.float32)
        input_data = [x_data, hidden_data]
        [outputs_onnx_copy, outputs_shape_onnx_copy] = inference_onnx_runtime(path_onnx_dest, input_data)
        output_onnx_subgraph = outputs_onnx_copy[0]
        output_shape_onnx_subgraph = outputs_shape_onnx_copy[0]

        # compare shape
        compare_shape(self, expected_shape, output_shape_onnx_subgraph, "ONNX", "ONNX SUB")

        # compare results
        compare_results(self, expected_output, output_onnx_subgraph, "ONNX", "ONNX SUB", 1e-6)

def run_tests():
    curr_file = os.path.abspath(__file__)
    curr_path = os.path.dirname(curr_file)
    with open(curr_path + '/../../../config/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    CompilerConfig(config)
    unittest.main()

if __name__ == "__main__":
    run_tests()