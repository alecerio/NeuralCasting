import os
import yaml
import unittest
from neural_cast.frontend.common.common import CompilerConfig
from tests.onnxdbg.copy_onnx.main_test import TestCopyOnnx
from tests.onnxdbg.subgraph_onnx.main_test import TestSubgraphOnnx

def run_onnxdbg_tests():
    curr_file = os.path.abspath(__file__)
    curr_path = os.path.dirname(curr_file)
    with open(curr_path + '/../../config/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    CompilerConfig(config)
    
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestCopyOnnx))
    test_suite.addTest(unittest.makeSuite(TestSubgraphOnnx))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)