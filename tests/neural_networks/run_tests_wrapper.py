from neural_cast.frontend.common.common import CompilerConfig
import os
import yaml
from tests.neural_networks.constant.main_test import TestConstant
import unittest

def run_neural_network_tests():
    curr_file = os.path.abspath(__file__)
    curr_path = os.path.dirname(curr_file)
    with open(curr_path + '/../../config/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    CompilerConfig(config)
    test_path : str = CompilerConfig()['test_path']
    run_tests_path : str = test_path + "neural_networks/"
    
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestConstant))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)