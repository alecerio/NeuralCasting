from neural_cast.frontend.common.common import CompilerConfig
import os
import yaml
from tests.neural_networks.constant.main_test import TestConstant
from tests.neural_networks.fc_add.main_test import TestFcAdd
from tests.neural_networks.fc_mul.main_test import TestFcMul
import unittest

def run_neural_network_tests():
    curr_file = os.path.abspath(__file__)
    curr_path = os.path.dirname(curr_file)
    with open(curr_path + '/../../config/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    CompilerConfig(config)
    
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestConstant))
    test_suite.addTest(unittest.makeSuite(TestFcAdd))
    test_suite.addTest(unittest.makeSuite(TestFcMul))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)