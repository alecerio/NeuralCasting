import os
import yaml
import unittest
from neural_cast.frontend.common.common import CompilerConfig
from tests.neural_networks.constant.main_test import TestConstant
from tests.neural_networks.fc_add.main_test import TestFcAdd
from tests.neural_networks.fc_mul.main_test import TestFcMul
from tests.neural_networks.fc_relu.main_test import TestFcRelu
from tests.neural_networks.fc_relu_fc_relu.main_test import TestFcReluFcRelu
from tests.neural_networks.fc_sigmoid.main_test import TestFcSigmoid
from tests.neural_networks.fc_sub.main_test import TestFcSub
from tests.neural_networks.fc_tanh.main_test import TestFcTanh
from tests.neural_networks.gather.main_test import TestGather

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
    test_suite.addTest(unittest.makeSuite(TestFcRelu))
    test_suite.addTest(unittest.makeSuite(TestFcReluFcRelu))
    test_suite.addTest(unittest.makeSuite(TestFcSigmoid))
    test_suite.addTest(unittest.makeSuite(TestFcSub))
    test_suite.addTest(unittest.makeSuite(TestFcTanh))
    test_suite.addTest(unittest.makeSuite(TestGather))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)