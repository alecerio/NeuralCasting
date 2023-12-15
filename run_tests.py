from tests.neural_networks.run_tests_wrapper import run_neural_network_tests
from tests.onnxdbg.run_tests_wrapper import run_onnxdbg_tests

def main():
    run_neural_network_tests()
    run_onnxdbg_tests()

if __name__ == '__main__':
    main()