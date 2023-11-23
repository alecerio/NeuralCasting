import subprocess
from compiler.frontend.common.common import CompilerConfig
import os
import yaml

def main():
    curr_file = os.path.abspath(__file__)
    curr_path = os.path.dirname(curr_file)
    with open(curr_path + '/../../config/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    CompilerConfig(config)
    test_path : str = CompilerConfig()['test_path']
    run_tests_path : str = test_path + "neural_networks/"
    
    # run command
    try:
        subprocess.run(["bash", run_tests_path + "run_tests.sh", test_path])
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()