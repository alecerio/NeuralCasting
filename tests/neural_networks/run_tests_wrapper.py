import hydra
import subprocess
from compiler.frontend.common.common import CompilerConfig

@hydra.main(version_base=None, config_path="../../config/", config_name="root.yaml")
def main(config):
    CompilerConfig(config)
    test_path : str = CompilerConfig().test_path
    run_tests_path : str = test_path + "neural_networks/"
    
    # run command
    try:
        subprocess.run(["bash", run_tests_path + "run_tests.sh", test_path])
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()