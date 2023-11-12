import hydra
from FcSub import FcSub
import torch
import compiler as cmp
from compiler.compiler import run
from compiler.frontend.common.common import CompilerConfig
import subprocess
import unittest
from hydra.experimental import compose, initialize

class TestFcSub(unittest.TestCase):
    def test_00(self):
        # init config file
        name : str = CompilerConfig().name
        output_path : str = CompilerConfig().output_path
        test_path : str = CompilerConfig().test_path
        
        # create pytorch model
        model = FcSub(2, 2)
        dummy_input = torch.randn(1, 2)

        # load params
        test_path : str = test_path + 'neural_networks/fc_sub/'
        params_path = test_path + 'params.pth'
        #torch.save(model.state_dict(), params_path)
        params = torch.load(params_path)
        model.load_state_dict(params)

        # inference pytorch
        input0 = torch.ones(1, 2)
        output_python = model(input0)
        output_python = torch.squeeze(output_python)

        # run compiler
        run(CompilerConfig(), model, dummy_input, params)

        # read main.c code and add include to nn
        f = open(test_path + 'main.c', 'r')
        main_code : str = "#include \"" + name + ".h\"\n"
        main_code += f.read()
        f.close()

        # generate main.c in output directory
        f = open(output_path + 'main.c', 'w')
        f.write(main_code)
        f.close()
        
        # run command
        try:
            subprocess.run(["bash", test_path + "build.sh", name, output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

        # read c output
        f = open(output_path + "test_output.txt")
        output_text : str = f.read()
        f.close()
        output_values_str : list[str] = output_text.split(" ")
        output_c : list[float] = []
        for i in range(len(output_values_str)):
            output_c.append(float(output_values_str[i]))

        print(output_python)
        print(output_c)
        # compare results
        self.assertEqual(len(output_python), len(output_c))

        N : int = len(output_python)
        for i in range(N):
            self.assertAlmostEqual(output_python[i], output_c[i], delta=1e-6)

def run_tests():
    initialize(config_path="../../../config/")
    config = compose(config_name="root.yaml")
    CompilerConfig(config)
    TestFcSub.config = config
    unittest.main()

if __name__ == "__main__":
    run_tests()