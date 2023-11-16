import hydra
from FcAdd import FcAdd
import torch
import compiler as cmp
from compiler.compiler import run
from compiler.frontend.common.common import CompilerConfig
import subprocess
import unittest
from hydra.experimental import compose, initialize
import json

class TestFcAdd(unittest.TestCase):
    def test_00(self):
        # init config file
        name : str = CompilerConfig().name
        output_path : str = CompilerConfig().output_path
        test_path : str = CompilerConfig().test_path
        
        # create pytorch model
        model = FcAdd(2, 2)
        dummy_input = torch.randn(1, 2)

        # load params
        test_path : str = test_path + 'neural_networks/fc_add/'
        params_path = test_path + 'params.pth'
        #torch.save(model.state_dict(), params_path)
        params = torch.load(params_path)
        model.load_state_dict(params)

        # inference pytorch
        input0 = torch.ones(1, 2)
        output_python = model(input0)
        output_shape_python = output_python.shape
        output_python = torch.squeeze(output_python)

        # run compiler
        run(CompilerConfig(), framework='pytorch', model=model, dummy_input=dummy_input, params=params)

        # read inferred output shape
        output_shape_path : str = CompilerConfig().temp_path + "out_shape.json"
        with open(output_shape_path, 'r') as json_file:
            data = json.load(json_file)
            output_keys = list(data.keys())
        output_shape_c = data[output_keys[0]]

        # compare shape
        print("Output shape python: ", output_shape_python)
        print("Output shape C: ", output_shape_c)
        self.assertEqual(len(output_shape_python), len(output_shape_c))
        for i in range(len(output_shape_python)):
            self.assertEquals(output_shape_python[i], output_shape_c[i])

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
    TestFcAdd.config = config
    unittest.main()

if __name__ == "__main__":
    run_tests()