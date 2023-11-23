import os
import yaml
from compiler.compiler import run
from examples.neural_networks.pytorch.dummy_nn import DummyNN
from examples.neural_networks.pytorch.dummy_nn import DummyNN2
from examples.neural_networks.pytorch.dummy_nn import DummyNN3
from examples.neural_networks.pytorch.dummy_nn import DummyGRU
import torch

def main():
    curr_file = os.path.abspath(__file__)
    curr_path = os.path.dirname(curr_file)
    
    with open(curr_path + '/config/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    print(config)

    model = DummyNN3(2, 3)
    dummy_input = torch.randn(1, 2)

    params_path = curr_path + '/examples/params/pytorch/'
    params_path += 'dummy_nn_params_3.pth'
    torch.save(model.state_dict(), params_path)

    params = torch.load(params_path)
    model.load_state_dict(params)

    run(config, framework='pytorch', model=model, dummy_input=dummy_input, params=params)

if __name__ == "__main__":
    main()