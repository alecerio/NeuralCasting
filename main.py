import os
import hydra
from compiler.compiler import run
from examples.neural_networks.pytorch.dummy_nn import DummyNN
from examples.neural_networks.pytorch.dummy_nn import DummyNN2
from examples.neural_networks.pytorch.dummy_nn import DummyNN3
import torch

@hydra.main(version_base=None, config_path="config/", config_name="root.yaml")
def main(config):
    print(config)

    #model = DummyNN(2, 3, 2)
    #model = DummyNN2(2, 3)
    model = DummyNN3(2, 3)
    dummy_input = torch.randn(1, 2)

    curr_file = os.path.abspath(__file__)
    curr_path = os.path.dirname(curr_file)
    params_path = curr_path + '/examples/params/pytorch/'
    params_path += 'dummy_nn_params_3.pth'
    torch.save(model.state_dict(), params_path)

    params = torch.load(params_path)
    model.load_state_dict(params)

    input0 = torch.ones(1, 2)
    output0 = model(input0)
    print("Output: ", output0)

    #state_dict = model.state_dict()
    #for name, param in state_dict.items():
    #    print(f"Parameter: {name}, Size: {param.tolist()}")

    run(config, model, dummy_input, params)

if __name__ == "__main__":
    main()