import hydra
from compiler.compiler import run
from examples.neural_networks.pytorch.dummy_nn import DummyNN
from examples.neural_networks.pytorch.dummy_nn import DummyNN2
import torch

@hydra.main(version_base=None, config_path="config/", config_name="root.yaml")
def main(config):
    print(config)

    model = DummyNN(2, 3, 2)
    #model = DummyNN2(2, 3)
    dummy_input = torch.randn(1, 2)

    input0 = torch.ones(1, 2)
    output0 = model(input0)
    print("Output: ", output0)

    #state_dict = model.state_dict()
    #for name, param in state_dict.items():
    #    print(f"Parameter: {name}, Size: {param.tolist()}")

    run(config, model, dummy_input)

if __name__ == "__main__":
    main()