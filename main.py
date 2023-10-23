import hydra
from compiler.compiler import run
from examples.neural_networks.pytorch.dummy_nn import DummyNN
import torch

@hydra.main(version_base=None, config_path="config/", config_name="root.yaml")
def main(config):
    print(config)

    model = DummyNN(3, 16, 3)
    dummy_input = torch.randn(1, 3)

    run(config, model, dummy_input)

if __name__ == "__main__":
    main()