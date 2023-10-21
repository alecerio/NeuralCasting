import hydra
from compiler.compiler import run

@hydra.main(version_base=None, config_path="config/", config_name="root.yaml")
def main(config):
    run(config)

if __name__ == "__main__":
    main()