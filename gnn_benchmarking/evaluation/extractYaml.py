import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../out/molecules_graph_regression/")
def my_app(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    # x = cfg["params"]
    # print(type(x))

    # print(cfg["params"]["seed"])
    #print()


if __name__ == "__main__":
    # path = "../out/molecules_graph_regression/configs/"
    my_app()