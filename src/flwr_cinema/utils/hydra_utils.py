from flwr.common import Context
from hydra import compose, initialize
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

def load_hydra_config(context: Context) -> DictConfig:
    """Extract, validate, and load the Hydra configuration specified in the run config based on flwr context."""

    config_dir = context.run_config.get("hydra_config_dir")
    config_name = context.run_config.get("hydra_config_name")
    
    if not config_dir or not config_name:
        raise ValueError(
            "Both config_dir (default: ./configs) and config_name must be provided.\n"
            "Example usage: flwr run . --run-config hydra_config_dir=./configs hydra_config_name=config"
        )
    
    GlobalHydra.instance().clear()  # Clear any existing Hydra instance

    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name=config_name)
    return cfg