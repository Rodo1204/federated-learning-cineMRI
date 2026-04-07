from flwr.app import Context, ConfigRecord
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra

def load_hydra_config(context: Context) -> DictConfig:
    """Extract, validate, and load the Hydra configuration specified in the run config based on flwr context."""

    config_dir = context.run_config.get("hydra_config_dir")
    config_name = context.run_config.get("hydra_config_name")
    
    if not config_dir or not config_name:
        raise ValueError(
            "Both hydra_config_dir (default: ./configs) and hydra_config_name must be provided.\n"
            "Example usage: flwr run . --run-config hydra_config_dir=./configs hydra_config_name=config"
        )
    
    GlobalHydra.instance().clear()  # Clear any existing Hydra instance

    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name=config_name)
    return cfg

def update_config_from_config_record(config: DictConfig, config_record: ConfigRecord) -> DictConfig:
    """Update the given Hydra config with values from the provided ConfigRecord, allowing for server-sent overrides."""
    # TODO: May need to add some validations to ensure that only expected keys are updated

    server_config_dict = dict(config_record)

    OmegaConf.set_struct(config, False)  # Temporarily disable struct mode to allow merging
    
    for key, value in server_config_dict.items():
        if key != "federated_timestamp" and key != "round":  # Exclude certain keys that we don't want to merge into the config
            OmegaConf.update(config, key, value, merge=True)

    OmegaConf.set_struct(config, True)  # Re-enable struct mode after merging

    return config