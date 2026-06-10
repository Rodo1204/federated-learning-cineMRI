"""Hydra configuration utilities for Flower federated runs.

Loads Hydra config from the run context and merges server-sent overrides
into the client config at runtime.
"""
from flwr.app import Context, ConfigRecord
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra

def load_hydra_config(context: Context) -> DictConfig:
    """Load Hydra config from the run context (config_dir + config_name + overrides)."""

    config_dir = context.run_config.get("hydra_config_dir")
    config_name = context.run_config.get("hydra_config_name")
    
    if not config_dir or not config_name:
        raise ValueError(
            "Both hydra_config_dir (default: ./configs) and hydra_config_name must be provided.\n"
            "Example usage: flwr run . --run-config hydra_config_dir=./configs hydra_config_name=config"
        )
    
    # Parse optional Hydra overrides (comma-separated, e.g. "train.lr=1e-3,train.batch_size=64")
    hydra_overrides_raw = context.run_config.get("hydra_overrides", "")
    overrides = [o.strip() for o in hydra_overrides_raw.split(" ") if o.strip()]

    GlobalHydra.instance().clear()  # Clear any existing Hydra instance

    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

def update_config_from_config_record(config: DictConfig, config_record: ConfigRecord) -> DictConfig:
    """Merge server-sent overrides from ConfigRecord into the Hydra config."""
    # TODO: May need to add some validations to ensure that only expected keys are updated

    server_config_dict = dict(config_record)

    OmegaConf.set_struct(config, False)  # Temporarily disable struct mode to allow merging
    
    for key, value in server_config_dict.items():
        if key not in ["federated_timestamp", "server-round", "optimizer_steps_this_round" ]:  # exclude metadata keys
            OmegaConf.update(config, key, value, merge=True)

    OmegaConf.set_struct(config, True)  # Re-enable struct mode after merging

    return config