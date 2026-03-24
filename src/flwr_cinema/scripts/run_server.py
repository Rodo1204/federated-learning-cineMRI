"""Script to run federated learning server."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime
import torch
import hydra
from omegaconf import DictConfig
from models.cinema.cinema.convvit import ConvViT
from models.cinema.cinema.device import get_amp_dtype_and_device  # noqa: E402
from models.cinema.cinema.optim import get_n_accum_steps  # noqa: E402
from flwr_cinema.scripts.lora import apply_lora_to_convvit, apply_dual_lora_to_convvit  # noqa: E402


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create federated_server subdirectory for server logs
    server_log_dir = log_dir / "federated_server"
    server_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger instead of just __main__
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    log_file = server_log_dir / "server.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Initialize logger with default setup (will be updated after output_dir is set)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_server_class(config: DictConfig):
    """Dynamically import FederatedServer based on task type.
    
    Args:
        config: Configuration with config.task field.
        
    Returns:
        FederatedServer class from appropriate module (classification or regression).
    """
    task = getattr(config, "task", "regression")
    
    if task == "classification":
        from cinema_classification.federated.server import FederatedServer
        logger.info("Using classification federated server")
    else:
        from cinema_regression.federated.server import FederatedServer
        logger.info("Using regression federated server")
    
    return FederatedServer

def get_model(config) -> torch.nn.Module:
    """Initialize model with optional LoRA (single or dual) using CineMA pattern."""
    # Load pretrained model: freeze backbone unless doing full finetuning
    freeze_backbone = config.model.get('freeze_pretrained', True) or config.train.use_lora
    
    logger.info(f"Loading pretrained model (freeze_backbone={freeze_backbone})...")
    model = ConvViT.from_pretrained(config=config, freeze=freeze_backbone)

    # Apply LoRA if enabled (using lora.blocks_to_lora for both standard and dual LoRA)
    blocks_to_lora = getattr(config.lora, 'blocks_to_lora', None)
    
    if config.train.use_lora:
        # Check if Dual LoRA is enabled
        if getattr(config, 'dual_lora', None) and getattr(config.dual_lora, 'enabled', False):
            logger.info("Applying Dual LoRA adapters (personalized federated learning)...")
            apply_dual_lora_to_convvit(
                model=model,
                global_rank=config.dual_lora.global_rank,
                local_rank=config.dual_lora.local_rank,
                alpha=config.dual_lora.alpha,
                dropout=config.dual_lora.dropout,
                target_modules=config.dual_lora.target_modules,
                blocks_to_lora=blocks_to_lora,
            )
        else:
            logger.info("Applying standard LoRA adapters...")
            apply_lora_to_convvit(
                model=model,
                rank=config.lora.rank,
                alpha=config.lora.alpha,
                dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
                blocks_to_lora=blocks_to_lora,
            )
    elif config.model.get('freeze_pretrained', False):
        # Head-only training: unfreeze classification head while keeping backbone frozen
        logger.info("Head-only training mode: unfreezing classification head...")
        if hasattr(model, 'pred_head_dict'):
            for head_param in model.pred_head_dict.parameters():
                head_param.requires_grad = True
    
    return model



def get_primary_metric_name(config) -> str:
    """Get the primary metric name for early stopping from config.
    
    Args:
        config: Configuration with config.train.early_stopping.metric field.
        
    Returns:
        Primary metric name (e.g., 'mae' for regression, 'accuracy' for classification).
    """
    # Read metric from config to align with centralized training
    if hasattr(config, "train") and hasattr(config.train, "early_stopping") and hasattr(config.train.early_stopping, "metric"):
        return config.train.early_stopping.metric
    
    # Fallback: derive from task if metric not specified in config
    task = getattr(config, "task", "regression")
    if task == "classification":
        return "accuracy"
    else:
        return "mae"

@hydra.main(version_base=None, config_path="../configs", config_name="cinema_fed.yaml")
def main(config: DictConfig) -> None:
    """Main function to run federated server with Hydra config only."""
    # Construct output directory using config template structure
    # Use coordinated timestamp from environment if available (set by run_federated.sh)
    # Otherwise generate new timestamp
    import os
    coordinated_timestamp = os.environ.get("FEDERATED_TIMESTAMP")
    
    if coordinated_timestamp:
        # Use coordinated timestamp from shell script (format: YYYY-MM-DD/HH-MM-SS)
        timestamp_str = coordinated_timestamp
    else:
        # Generate new timestamp if running independently
        timestamp_date = datetime.now().strftime("%Y-%m-%d")
        timestamp_time = datetime.now().strftime("%H-%M-%S")
        timestamp_str = f"{timestamp_date}/{timestamp_time}"
    
    base_output_dir = Path("outputs") / (
        f"{config.model_name}_{config.task}/"
        f"{config.mode}/"
        f"{config.data.name}/{config.data.class_column}/"
        f"lora_{config.train.use_lora}/seed{config.seed}/"
        f"{timestamp_str}"
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging with file output
    global logger
    logger = setup_logging(base_output_dir)
    
    # All arguments are now expected in the config file or as Hydra CLI overrides
    host = getattr(config, "host", config.federated.server_host)
    port = getattr(config, "port", config.federated.server_port)
    n_rounds = getattr(config, "n_rounds", None) or config.federated.n_rounds
    save_dir = base_output_dir / "federated_server"

    # Get expected clients from config.federated.client_ids if provided, otherwise infer from dataset config
    if hasattr(config.federated, "client_ids") and config.federated.client_ids:
        expected_client_ids = list(config.federated.client_ids)
        n_clients = len(expected_client_ids)
        logger.info(f"Expected clients from config: {expected_client_ids}")
    else:
        # Fallback to inferring from dataset config (old behavior)
        expected_client_ids = []
        if config.data.use_acdc:
            expected_client_ids.append("acdc")
        if config.data.use_mnms_merged:
            expected_client_ids.extend(["GE", "Philips", "Siemens", "Canon"])
        if config.data.use_mnms:
            expected_client_ids.extend(["GE", "Philips", "Siemens"])
        if config.data.use_mnms2:
            expected_client_ids.extend(["GE", "Philips"])
        n_clients = len(expected_client_ids)
        logger.warning(f"client_ids not in config, inferred from dataset config: {expected_client_ids}")

    logger.info(f"Server configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Expected clients: {n_clients} - {expected_client_ids}")
    logger.info(f"  Communication rounds: {n_rounds}")
    logger.info(f"  Save directory: {save_dir}")
    logger.info(f"  Task: {getattr(config, 'task', 'regression')}")

    # Save configuration to output directory
    from omegaconf import OmegaConf
    config_file = base_output_dir / "config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(config, f)
    logger.info(f"Saved configuration to {config_file}")

    # Initialize global model
    logger.info("Initializing global model...")
    amp_dtype, device = get_amp_dtype_and_device()
    global_model = get_model(config)
    global_model.to(device)
    
    # Calculate gradient accumulation steps for step tracking
    n_accum_steps = get_n_accum_steps(
        batch_size=config.train.batch_size,
        batch_size_per_device=config.train.batch_size_per_device,
        world_size=1,
    )
    logger.info(f"Server step tracking: batch_size={config.train.batch_size}, batch_size_per_device={config.train.batch_size_per_device}, n_accum_steps={n_accum_steps}")

    # Get primary metric from config
    primary_metric_name = get_primary_metric_name(config)
    logger.info(f"Primary metric for early stopping: {primary_metric_name}")

    # Dynamically get FederatedServer class based on task
    FederatedServer = get_server_class(config)

    # Initialize server
    server = FederatedServer(
        global_model=global_model,
        host=host,
        port=port,
        n_clients=n_clients,
        n_rounds=n_rounds,
        save_dir=save_dir,
        use_lora=config.train.use_lora,
        freeze_pretrained=config.model.freeze_pretrained,
        aggregate_head=getattr(config.federated, "aggregate_head", True),
        local_epochs=config.federated.local_epochs,
        batch_size_per_device=config.train.batch_size_per_device,
        n_accum_steps=n_accum_steps,
        eval_steps=config.train.eval_steps,
        max_n_ckpts=getattr(config.train, "max_n_ckpts", 5),
        expected_client_ids=expected_client_ids,
        early_stopping_patience=config.train.early_stopping.patience,
        early_stopping_min_delta=config.train.early_stopping.min_delta,
        primary_metric_name=primary_metric_name,
        output_dir=str(base_output_dir),
    )

    # Run server
    logger.info("Starting federated server...")
    server.run()

    logger.info("Server finished successfully!")


if __name__ == "__main__":
    main()
