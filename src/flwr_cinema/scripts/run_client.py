"""Script to run federated learning client."""

from __future__ import annotations

import json
import logging
import os
import sys
print("PYTHONPATH:", sys.path)
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from models.cinema.cinema.classification.dataset import get_image_transforms  # noqa: E402
from models.cinema.cinema.classification.train import get_classification_or_regression_model  # noqa: E402
from models.cinema.cinema.convvit import ConvViT
from models.cinema.cinema.train import maybe_reduce_batch_size  # noqa: E402

from torch.utils.data import DataLoader  # noqa: E402

from flwr_cinema.scripts.lora import apply_lora_to_convvit, apply_dual_lora_to_convvit  # noqa: E402
from models.cinema.cinema.optim import get_n_accum_steps  # noqa: E402
from flwr_cinema.scripts.federated_dataset_loader import load_client_dataloaders  # noqa: E402
import hydra  # noqa: E402


def setup_logging(log_dir: Path, client_id: str) -> logging.Logger:
    """Setup logging to both console and file for a client."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"client_{client_id}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    log_file = log_dir / f"client_{client_id}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger



def get_client_class(config):
    """Dynamically import FederatedClient based on task type.
    
    Args:
        config: Configuration with config.task field.
        
    Returns:
        FederatedClient class from appropriate module (classification or regression).
    """
    task = getattr(config, "task", "regression")
    
    if task == "classification":
        from cinema_classification.federated.client import FederatedClient
        logger.info("Using classification federated client")
    else:
        from cinema_regression.federated.client import FederatedClient
        logger.info("Using regression federated client")
    
    return FederatedClient


def get_task_specific_functions(config):
    """Load task-specific loss and eval functions based on config.task.
    
    Args:
        config: Configuration dictionary with config.task (classification or regression).
        
    Returns:
        Tuple of (loss_fn, eval_dataloader_fn) appropriate for the task.
    """
    task = getattr(config, "task", "regression")  # Default to regression for backward compatibility
    
    if task == "classification":
        from models.cinema.cinema.classification.train import (
            classification_loss,
            classification_eval_dataloader,
        )
        return classification_loss, classification_eval_dataloader
    else:
        from models.cinema.cinema.regression.train import (
            regression_loss,
            regression_eval_dataloader,
        )
        return regression_loss, regression_eval_dataloader
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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



def load_client_data(
    config,
    client_id: str,
    vendor: str | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    """Load training data for a specific federated learning client.
    
    This is a thin wrapper around the common load_client_dataloaders() from
    data/federated_dataset_loader.py that handles task-agnostic data loading.
    
    Terminology:
        - client_id: Federated learning participant (acdc, GE, Canon, Siemens, Philips)
        - vendor: Scanner manufacturer in metadata (only for filtered MnMs data)
    
    Args:
        config: Configuration dictionary.
        client_id: Federated client identifier (e.g., "acdc", "GE", "Canon").
                  Matches the 'client' column in combined dataset metadata.
        vendor: [DEPRECATED] Vendor to filter by. Now uses client_id directly.
        
    Returns:
        Tuple of (train_loader, val_loader, stats_dict) for the client.
    """
    logger.info(f"Loading data for client {client_id} using common task-agnostic loader...")
    
    # Use the common loader from data module
    train_loader, val_loader, stats = load_client_dataloaders(
        config=config,
        client_id=client_id,
        vendor=vendor,
        batch_size=config.train.batch_size_per_device,
        num_workers=config.train.n_workers,
    )
    
    # Log statistics for IID/non-IID check
    logger.info(f"Client statistics: {stats}")
    
    # Save statistics for analysis
    output_dir = Path("outputs") / client_id
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_save_path = output_dir / f"client_stats_{client_id}_{timestamp}.json"
    with open(stats_save_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    return train_loader, val_loader, stats


import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="cinema_fed.yaml")
def main(config: DictConfig) -> None:
    """Main function to run federated client with Hydra config only."""
    # Construct output directory using same config template structure as server
    # Use coordinated timestamp from environment if available (set by run_federated.sh)
    # Otherwise generate new timestamp
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
    
    # Initialize logger with default setup (will be updated with file logging after receiving output_dir from server)
    global logger
    
    # Get client_id from config.client_id (override), or from config.federated.client_ids by index (environment variable),
    # or raise error if not provided
    
    # Priority 1: Direct config.client_id override
    if hasattr(config, "client_id"):
        client_id = config.client_id
    # Priority 2: Environment variable CLIENT_INDEX pointing to federated.client_ids
    elif "CLIENT_INDEX" in os.environ:
        client_index = int(os.environ["CLIENT_INDEX"])
        if hasattr(config, "federated") and hasattr(config.federated, "client_ids"):
            client_ids = config.federated.client_ids
            if 0 <= client_index < len(client_ids):
                client_id = client_ids[client_index]
            else:
                raise ValueError(f"CLIENT_INDEX={client_index} out of range for client_ids={client_ids}")
        else:
            raise ValueError("CLIENT_INDEX set but federated.client_ids not found in config")
    # Priority 3: federated.client_ids exists, use first one (for single-client test)
    elif hasattr(config, "federated") and hasattr(config.federated, "client_ids") and len(config.federated.client_ids) > 0:
        logger.warning("No client_id specified, using first client_id from federated.client_ids")
        client_id = config.federated.client_ids[0]
    else:
        raise ValueError("client_id must be provided as config.client_id override, CLIENT_INDEX environment variable, or federated.client_ids in config")
    
    vendor = getattr(config, "vendor", None)
    server_host = getattr(config, "server_host", config.federated.server_host)
    server_port = getattr(config, "server_port", config.federated.server_port)
    n_rounds = getattr(config, "n_rounds", None) or config.federated.n_rounds

    logger.info(f"Client configuration:")
    logger.info(f"  Client ID: {client_id}")
    logger.info(f"  Vendor filter: {vendor}")
    logger.info(f"  Server: {server_host}:{server_port}")
    logger.info(f"  Communication rounds: {n_rounds}")

    # Initialize model
    logger.info("Initializing local model...")
    model = get_model(config)

    # Load client data
    logger.info("Loading client data...")
    train_loader, val_loader, client_stats = load_client_data(config, client_id, vendor)

    # Print and save client statistics for IID/non-IID check (overall, train, val, test)
    logger.info(f"Client statistics for IID/non-IID check: {client_stats}")
    
    # Initialize client (temporary run_dir - will be updated by server)
    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    
    # Calculate gradient accumulation steps for federated client
    n_accum_steps = get_n_accum_steps(
        batch_size=config.train.batch_size,
        batch_size_per_device=config.train.batch_size_per_device,
        world_size=1,  # Single GPU per client
    )
    logger.info(f"Client gradient accumulation: batch_size={config.train.batch_size}, batch_size_per_device={config.train.batch_size_per_device}, n_accum_steps={n_accum_steps}")

    # Get task-specific functions
    loss_fn, eval_dataloader_fn = get_task_specific_functions(config)
    logger.info(f"Loaded task-specific functions for task: {getattr(config, 'task', 'regression')}")
    
    # Wrap loss function with class weighting if available and enabled (for classification)
    use_class_weighted_loss = getattr(config.train, "use_class_weighted_loss", False)
    if getattr(config, "task", "regression") == "classification" and use_class_weighted_loss:
        from scripts.weighted_losses import wrap_loss_with_class_weights
        class_distribution = client_stats.get("train", {}).get("class_distribution", None)
        if class_distribution:
            loss_fn = wrap_loss_with_class_weights(loss_fn, class_distribution)
            logger.info(f"Applied class-weighted BCE with distribution: {class_distribution}")
        else:
            logger.info("Classification task: class_weighted_loss enabled but no class_distribution available")

    # Dynamically get FederatedClient class based on task
    FederatedClient = get_client_class(config)

    # Create client-specific directories at same level as federated_server
    client_run_dir = base_output_dir / f"client_{client_id}"
    client_run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_save_path = str(client_run_dir / f"client_stats_{client_id}_{timestamp}.json")
    metrics_save_path = str(client_run_dir / f"metrics_{client_id}_{timestamp}.json")
    
    with open(stats_save_path, "w") as f:
        json.dump(client_stats, f, indent=2)

    client = FederatedClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        server_host=server_host,
        server_port=server_port,
        loss_fn=loss_fn,
        views=views,
        local_epochs=config.federated.local_epochs,
        lr=float(config.train.lr),
        weight_decay=float(config.train.weight_decay),
        clip_grad=float(config.train.clip_grad) if float(config.train.clip_grad) > 0 else None,
        use_lora=config.train.use_lora,
        freeze_pretrained=config.model.freeze_pretrained,
        metrics_save_path=metrics_save_path,
        stats_save_path=stats_save_path,
        eval_dataloader_fn=eval_dataloader_fn,
        config=config,
        run_dir=str(client_run_dir),
        n_accum_steps=n_accum_steps,
    )

    # Run client (will connect to server and align paths via output_dir communication)
    logger.info("Starting federated client...")
    client.run(n_rounds)

    logger.info("Client finished successfully!")


if __name__ == "__main__":
    main()
