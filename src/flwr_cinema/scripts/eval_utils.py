"""Shared evaluation utilities for centralized and federated regression models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler

from models.cinema.cinema.classification.dataset import (
    get_image_transforms,
    EndDiastoleEndSystoleDataset as ClassificationDataset,
)
from models.cinema.cinema.classification.train import get_classification_or_regression_model
from models.cinema.cinema.device import get_amp_dtype_and_device
from models.cinema.cinema.regression.dataset import EndDiastoleEndSystoleDataset as RegressionDataset
from models.cinema.cinema.regression.train import regression_eval, regression_metrics

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_metadata(metadata_path: Path, metadata_type: str = "test") -> pd.DataFrame:
    """Load metadata CSV with fallback to cache directory.
    
    Attempts to load metadata from the specified path. If not found, falls back to
    a hardcoded cache directory at ~/.cache/cinema_datasets/acdc_mnms_combined/processed/.
    
    Args:
        metadata_path: Primary path to metadata CSV file.
        metadata_type: Type of metadata (train/val/test) for logging/error messages.
    
    Returns:
        Loaded metadata DataFrame.
    
    Raises:
        FileNotFoundError: If metadata not found in either location.
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        # Fall back to hardcoded cache directory
        cache_path = Path.home() / ".cache" / "cinema_datasets" / "acdc_mnms_combined" / "processed" / metadata_path.name
        if cache_path.exists():
            metadata_path = cache_path
            logger.info(f"{metadata_type.capitalize()} metadata not found in config directory, using cache: {metadata_path}")
        else:
            logger.error(f"{metadata_type.capitalize()} metadata not found at {metadata_path} or {cache_path}. "
                        f"Please ensure {metadata_path.name} exists in either location.")
            raise FileNotFoundError(f"{metadata_type.capitalize()} metadata not found at {metadata_path} or {cache_path}")
    
    df = pd.read_csv(metadata_path, dtype={"pid": str})
    logger.info(f"Loaded {metadata_type} metadata with {len(df)} samples from {metadata_path}")
    return df


def resolve_data_dir_with_fallback(data_dir: Path) -> Path:
    """Resolve data directory with fallback to cache.
    
    Attempts to use the specified data_dir. If it doesn't exist, falls back to
    a hardcoded cache directory at ~/.cache/cinema_datasets/acdc_mnms_combined/processed/.
    
    Args:
        data_dir: Primary data directory path.
    
    Returns:
        Resolved data directory path (either original or fallback).
    
    Raises:
        FileNotFoundError: If neither location exists.
    """
    data_dir = Path(data_dir)
    
    if data_dir.exists():
        return data_dir
    
    # Try fallback cache directory
    cache_data_dir = Path.home() / ".cache" / "cinema_datasets" / "acdc_mnms_combined" / "processed"
    if cache_data_dir.exists():
        logger.info(f"Data directory not found at {data_dir}, using cache: {cache_data_dir}")
        return cache_data_dir
    
    logger.error(f"Data directory not found at {data_dir} or {cache_data_dir}. "
                f"Please ensure data exists in either location.")
    raise FileNotFoundError(f"Data directory not found at {data_dir} or {cache_data_dir}")


def find_best_checkpoint(checkpoint_dir: Path) -> tuple[Path, int | None]:
    """Find the best checkpoint by validation metric in a centralized checkpoint directory.
    
    Searches for .pt files and looks for validation metrics in:
    1. Nearby config.yaml or hydra config
    2. Checkpoint metadata files
    3. Stored JSON logs
    
    Args:
        checkpoint_dir: Directory containing checkpoints.
    
    Returns:
        Path to best checkpoint and its epoch number (if found), or last checkpoint if no metrics.
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Search for all .pt files in checkpoint directory and subdirectories
    pt_files = sorted(checkpoint_dir.glob("**/*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint files found in {checkpoint_dir}")
    
    # Try to find validation metrics from nearby config or logs
    best_ckpt = None
    best_epoch = None
    best_metric = float('inf')
    
    # Look for config.yaml to find best checkpoint info
    config_paths = list(checkpoint_dir.parent.glob("config.yaml")) + \
                   list(checkpoint_dir.parent.parent.glob("config.yaml")) + \
                   list(checkpoint_dir.parent.parent.parent.glob("config.yaml"))
    
    # Look for validation log file
    metrics_log = checkpoint_dir.parent / "metrics_log.json"
    if not metrics_log.exists():
        metrics_log = checkpoint_dir.parent.parent / "metrics_log.json"
    if not metrics_log.exists():
        metrics_log = checkpoint_dir.parent.parent.parent / "metrics_log.json"
    
    if metrics_log.exists():
        try:
            with open(metrics_log) as f:
                metrics_data = json.load(f)
                for entry in metrics_data:
                    if "val_mae" in entry and entry["val_mae"] < best_metric:
                        best_metric = entry["val_mae"]
                        best_epoch = entry.get("epoch")
                        # Find corresponding checkpoint
                        if best_epoch is not None:
                            epoch_ckpts = [p for p in pt_files if f"epoch_{best_epoch}" in p.name or f"_{best_epoch}" in p.name]
                            if epoch_ckpts:
                                best_ckpt = epoch_ckpts[0]
        except Exception as e:
            logger.warning(f"Failed to read metrics log: {e}")
    
    # If no best checkpoint found, use model_best.pt if it exists, else latest
    if best_ckpt is None:
        best_ckpts = [p for p in pt_files if "best" in p.name.lower()]
        if best_ckpts:
            best_ckpt = best_ckpts[0]
        else:
            # Use latest checkpoint by modification time
            best_ckpt = max(pt_files, key=lambda p: p.stat().st_mtime)
            logger.warning(f"Using latest checkpoint {best_ckpt.name} (no best checkpoint info found)")
    
    # Extract epoch if possible
    if best_epoch is None and best_ckpt:
        name = best_ckpt.stem
        # Try to extract epoch number from filename
        parts = name.split("_")
        for i, part in enumerate(parts):
            if part == "epoch" and i + 1 < len(parts):
                try:
                    best_epoch = int(parts[i + 1])
                    break
                except ValueError:
                    pass
    
    logger.info(f"Selected best checkpoint: {best_ckpt.name} (epoch: {best_epoch})")
    return best_ckpt, best_epoch


def find_last_federated_round(output_dir: Path) -> tuple[Path, int]:
    """Find the last (most recent) federated global model checkpoint.
    
    Args:
        output_dir: Federated training output directory.
    
    Returns:
        Path to global model checkpoint and round number.
    """
    federated_server_dir = output_dir / "federated_server"
    if not federated_server_dir.exists():
        raise FileNotFoundError(f"No federated_server directory found in {output_dir}")
    
    # Search for global_model_round_*.pt files
    round_files = sorted(federated_server_dir.glob("global_model_round_*.pt"))
    
    if round_files:
        last_ckpt = round_files[-1]
    else:
        # Fall back to global_model_final.pt
        final_ckpt = federated_server_dir / "global_model_final.pt"
        if final_ckpt.exists():
            last_ckpt = final_ckpt
        else:
            raise FileNotFoundError(f"No global model checkpoints found in {federated_server_dir}")
    
    # Extract round number
    name = last_ckpt.stem
    round_num = None
    if "round" in name:
        parts = name.split("_")
        for i, part in enumerate(parts):
            if part == "round" and i + 1 < len(parts):
                try:
                    round_num = int(parts[i + 1])
                    break
                except ValueError:
                    pass
    
    if round_num is None:
        round_num = len(round_files)
    
    logger.info(f"Selected last federated round: {last_ckpt.name} (round: {round_num})")
    return last_ckpt, round_num


def find_best_federated_round(output_dir: Path) -> tuple[Path, int]:
    """Find the best federated global model checkpoint by validation metric.
    
    Args:
        output_dir: Federated training output directory.
    
    Returns:
        Path to global model checkpoint and round number.
    """
    federated_server_dir = output_dir / "federated_server"
    if not federated_server_dir.exists():
        raise FileNotFoundError(f"No federated_server directory found in {output_dir}")
    
    # Load aggregated metrics history
    metrics_file = federated_server_dir / "aggregated_metrics_history.json"
    if not metrics_file.exists():
        logger.warning(f"No aggregated metrics found, using last round instead")
        return find_last_federated_round(output_dir)
    
    try:
        with open(metrics_file) as f:
            metrics_history = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read metrics file: {e}, using last round instead")
        return find_last_federated_round(output_dir)
    
    # Find round with lowest fedavg_metrics mae
    best_round = None
    best_metric = float('inf')
    
    for entry in metrics_history:
        if "fedavg_metrics" in entry and "mae" in entry["fedavg_metrics"]:
            mae = entry["fedavg_metrics"]["mae"]
            if mae < best_metric:
                best_metric = mae
                best_round = entry.get("round")
    
    if best_round is None:
        logger.warning("Could not find best round from metrics, using last round instead")
        return find_last_federated_round(output_dir)
    
    # Find corresponding checkpoint file
    ckpt_path = federated_server_dir / f"global_model_round_{best_round:03d}.pt"
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint for round {best_round} not found, using last round instead")
        return find_last_federated_round(output_dir)
    
    logger.info(f"Selected best federated round: round_{best_round:03d}.pt (mae: {best_metric:.4f})")
    return ckpt_path, best_round


def get_round_participants(output_dir: Path, round_num: int) -> list[str]:
    """Get list of clients that participated in a specific federated round.
    
    Infers from:
    1. Aggregated metrics history (client_steps keys)
    2. Client checkpoint directories
    3. Config file
    
    Args:
        output_dir: Federated training output directory.
        round_num: Round number.
    
    Returns:
        List of client IDs (e.g., ["acdc", "GE"]).
    """
    federated_server_dir = output_dir / "federated_server"
    
    # Try to extract from metrics history
    metrics_file = federated_server_dir / "aggregated_metrics_history.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                metrics_history = json.load(f)
                for entry in metrics_history:
                    if entry.get("round") == round_num:
                        if "client_steps" in entry:
                            return list(entry["client_steps"].keys())
        except Exception as e:
            logger.warning(f"Failed to extract participants from metrics: {e}")
    
    # Fall back to client directories
    client_dirs = sorted(federated_server_dir.parent.glob("client_*/"))
    if client_dirs:
        participants = [d.name.replace("client_", "") for d in client_dirs]
        logger.info(f"Inferred participants from client directories: {participants}")
        return participants
    
    # Fall back to config
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        config_path = output_dir.parent / "config.yaml"
    if not config_path.exists():
        config_path = output_dir.parent.parent / "config.yaml"
    
    if config_path.exists():
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_path)
            if "federated" in cfg and "n_clients" in cfg.federated:
                # Default clients based on config
                if cfg.federated.n_clients >= 2:
                    return ["acdc", "GE", "Philips", "Siemens", "Canon"]  # Common default
        except Exception as e:
            logger.warning(f"Failed to read config: {e}")
    
    logger.warning("Could not determine round participants, assuming all clients")
    return ["acdc", "GE", "Philips", "Siemens", "Canon"]  # Default assumption


def load_checkpoint_and_config(
    checkpoint_path: Path, 
    output_dir: Path | None = None
) -> tuple[dict, dict]:
    """Load checkpoint state dict and associated config.
    
    Args:
        checkpoint_path: Path to checkpoint (.pt file).
        output_dir: Output directory containing config.yaml (optional, will search parents).
    
    Returns:
        Tuple of (checkpoint_dict, config_dict)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Find and load config
    config_paths = []
    if output_dir:
        config_paths.extend([
            output_dir / "config.yaml",
            output_dir.parent / "config.yaml",
            output_dir.parent.parent / "config.yaml",
            output_dir / ".hydra" / "config.yaml",
            output_dir.parent / ".hydra" / "config.yaml",
        ])
    
    # Also search from checkpoint directory
    config_paths.extend([
        checkpoint_path.parent / "config.yaml",
        checkpoint_path.parent.parent / "config.yaml",
        checkpoint_path.parent.parent.parent / "config.yaml",
        checkpoint_path.parent.parent / ".hydra" / "config.yaml",
        checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml",
    ])
    
    config_dict = None
    for config_path in config_paths:
        if config_path.exists():
            try:
                from omegaconf import OmegaConf
                config_dict = OmegaConf.to_container(OmegaConf.load(config_path))
                logger.info(f"Loaded config from {config_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
    
    if config_dict is None:
        # Fall back to default config from configs/ directory
        try:
            from omegaconf import OmegaConf
            repo_root = Path(__file__).parent.parent
            default_config_path = repo_root / "configs" / "cinema_fed.yaml"
            if default_config_path.exists():
                config_dict = OmegaConf.to_container(OmegaConf.load(default_config_path))
                logger.warning(f"Using default config from {default_config_path}")
            else:
                raise FileNotFoundError(f"Could not find config.yaml for checkpoint {checkpoint_path}")
        except Exception as e:
            raise FileNotFoundError(f"Could not find config.yaml for checkpoint {checkpoint_path}") from e
    
    return checkpoint, config_dict # type: ignore


def load_test_metadata(config: dict) -> pd.DataFrame:
    """Load test metadata from combined dataset.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        DataFrame with test metadata including client column.
    """
    data_dir = Path(config["data"]["combined"]["dir"]).expanduser()
    metadata_path = data_dir / "test_metadata.csv"
    return load_metadata(metadata_path, metadata_type="test")


def get_test_datasets_by_client(
    config: dict,
    test_meta_df: pd.DataFrame,
) -> dict[str, tuple[int, ClassificationDataset | RegressionDataset]]:
    """Create test datasets split by client. Supports both classification and regression.
    
    Args:
        config: Configuration dictionary.
        test_meta_df: Test metadata DataFrame.
    
    Returns:
        Dictionary mapping client_id to (n_samples, dataset).
    """
    from omegaconf import OmegaConf
    
    data_dir = Path(config["data"]["combined"]["dir"]).expanduser()
    data_dir = resolve_data_dir_with_fallback(data_dir)
    cfg = OmegaConf.create(config)
    
    # Determine task type
    task = getattr(cfg, "task", "regression")
    is_classification = task == "classification"
    
    # Get image transforms
    _, transform = get_image_transforms(cfg)
    
    # Identify clients
    if "client" in test_meta_df.columns:
        client_col = "client"
    elif "vendor" in test_meta_df.columns:
        client_col = "vendor"
    else:
        # Single client dataset
        client_col = None
    
    datasets_by_client = {}
    
    if is_classification:
        # Classification datasets
        class_col = config["data"]["class_column"]
        train_meta_df = load_metadata(data_dir / "train_metadata.csv", metadata_type="train")
        # Extract classes from config to preserve training order (CRITICAL!)
        try:
            classes = list(config["data"][class_col]) if config["data"][class_col] else None
        except (KeyError, TypeError):
            classes = None
        
        if not classes:
            logger.warning(f"No explicit class list in config.data.{class_col}, using sorted unique from training")
            classes = sorted(train_meta_df[class_col].unique().tolist())
        
        if client_col is None:
            # Single client case
            client_id = test_meta_df["dataset"].iloc[0] if "dataset" in test_meta_df.columns else "unknown"
            dataset = ClassificationDataset(
                data_dir=data_dir / "test",
                meta_df=test_meta_df,
                class_col=class_col,
                classes=classes,
                views=cfg.model.views,
                transform=transform,
            )
            datasets_by_client[client_id] = (len(test_meta_df), dataset)
        else:
            # Multi-client case
            for client_id in sorted(test_meta_df[client_col].unique()):
                client_meta_df = test_meta_df[test_meta_df[client_col] == client_id]
                dataset = ClassificationDataset(
                    data_dir=data_dir / "test",
                    meta_df=client_meta_df,
                    class_col=class_col,
                    classes=classes,
                    views=cfg.model.views,
                    transform=transform,
                )
                datasets_by_client[client_id] = (len(client_meta_df), dataset)
    else:
        # Regression datasets
        reg_col = config["data"]["regression_column"]
        train_meta_df = load_metadata(data_dir / "train_metadata.csv", metadata_type="train")
        reg_mean = train_meta_df[reg_col].mean()
        reg_std = train_meta_df[reg_col].std()
        
        if client_col is None:
            # Single client case
            client_id = test_meta_df["dataset"].iloc[0] if "dataset" in test_meta_df.columns else "unknown"
            dataset = RegressionDataset(
                data_dir=data_dir / "test",
                meta_df=test_meta_df,
                reg_col=reg_col,
                reg_mean=reg_mean,
                reg_std=reg_std,
                views=cfg.model.views,
                transform=transform,
            )
            datasets_by_client[client_id] = (len(test_meta_df), dataset)
        else:
            # Multi-client case
            for client_id in sorted(test_meta_df[client_col].unique()):
                client_meta_df = test_meta_df[test_meta_df[client_col] == client_id]
                dataset = RegressionDataset(
                    data_dir=data_dir / "test",
                    meta_df=client_meta_df,
                    reg_col=reg_col,
                    reg_mean=reg_mean,
                    reg_std=reg_std,
                    views=cfg.model.views,
                    transform=transform,
                )
                datasets_by_client[client_id] = (len(client_meta_df), dataset)
    
    logger.info(f"Created test datasets ({task}) for {len(datasets_by_client)} clients: {list(datasets_by_client.keys())}")
    return datasets_by_client


def compute_metrics_on_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    reg_mean: float,
    reg_std: float,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    """Compute regression metrics on a dataloader.
    
    Args:
        model: Regression model.
        dataloader: DataLoader to evaluate.
        reg_mean: Normalization mean for denormalization.
        reg_std: Normalization std for denormalization.
        config: Configuration dictionary.
        device: Device to use.
    
    Returns:
        Dictionary with normalized and restored metrics.
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(config)
    amp_dtype, _ = get_amp_dtype_and_device()
    
    # Build patch size dict - handle missing views gracefully
    views = cfg.model.views if isinstance(cfg.model.views, list) else [cfg.model.views]
    patch_size_dict = {}
    spacing_dict = {}
    
    for v in views:
        if v == "sax" and "sax" in cfg.data:
            patch_size_dict[v] = cfg.data.sax.patch_size
            spacing_dict[v] = cfg.data.sax.spacing
        elif v == "lax" and "lax" in cfg.data:
            patch_size_dict[v] = cfg.data.lax.patch_size
            spacing_dict[v] = cfg.data.lax.spacing
        else:
            # Default to sax if view config missing
            logger.warning(f"View {v} not found in config, using default sax patch size")
            patch_size_dict[v] = cfg.data.sax.patch_size if "sax" in cfg.data else [192, 192, 16]
            spacing_dict[v] = cfg.data.sax.spacing if "sax" in cfg.data else [1.0, 1.0, 10.0]
    
    pred_labels = []
    true_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            preds, _ = regression_eval(
                model=model,
                batch=batch,
                patch_size_dict=patch_size_dict,
                amp_dtype=amp_dtype,
                device=device,
            )
            pred_labels.append(preds)
            true_labels.append(batch["label"])
    
    pred_labels = torch.cat(pred_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels = torch.cat(true_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    
    # Compute metrics on normalized scale
    metrics = regression_metrics(true_labels=true_labels, pred_labels=pred_labels)
    
    # Denormalize and compute metrics on original scale
    restored_pred_labels = (pred_labels * reg_std) + reg_mean
    restored_true_labels = (true_labels * reg_std) + reg_mean
    restored_metrics = regression_metrics(
        true_labels=restored_true_labels,
        pred_labels=restored_pred_labels,
    )
    restored_metrics = {f"restored_{k}": v for k, v in restored_metrics.items()}
    metrics.update(restored_metrics)
    
    return metrics


def compute_weighted_metrics(
    metrics_by_client: dict[str, dict],
    test_meta_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute weighted average metrics across clients.
    
    Args:
        metrics_by_client: Dictionary mapping client_id to metrics dict.
        test_meta_df: Test metadata to get client sample counts.
    
    Returns:
        Weighted average metrics.
    """
    if not metrics_by_client:
        return {}
    
    # Identify client column
    if "client" in test_meta_df.columns:
        client_col = "client"
    elif "vendor" in test_meta_df.columns:
        client_col = "vendor"
    else:
        # Single client, no weighting needed
        return list(metrics_by_client.values())[0]
    
    # Get sample counts per client
    client_counts = test_meta_df[client_col].value_counts().to_dict()
    total_samples = len(test_meta_df)
    
    weighted_metrics = {}
    for metric_name in metrics_by_client[list(metrics_by_client.keys())[0]].keys():
        weighted_value = 0.0
        for client_id, metrics in metrics_by_client.items():
            count = client_counts.get(client_id, 1)
            weight = count / total_samples
            weighted_value += metrics[metric_name] * weight
        weighted_metrics[metric_name] = weighted_value
    
    return weighted_metrics


def compute_cross_center_metrics(
    model: torch.nn.Module,
    datasets_by_client: dict[str, tuple[int, ClassificationDataset | RegressionDataset]],
    test_meta_df: pd.DataFrame,
    reg_mean: float,
    reg_std: float,
    config: dict,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Compute cross-center validation metrics.
    
    For each client's model, evaluate on all other clients' test data.
    Automatically detects task type (classification vs regression) and uses appropriate metrics.
    
    Args:
        model: Global model.
        datasets_by_client: Datasets split by client.
        test_meta_df: Test metadata.
        reg_mean: Normalization mean (for regression only).
        reg_std: Normalization std (for regression only).
        config: Configuration dictionary.
        device: Device to use.
    
    Returns:
        Dictionary mapping "client_X_to_client_Y" to metrics dict.
    """
    from omegaconf import OmegaConf
    
    cross_center_metrics = {}
    clients = sorted(datasets_by_client.keys())
    
    # Determine task type
    cfg = OmegaConf.create(config)
    task = getattr(cfg, "task", "regression")
    is_classification = task == "classification"
    
    # For each source client, evaluate on all target clients
    for source_client in clients:
        for target_client in clients:
            if source_client == target_client:
                continue
            
            # Create dataloader for target client data
            n_samples, dataset = datasets_by_client[target_client]
            sampler = SequentialSampler(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=1,
                drop_last=False,
                pin_memory=True,
                num_workers=0,
            )
            
            # Compute metrics (task-specific)
            try:
                if is_classification:
                    # For classification, try to use the classification metrics function if available
                    # Otherwise fall back to regression metrics (which may fail gracefully and be caught)
                    try:
                        from cinema_classification.federated.evaluate import compute_metrics_on_dataloader_with_balanced_accuracy
                        metrics, _, _ = compute_metrics_on_dataloader_with_balanced_accuracy(
                            model=model,
                            dataloader=dataloader,
                            config=config,
                            device=device,
                            task="classification",
                            reg_mean=None,
                            reg_std=None,
                        )
                    except (ImportError, TypeError):
                        # Fall back to generic metrics computation
                        metrics = compute_metrics_on_dataloader(
                            model=model,
                            dataloader=dataloader,
                            reg_mean=0,
                            reg_std=1,
                            config=config,
                            device=device,
                        )
                else:
                    # Use regression metrics
                    metrics = compute_metrics_on_dataloader(
                        model=model,
                        dataloader=dataloader,
                        reg_mean=reg_mean,
                        reg_std=reg_std,
                        config=config,
                        device=device,
                    )
                
                cross_center_key = f"{source_client}_to_{target_client}"
                cross_center_metrics[cross_center_key] = metrics
                logger.debug(f"Cross-center evaluation {cross_center_key}: success")
            except Exception as e:
                cross_center_key = f"{source_client}_to_{target_client}"
                logger.debug(f"Failed to evaluate {cross_center_key}: {type(e).__name__}: {e}")
                # Don't fail entire cross-center evaluation for a single pair - just skip it
                continue
    
    return cross_center_metrics


def format_results_json_csv(
    overall_metrics: dict[str, float],
    per_client_metrics: dict[str, dict[str, float]],
    weighted_metrics: dict[str, float],
    cross_center_metrics: dict[str, dict[str, float]] | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, dict[str, float] | dict[str, dict[str, float]]], pd.DataFrame, pd.DataFrame | None]:
    """Format evaluation results into JSON and CSV format.
    
    Supports both classification and regression metrics automatically.
    
    Args:
        overall_metrics: Overall metrics dictionary.
        per_client_metrics: Per-client metrics dictionary.
        weighted_metrics: Weighted average metrics.
        cross_center_metrics: Cross-center evaluation results (optional).
        output_dir: Directory to save results (optional).
    
    Returns:
        Tuple of (results_dict, metrics_dataframe, cross_center_dataframe)
    """
    # Classification metrics
    classification_metrics = [
        "accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc", 
        "sensitivity", "specificity"
    ]
    
    # Regression metrics
    regression_metrics_list = [
        "rmse", "mae", "max_error", "min_error",
        "restored_rmse", "restored_mae", "restored_max_error", "restored_min_error"
    ]
    
    # Detect which metrics are present (classification vs regression)
    all_metrics_keys = set(overall_metrics.keys()) | set(weighted_metrics.keys())
    for metrics in per_client_metrics.values():
        all_metrics_keys.update(metrics.keys())
    
    is_classification = any(m in all_metrics_keys for m in classification_metrics)
    is_regression = any(m in all_metrics_keys for m in regression_metrics_list)
    
    # Choose which metrics to keep based on what's present
    if is_classification:
        metrics_to_keep = classification_metrics
    elif is_regression:
        metrics_to_keep = regression_metrics_list
    else:
        # Keep all metrics if we can't determine type
        metrics_to_keep = list(all_metrics_keys)
    
    # Helper to convert numpy types to Python floats for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (float, int)):
            return obj
        else:
            return obj
    
    # Filter to only selected metrics
    overall_filtered = {k: v for k, v in overall_metrics.items() if k in metrics_to_keep}
    weighted_filtered = {k: v for k, v in weighted_metrics.items() if k in metrics_to_keep}
    per_client_filtered = {
        client: {k: v for k, v in metrics.items() if k in metrics_to_keep}
        for client, metrics in per_client_metrics.items()
    }
    
    # Prepare JSON results
    results_dict = {
        "overall": overall_filtered,
        "weighted": weighted_filtered,
        "per_client": per_client_filtered,
    }
    
    if cross_center_metrics:
        cross_center_filtered = {
            k: {m: v for m, v in metrics.items() if m in metrics_to_keep}
            for k, metrics in cross_center_metrics.items()
        }
        results_dict["cross_center"] = cross_center_filtered
    
    # Convert to serializable format with explicit type casting
    results_dict_serializable: dict[str, dict[str, float] | dict[str, dict[str, float]]] = convert_to_serializable(results_dict)  # type: ignore
    
    # Prepare DataFrames for CSV
    metrics_data: dict[str, list] = {"metric": []}
    metrics_data["overall"] = []
    metrics_data["weighted"] = []
    for client in sorted(per_client_filtered.keys()):
        metrics_data[client] = []
    
    for metric in metrics_to_keep:
        metrics_data["metric"].append(metric)
        metrics_data["overall"].append(float(overall_filtered.get(metric, np.nan)))
        metrics_data["weighted"].append(float(weighted_filtered.get(metric, np.nan)))
        for client in sorted(per_client_filtered.keys()):
            metrics_data[client].append(float(per_client_filtered[client].get(metric, np.nan)))
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Prepare cross-center DataFrame
    cross_center_df = None
    if cross_center_metrics:
        cross_center_data: dict[str, list] = {"metric": []}
        for key in sorted(cross_center_metrics.keys()):
            cross_center_data[key] = []
        
        for metric in metrics_to_keep:
            cross_center_data["metric"].append(metric)
            for key in sorted(cross_center_metrics.keys()):
                metrics = cross_center_metrics[key]
                cross_center_data[key].append(float(metrics.get(metric, np.nan)))
        
        cross_center_df = pd.DataFrame(cross_center_data)
    
    # Save to files if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / "test_metrics.json"
        with open(json_path, "w") as f:
            json.dump(results_dict_serializable, f, indent=2)
        logger.info(f"Saved metrics to {json_path}")
        
        # Save CSV
        csv_path = output_dir / "test_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics to {csv_path}")
        
        # Save cross-center CSV if available
        if cross_center_df is not None:
            cross_center_csv_path = output_dir / "test_cross_center_metrics.csv"
            cross_center_df.to_csv(cross_center_csv_path, index=False)
            logger.info(f"Saved cross-center metrics to {cross_center_csv_path}")
    
    return results_dict_serializable, metrics_df, cross_center_df
