"""Task-agnostic federated dataset loader for both classification and regression.

This module provides dataset loading utilities for federated learning that work
for both classification and regression tasks:

1. **Client Partitioning**: Partitions data by vendor (client) from metadata CSV files
2. **Per-Client Normalization**: Computes normalization statistics per client
   - For regression: target variable (e.g., LV EF) mean/std
   - For classification: (no per-sample normalization needed)
3. **Split Management**: Maintains original train/val/test splits per dataset
4. **Task Flexibility**: Works with both centralized and federated training modes

Usage:
    # For regression
    loaders = load_federated_dataloaders(
        dataset_configs=configs,
        task="regression",
        target_col="lv_ef",
    )
    
    # For classification
    loaders = load_federated_dataloaders(
        dataset_configs=configs,
        task="classification",
        target_col="pathology",
    )
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from flwr_cinema.scripts.eval_utils import load_metadata

if TYPE_CHECKING:
    from monai.transforms import Transform

logger = logging.getLogger(__name__)


# Vendor normalization mapping between datasets
VENDOR_NORMALIZATION = {
    # MnMs vendor codes (from Vendor column in CSV)
    "A": "Siemens",
    "B": "Philips",
    "C": "GE",
    "D": "Canon",
    # MnMs2 full vendor names (from VENDOR column in CSV)
    "SIEMENS": "Siemens",
    "Philips Medical Systems": "Philips",
    "GE MEDICAL SYSTEMS": "GE",
    # Normalized names
    "Siemens": "Siemens",
    "Philips": "Philips",
    "GE": "GE",
    "Canon": "Canon",
}


def normalize_vendor(vendor: str) -> str:
    """Normalize vendor names across datasets.
    
    Args:
        vendor: Vendor name to normalize.
        
    Returns:
        Normalized vendor name.
    """
    return VENDOR_NORMALIZATION.get(vendor, vendor)


class ClientDataStats:
    """Statistics for a single client's data."""
    
    def __init__(
        self,
        client_id: str,
        dataset_name: str,
        vendor: str | None,
        train_size: int,
        val_size: int,
        test_size: int,
        target_mean: float | None = None,
        target_std: float | None = None,
        class_distribution: dict | None = None,
        class_imbalance_ratio: float | None = None,
    ) -> None:
        """Initialize client data statistics.
        
        Args:
            client_id: Unique client identifier.
            dataset_name: Source dataset (ACDC, MnMs, MnMs2).
            vendor: Vendor/scanner type (None for ACDC).
            train_size: Number of training samples.
            val_size: Number of validation samples.
            test_size: Number of test samples.
            target_mean: Mean of target variable (for regression normalization).
            target_std: Std of target variable (for regression normalization).
            class_distribution: Dict mapping class labels to their counts (for classification).
            class_imbalance_ratio: Ratio of max to min class counts (for classification).
        """
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.vendor = vendor
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.target_mean = target_mean
        self.target_std = target_std
        self.class_distribution = class_distribution
        self.class_imbalance_ratio = class_imbalance_ratio
    
    def __repr__(self) -> str:
        """String representation."""
        stats_str = f"train={self.train_size}, val={self.val_size}, test={self.test_size}"
        if self.target_mean is not None and self.target_std is not None:
            stats_str += f", target={self.target_mean:.2f}±{self.target_std:.2f}"
        return (
            f"ClientDataStats(id={self.client_id}, dataset={self.dataset_name}, "
            f"vendor={self.vendor}, {stats_str})"
        )


def compute_client_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    client_id: str,
    target_col: str | None = None,
    task: str = "regression",
) -> dict[str, ClientDataStats]:
    """Compute per-client statistics from metadata DataFrames.
    
    Args:
        train_df: Training metadata DataFrame (already filtered by client_id).
        val_df: Validation metadata DataFrame (already filtered by client_id, optional).
        test_df: Test metadata DataFrame (already filtered by client_id, optional).
        client_id: Client identifier (e.g., 'acdc', 'Siemens').
        target_col: Column name for target variable (e.g., 'lv_ef' for regression).
                   If None, normalization stats won't be computed (classification).
        task: Task type ("regression" or "classification"). Only compute numeric stats for regression.
        
    Returns:
        Dictionary mapping client_id to ClientDataStats.
    """
    client_stats = {}
    
    train_size = len(train_df)
    val_size = len(val_df) if val_df is not None else 0
    test_size = len(test_df) if test_df is not None else 0
    
    # Compute target statistics if provided (for regression)
    target_mean = None
    target_std = None
    if target_col is not None and target_col in train_df.columns and task == "regression":
        target_mean = float(train_df[target_col].mean())
        target_std = float(train_df[target_col].std())
        if target_std < 1e-6:
            logger.warning(f"Client {client_id} has very small std ({target_std:.6f}), using 1.0")
            target_std = 1.0
    
    # Compute class distribution for classification
    class_distribution = None
    class_imbalance_ratio = None
    if target_col is not None and target_col in train_df.columns and task == "classification":
        value_counts = train_df[target_col].value_counts().to_dict()
        class_distribution = {str(k): int(v) for k, v in value_counts.items()}
        
        # Compute imbalance ratio (max_count / min_count)
        if len(value_counts) > 1:
            max_count = max(value_counts.values())
            min_count = min(value_counts.values())
            class_imbalance_ratio = float(max_count / min_count) if min_count > 0 else None
    
    client_stats[client_id] = ClientDataStats(
        client_id=client_id,
        dataset_name=client_id,
        vendor=None,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        target_mean=target_mean,
        target_std=target_std,
        class_distribution=class_distribution,
        class_imbalance_ratio=class_imbalance_ratio,
    )
    
    return client_stats


def load_client_datasets(
    data_dir_train: Path,
    data_dir_val: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    client_stats: ClientDataStats,
    views: str | list[str],
    train_transform: Transform | None,
    val_transform: Transform | None,
    target_col: str | None = None,
    task: str = "regression",
    classes: list | None = None,
) -> tuple:
    """Load datasets for a single client with per-client normalization.
    
    Args:
        data_dir_train: Path to training data directory.
        data_dir_val: Path to validation data directory.
        train_df: Training metadata DataFrame (already filtered by client).
        val_df: Validation metadata DataFrame (already filtered by client).
        test_df: Test metadata DataFrame (already filtered by client).
        client_stats: Client statistics with normalization params.
        views: Views to load (e.g., "sax" or ["sax", "lax_4c"]).
        train_transform: Training data augmentation.
        val_transform: Validation data augmentation.
        target_col: Target column name (e.g., 'lv_ef' for regression, 'pathology' for classification).
        task: Task type ("classification" or "regression").
        classes: List of class names for classification task.
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    # Dynamically import dataset class based on task
    if task == "classification":
        from models.cinema.cinema.classification.dataset import EndDiastoleEndSystoleDataset
    elif task == "segmentation":
        from models.cinema.cinema.segmentation.dataset import EndDiastoleEndSystoleDataset
    else:
        from models.cinema.cinema.regression.dataset import EndDiastoleEndSystoleDataset
    
    # Use dataframes directly (no vendor-based filtering needed)
    train_client_df = train_df.reset_index(drop=True)
    val_client_df = val_df.reset_index(drop=True) if val_df is not None else None
    test_client_df = test_df.reset_index(drop=True) if test_df is not None else None
    
    
    # Create training dataset
    train_kwargs = {
        "data_dir": data_dir_train,
        "meta_df": train_client_df,
        "views": views,
        "transform": train_transform,
    }
    
    # Add normalization params for regression
    if task == "regression" and target_col is not None:
        train_kwargs.update({
            "reg_col": target_col,
            "reg_mean": client_stats.target_mean,
            "reg_std": client_stats.target_std,
        })
    # Add class column for classification
    elif task == "classification" and target_col is not None:
        train_kwargs["class_col"] = target_col
        if classes is not None:
            train_kwargs["classes"] = classes
    
    train_dataset = EndDiastoleEndSystoleDataset(**train_kwargs)
    
    # Create validation dataset
    val_dataset = None
    if val_client_df is not None and len(val_client_df) > 0:
        val_kwargs = {
            "data_dir": data_dir_val,
            "meta_df": val_client_df,
            "views": views,
            "transform": val_transform,
        }
        if task == "regression" and target_col is not None:
            val_kwargs.update({
                "reg_col": target_col,
                "reg_mean": client_stats.target_mean,
                "reg_std": client_stats.target_std,
            })
        elif task == "classification" and target_col is not None:
            val_kwargs["class_col"] = target_col
            if classes is not None:
                val_kwargs["classes"] = classes
        val_dataset = EndDiastoleEndSystoleDataset(**val_kwargs)
    
    # Create test dataset (uses val directory)
    test_dataset = None
    if test_client_df is not None and len(test_client_df) > 0:
        test_kwargs = {
            "data_dir": data_dir_val,
            "meta_df": test_client_df,
            "views": views,
            "transform": val_transform,
        }
        if task == "regression" and target_col is not None:
            test_kwargs.update({
                "reg_col": target_col,
                "reg_mean": client_stats.target_mean,
                "reg_std": client_stats.target_std,
            })
        elif task == "classification" and target_col is not None:
            test_kwargs["class_col"] = target_col
            if classes is not None:
                test_kwargs["classes"] = classes
        test_dataset = EndDiastoleEndSystoleDataset(**test_kwargs)
    
    logger.info(f"Loaded client {client_stats.client_id}: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}, test={len(test_dataset) if test_dataset else 0}")
    
    return train_dataset, val_dataset, test_dataset


def load_client_dataloaders(
    config: Any,
    client_id: str,
    vendor: str | None = None,
    batch_size: int | None = None,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader | None, dict]:
    """Load training data for a specific federated learning client (single-client wrapper).
    
    This is a convenience wrapper around load_federated_dataloaders that loads a single
    client's data and matches the interface of the old run_client.py load_client_data() function.
    
    Args:
        config: Hydra configuration object with:
            - config.task: "classification" or "regression"
            - config.data.regression_column: Target column for regression
            - config.model.views: List of views to load
            - config.train.batch_size_per_device: Batch size (if batch_size not provided)
            - config.train.n_workers: Number of workers (overridden by num_workers param)
            - config.data.combined.dir: Path to combined dataset (optional)
            - config.data.acdc.dir, config.data.mnms_merged.dir: Dataset paths
        client_id: Client identifier (matches 'client' column in metadata).
        vendor: [DEPRECATED] Vendor to filter by. Now uses client_id directly.
        batch_size: Batch size for dataloaders. If None, uses config.train.batch_size_per_device.
        num_workers: Number of data loading workers.
        
    Returns:
        Tuple of (train_loader, val_loader, stats_dict) where:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data (or None)
            - stats_dict: Dictionary with statistics for logging/auditing
    """
    import json
    from datetime import datetime
    
    # Get task and target column from config
    task = getattr(config, "task", "regression")
    if task == "regression":
        target_col = config.data.regression_column
    elif task == "classification":
        target_col = config.data.class_column
    else:
        target_col = None  # segmentation has no target column
    
    if batch_size is None:
        batch_size = config.train.batch_size_per_device
    
    # Determine data paths based on config
    if hasattr(config.data, "combined") and getattr(config.data.combined, "dir", None):
        combined_dir = Path(config.data.combined.dir).expanduser()
        data_dir = combined_dir
        train_meta_path = combined_dir / "train_metadata.csv"
        val_meta_path = combined_dir / "val_metadata.csv"
        test_meta_path = combined_dir / "test_metadata.csv"
    else:
        # Fallback to legacy dataset paths
        if "ACDC" in client_id:
            base_dir = Path(config.data.acdc.dir).expanduser()
        elif "MnMs" in client_id:
            base_dir = Path(config.data.mnms_merged.dir).expanduser()
        else:
            raise ValueError(f"Unknown client dataset: {client_id}")
        
        data_dir = base_dir
        train_meta_path = base_dir / "train_metadata.csv"
        val_meta_path = base_dir / "val_metadata.csv"
        test_meta_path = base_dir / "test_metadata.csv"
    
    # Load metadata with fallback to cache directory
    try:
        train_df = load_metadata(train_meta_path, metadata_type="train")
    except FileNotFoundError:
        train_df = None
    
    try:
        val_df = load_metadata(val_meta_path, metadata_type="val")
    except FileNotFoundError:
        val_df = None
    
    try:
        test_df = load_metadata(test_meta_path, metadata_type="test")
    except FileNotFoundError:
        test_df = None
    
    if train_df is None:
        raise FileNotFoundError(f"Train metadata not found at {train_meta_path}")
    
    # Determine vendor column
    vendor_col = None
    for col in ["Vendor", "vendor", "VENDOR"]:
        if col in train_df.columns:
            vendor_col = col
            break
    
    # Filter by client if specified
    if client_id and "client" in train_df.columns:
        train_df = train_df[train_df["client"] == client_id].reset_index(drop=True)
        if val_df is not None:
            val_df = val_df[val_df["client"] == client_id].reset_index(drop=True)
        if test_df is not None:
            test_df = test_df[test_df["client"] == client_id].reset_index(drop=True)
    elif vendor is not None and vendor_col:
        # Legacy vendor filtering
        train_df = train_df[train_df[vendor_col] == vendor].reset_index(drop=True)
        if val_df is not None:
            val_df = val_df[val_df[vendor_col] == vendor].reset_index(drop=True)
        if test_df is not None:
            test_df = test_df[test_df[vendor_col] == vendor].reset_index(drop=True)
    
    # Compute per-client statistics (using client_id directly, no vendor grouping)
    client_stats = compute_client_statistics(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        client_id=client_id,
        target_col=target_col,
        task=task,
    )
    
    # Get the stats for this client
    stats_obj = client_stats[client_id]
    
    # Get transforms from config
    if task == "segmentation":
        from models.cinema.cinema.segmentation.dataset import get_segmentation_transforms
        train_transform, val_transform = get_segmentation_transforms(config)
    else:
        from models.cinema.cinema.classification.dataset import get_image_transforms
        train_transform, val_transform = get_image_transforms(config)
    
    # Get classes for classification task (same as centralized: config.data[class_col])
    classes = None
    if task == "classification":
        classes = config.data[target_col]
    
    # Load datasets with appropriate split directories (like centralized training)
    train_dataset, val_dataset, test_dataset = load_client_datasets(
        data_dir_train=data_dir / "train",
        data_dir_val=data_dir / "val",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        client_stats=stats_obj,
        views=config.model.views,
        train_transform=train_transform,
        val_transform=val_transform,
        target_col=target_col,
        task=task,
        classes=classes,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Don't batch validation as each sample has different number of slices
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    # Build stats dictionary matching old format
    stats_dict = {
        "overall": {
            "n_samples": stats_obj.train_size + (stats_obj.val_size or 0) + (stats_obj.test_size or 0),
        },
        "train": {
            "n_samples": stats_obj.train_size,
        },
        "val": {
            "n_samples": stats_obj.val_size or 0,
        },
        "test": {
            "n_samples": stats_obj.test_size or 0,
        },
    }
    
    # Add target statistics if available (regression)
    if stats_obj.target_mean is not None and stats_obj.target_std is not None:
        stats_dict["train"].update({
            f"{target_col}_mean": stats_obj.target_mean,
            f"{target_col}_std": stats_obj.target_std,
        })
    
    # Add class distribution and imbalance ratio if available (classification)
    if stats_obj.class_distribution is not None:
        stats_dict["train"]["class_distribution"] = stats_obj.class_distribution
    if stats_obj.class_imbalance_ratio is not None:
        stats_dict["train"]["class_imbalance_ratio"] = stats_obj.class_imbalance_ratio
    
    return train_loader, val_loader, stats_dict


def load_federated_dataloaders(
    dataset_configs: list[dict],
    views: str | list[str],
    train_transform: Transform | None,
    val_transform: Transform | None,
    batch_size: int,
    num_workers: int = 4,
    target_col: str | None = None,
    task: str = "regression",
) -> dict[str, dict]:
    """Load dataloaders for all clients in federated learning setup.
    
    Args:
        dataset_configs: List of dataset configurations, each containing:
            - name: Dataset name (ACDC, MnMs, MnMs2)
            - data_dir: Path to processed data directory
            - train_meta: Path to train metadata CSV
            - val_meta: Path to val metadata CSV (optional)
            - test_meta: Path to test metadata CSV
            - vendor_col: Column name for vendor (None for ACDC)
        views: Views to load.
        train_transform: Training augmentation.
        val_transform: Validation augmentation.
        batch_size: Batch size for dataloaders.
        num_workers: Number of data loading workers.
        target_col: Target column name (e.g., 'lv_ef' for regression, 'pathology' for classification).
        task: Task type ("classification" or "regression").
        
    Returns:
        Dictionary mapping client_id to dict with:
            - stats: ClientDataStats
            - train_loader: Training DataLoader
            - val_loader: Validation DataLoader (if available)
            - test_loader: Test DataLoader
    """
    all_clients = {}
    
    for config in dataset_configs:
        dataset_name = config["name"]
        data_dir = Path(config["data_dir"])
        
        # Load metadata
        train_df = pd.read_csv(config["train_meta"])
        val_df = pd.read_csv(config["val_meta"]) if config.get("val_meta") else None
        test_df = pd.read_csv(config["test_meta"])
        
        # Filter by client if available, otherwise use dataset as single client
        if "client" in train_df.columns:
            # Multi-client dataset - iterate over each client
            for client_id in train_df["client"].unique():
                train_client_df = train_df[train_df["client"] == client_id].reset_index(drop=True)
                val_client_df = val_df[val_df["client"] == client_id].reset_index(drop=True) if val_df is not None else None
                test_client_df = test_df[test_df["client"] == client_id].reset_index(drop=True) if test_df is not None else None
                
                # Compute client statistics
                client_stats_dict = compute_client_statistics(
                    train_df=train_client_df,
                    val_df=val_client_df,
                    test_df=test_client_df,
                    client_id=client_id,
                    target_col=target_col,
                    task=task,
                )
                
                stats = client_stats_dict[client_id]
                
                # Load datasets
                train_dataset, val_dataset, test_dataset = load_client_datasets(
                    data_dir_train=data_dir / "train",
                    data_dir_val=data_dir / "val",
                    train_df=train_client_df,
                    val_df=val_client_df,
                    test_df=test_client_df,
                    client_stats=stats,
                    views=views,
                    train_transform=train_transform,
                    val_transform=val_transform,
                    target_col=target_col,
                    task=task,
                )
        else:
            # Single client dataset - use dataset name as client_id
            client_id = dataset_name
            
            # Compute client statistics
            client_stats_dict = compute_client_statistics(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                client_id=client_id,
                target_col=target_col,
                task=task,
            )
            
            stats = client_stats_dict[client_id]
            
            # Load datasets
            train_dataset, val_dataset, test_dataset = load_client_datasets(
                data_dir_train=data_dir / "train",
                data_dir_val=data_dir / "val",
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                client_stats=stats,
                views=views,
                train_transform=train_transform,
                val_transform=val_transform,
                target_col=target_col,
                task=task,
            )

            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
            
            val_loader = None
            if val_dataset is not None:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            
            all_clients[client_id] = {
                "stats": stats,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
            }
            
            logger.info(f"Created dataloaders for {client_id}: {stats}")
    
    logger.info(f"Loaded {len(all_clients)} clients for federated learning (task={task})")
    return all_clients
