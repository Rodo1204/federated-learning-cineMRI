"""Segmentation task: loss, eval, transforms, and dataset factories.

These factories are discovered by ``adapters.py`` via duck typing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig


def get_loss_fn():
    """Return the segmentation training loss callable."""
    from models.cinema.cinema.segmentation.train import segmentation_loss

    return segmentation_loss


def get_eval_fn():
    """Return the segmentation dataloader evaluation callable."""
    from models.cinema.cinema.segmentation.train import segmentation_eval_dataloader

    return segmentation_eval_dataloader


def get_transforms(config: DictConfig):
    """Return (train_transform, val_transform) for segmentation."""
    from models.cinema.cinema.segmentation.dataset import get_segmentation_transforms

    return get_segmentation_transforms(config)


def get_dataset(
    config: DictConfig,
    data_dir: Path,
    meta_df: Any,
    transform: Any,
    train_meta_df: Any | None = None,
):
    """Build the segmentation dataset for a split."""
    from models.cinema.cinema.segmentation.dataset import EndDiastoleEndSystoleDataset

    return EndDiastoleEndSystoleDataset(
        data_dir=data_dir,
        meta_df=meta_df,
        views=config.model.views,
        transform=transform,
    )
