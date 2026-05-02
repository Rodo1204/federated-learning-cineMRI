"""Classification task: loss, eval, transforms, and dataset factories.

These factories are discovered by ``adapters.py`` via duck typing. Each task
module exposes the same set of callables so the registry can wire them into a
``TaskSpec`` without hardcoding per-task imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig


def get_loss_fn():
    """Return the classification training loss callable."""
    from models.cinema.cinema.classification.train import classification_loss

    return classification_loss


def get_eval_fn():
    """Return the classification dataloader evaluation callable."""
    from models.cinema.cinema.classification.train import classification_eval_dataloader

    return classification_eval_dataloader


def get_transforms(config: DictConfig):
    """Return (train_transform, val_transform) for classification/regression."""
    from models.cinema.cinema.classification.dataset import get_image_transforms

    return get_image_transforms(config)


def get_dataset(
    config: DictConfig,
    data_dir: Path,
    meta_df: Any,
    transform: Any,
    train_meta_df: Any | None = None,  # noqa: ARG001 — kept for signature parity
):
    """Build the classification dataset for a split."""
    from models.cinema.cinema.classification.dataset import EndDiastoleEndSystoleDataset

    class_col = config.data.class_column
    classes = config.data[class_col]
    return EndDiastoleEndSystoleDataset(
        data_dir=data_dir,
        meta_df=meta_df,
        class_col=class_col,
        classes=classes,
        views=config.model.views,
        transform=transform,
    )
