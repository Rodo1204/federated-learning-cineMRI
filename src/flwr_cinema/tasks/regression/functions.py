"""Regression task: loss, eval, transforms, dataset factories, and stats helpers.

These factories are discovered by ``adapters.py`` via duck typing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig


def get_loss_fn():
    """Return the regression training loss callable."""
    from models.cinema.cinema.regression.train import regression_loss

    return regression_loss


def get_eval_fn():
    """Return the regression dataloader evaluation callable."""
    from models.cinema.cinema.regression.train import regression_eval_dataloader

    return regression_eval_dataloader


def get_transforms(config: DictConfig):
    """Return (train_transform, val_transform) for classification/regression."""
    from models.cinema.cinema.classification.dataset import get_image_transforms

    return get_image_transforms(config)


def get_dataset(
    config: DictConfig,
    data_dir: Path,
    meta_df: Any,
    transform: Any,
    train_meta_df: Any | None = None,
):
    """Build the regression dataset for a split.

    Regression needs normalization statistics derived from the TRAIN split.
    """
    from models.cinema.cinema.regression.dataset import EndDiastoleEndSystoleDataset

    if train_meta_df is None:
        raise ValueError(
            "Regression dataset factory requires train_meta_df for normalization stats."
        )

    reg_col = config.data.regression_column
    reg_mean, reg_std = resolve_regression_stats(config, train_meta_df)

    return EndDiastoleEndSystoleDataset(
        data_dir=data_dir,
        meta_df=meta_df,
        reg_col=reg_col,
        reg_mean=reg_mean,
        reg_std=reg_std,
        views=config.model.views,
        transform=transform,
    )


def resolve_regression_stats(
    config: DictConfig,
    train_meta_df: Any,
) -> tuple[Any, Any]:
    """Resolve regression normalization stats.

    Preference: config > compute from train_meta_df.
    Re-exported by ``adapters.resolve_regression_stats`` for external consumers.
    """
    reg_col = config.data.regression_column
    stats_cfg = config.data.get(reg_col, None) if hasattr(config.data, "get") else None

    reg_mean = None
    reg_std = None
    if stats_cfg is not None:
        reg_mean = stats_cfg.get("mean", None) if hasattr(stats_cfg, "get") else getattr(stats_cfg, "mean", None)
        reg_std = stats_cfg.get("std", None) if hasattr(stats_cfg, "get") else getattr(stats_cfg, "std", None)

    if reg_mean is None or reg_std is None:
        reg_mean = train_meta_df[reg_col].mean()
        reg_std = train_meta_df[reg_col].std()

    if abs(float(reg_std)) < 1e-6:
        reg_std = 1.0

    return reg_mean, reg_std
