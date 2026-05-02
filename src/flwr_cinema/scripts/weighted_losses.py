"""Wrapper for class-weighted loss functions without modifying source code."""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def wrap_loss_with_class_weights(
    original_loss_fn: Callable,
    class_distribution: dict | None = None,
    classes: list | None = None,
) -> Callable:
    """Wrap a loss function to use class-weighted cross-entropy.

    Computes inverse-frequency weights per class and passes them to
    ``F.cross_entropy`` while preserving ``label_smoothing``.

    Args:
        original_loss_fn: Original loss function (unused, kept for API compat).
        class_distribution: Dict mapping class label string to count
            (e.g., {"NOR": 83, "DISEASED": 383}).
        classes: Ordered list of class names matching model output indices
            (e.g., ["NOR", "DISEASED"]). If None, falls back to sorted keys.

    Returns:
        Wrapped loss function or ``original_loss_fn`` if distribution is invalid.
    """
    if class_distribution is None or len(class_distribution) < 2:
        return original_loss_fn

    total = sum(class_distribution.values())
    n_classes = len(class_distribution)

    if classes is not None:
        weight_list = [total / (n_classes * class_distribution.get(cls, 1)) for cls in classes]
    else:
        # Fallback: alphabetic order (matches pandas sort_index)
        sorted_items = sorted(class_distribution.items())
        weight_list = [total / (n_classes * count) for _, count in sorted_items]

    weight_tensor = torch.tensor(weight_list, dtype=torch.float)
    weight_dict = dict(zip(classes or range(len(weight_list)), weight_list))
    logger.info(f"Using class-weighted CE with weights={weight_dict}")

    def weighted_loss_fn(model, batch, views, device, label_smoothing=0.1):
        image_dict = {view: batch[f"{view}_image"].to(device) for view in views}
        logits = model(image_dict)
        label = batch["label"].long().to(device)

        ce = F.cross_entropy(
            logits,
            label,
            weight=weight_tensor.to(device),
            label_smoothing=label_smoothing,
        )
        metrics = {"weighted_ce": ce.item(), "loss": ce.item()}
        return ce, metrics

    return weighted_loss_fn
