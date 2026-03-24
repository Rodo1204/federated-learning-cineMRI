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
) -> Callable:
    """Wrap a loss function to use class-weighted BCE for binary classification.
    
    This wrapper transparently replaces cross-entropy with weighted BCE when
    class_distribution is provided, without modifying the original loss function.
    
    Args:
        original_loss_fn: Original loss function (e.g., classification_loss from CineMA).
        class_distribution: Dict mapping class label to count (e.g., {"0": 100, "1": 20}).
                           If None or not binary, returns original loss function.
    
    Returns:
        Wrapped loss function that uses weighted BCE for imbalanced binary classification.
    """
    # If no distribution or not binary, return original
    if class_distribution is None or len(class_distribution) != 2:
        return original_loss_fn
    
    # Compute pos_weight for binary classification
    counts = [class_distribution.get(str(k), 1) for k in [0, 1]]
    pos_weight = counts[0] / max(counts[1], 1)
    logger.info(f"Using weighted BCE with pos_weight={pos_weight:.4f} (class distribution: {class_distribution})")
    
    def weighted_loss_fn(model, batch, views, device, label_smoothing=0.1):
        """Wrapper that uses weighted BCE instead of cross-entropy."""
        # Get logits from model (replicating original logic)
        image_dict = {view: batch[f"{view}_image"].to(device) for view in views}
        logits = model(image_dict)  # Shape: [batch_size, 2] for binary classification
        label = batch["label"].long().to(device)  # Shape: [batch_size]
        
        # For binary classification: extract positive class logit (index 1)
        # logits shape [batch_size, 2] -> positive_logits shape [batch_size]
        positive_logits = logits[:, 1]
        
        # Use weighted BCE with logits
        loss = F.binary_cross_entropy_with_logits(
            positive_logits,
            label.float(),
            pos_weight=torch.tensor(pos_weight, device=device),
        )
        
        metrics = {"weighted_bce": loss.item(), "loss": loss.item()}
        return loss, metrics
    
    return weighted_loss_fn
