"""Runtime dropout injection for MC Dropout without modifying cinema.

Cinema models already contain ``nn.Dropout`` layers (in Attention, Mlp, ConvMlp)
but they default to ``p=0.0``. This utility sets a non-zero dropout rate at
runtime so the model can be trained with stochastic dropout and later used for
Monte Carlo Dropout uncertainty estimation — all without touching cinema source.
"""

from __future__ import annotations

import torch
from torch import nn

_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
)


def inject_mc_dropout(model: nn.Module, dropout_rate: float) -> int:
    """Set ``p`` on every ``nn.Dropout*`` module in *model* to *dropout_rate*.

    Does NOT touch ``DropPath`` (stochastic depth) or any non-Dropout modules.

    Args:
        model: PyTorch model (ConvViT, ConvUNetR, etc.).
        dropout_rate: Dropout probability to set (0.0 = disabled, 0.1 = 10%, etc.).

    Returns:
        Number of modules modified.
    """
    if not 0.0 <= dropout_rate <= 1.0:
        raise ValueError(f"dropout_rate must be in [0.0, 1.0], got {dropout_rate}")

    count = 0
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.p = dropout_rate
            count += 1

    return count
