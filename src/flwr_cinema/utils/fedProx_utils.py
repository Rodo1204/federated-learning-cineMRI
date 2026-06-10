"""FedProx utilities: proximal term loss wrapper.

Implements the client-side proximal regularization term described in
Li et al. (2018) "Federated Optimization in Heterogeneous Networks"
(https://arxiv.org/abs/1812.06127).

The proximal term is (mu / 2) * ||w - w^t||^2, where w^t is the global
model at the start of the round and w is the current local model.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


def create_fedprox_loss_wrapper(
    original_loss_fn: Callable[..., tuple[torch.Tensor, dict[str, float]]],
    model: nn.Module,
    proximal_mu: float,
) -> Callable[..., tuple[torch.Tensor, dict[str, float]]]:
    """Wrap a loss function with the FedProx proximal term (mu/2)*||w - w^t||^2.

    Snapshots trainable parameters at call time — the caller must ensure model
    already holds global weights w^t (after update_model, before local training).

    The returned callable computes: L_task + (mu / 2) * ||w - w^t||^2.

    Args:
        original_loss_fn: Task loss callable(model, batch, views, device, **kwargs).
        model: Model whose trainable params are snapshotted as w^t.
        proximal_mu: Proximal term weight (0.0 = vanilla FedAvg).

    Returns:
        Wrapped callable with same signature as original_loss_fn.
    """
    # Snapshot all trainable parameters at wrapper-creation time.
    # Kept on the model's device so the per-batch proximal term
    # computation avoids repeated CPU<->GPU transfers.
    _snapshot: dict[str, torch.Tensor] = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    def fedprox_loss_fn(
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        views: list[str],
        device: torch.device,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Original task loss — forward everything transparently
        loss, metrics = original_loss_fn(model, batch, views, device, **kwargs)

        # 2. Proximal term: ||w - w^t||^2 across trainable params
        proximal_term = torch.tensor(0.0, device=loss.device)
        for name, param in model.named_parameters():
            if name in _snapshot:
                global_param = _snapshot[name].to(param.device)
                proximal_term += (param - global_param).norm(2) ** 2

        # 3. FedProx-regularized loss
        loss = loss + (proximal_mu / 2.0) * proximal_term

        metrics["proximal_term"] = proximal_term.item()
        return loss, metrics

    return fedprox_loss_fn
