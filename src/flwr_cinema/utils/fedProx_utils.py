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
    """Wrap a loss function with the FedProx proximal regularization term.

    Takes a snapshot of *all trainable parameters* from ``model`` at
    wrapper-creation time — the caller is responsible for ensuring that
    ``model`` already holds the global weights ``w^t`` (i.e. after
    ``update_model()`` and before any local training).

    The returned callable accepts the four shared positional parameters
    (``model``, ``batch``, ``views``, ``device``) plus any extra keyword
    arguments that ``original_loss_fn`` may need.  This makes the wrapper
    compatible with classification (``label_smoothing``), segmentation
    (``loss_fn``), and future task loss functions without change.

    On every call the wrapper:

    1. Calls ``original_loss_fn(model, batch, views, device, **kwargs)``
       to obtain the task loss ``L_task`` and the standard metrics dict.
    2. Computes ``||w - w^t||^2`` — the squared L2 norm of the
       difference between the current *trainable* model parameters
       and the frozen global snapshot.
    3. Returns ``L_task + (mu / 2) * proximal_term`` and adds
       ``"proximal_term"`` to the metrics dict.

    Parameters
    ----------
    original_loss_fn:
        The task-specific loss callable. Must accept ``model``, ``batch``,
        ``views``, and ``device`` as its first four positional arguments.
    model:
        The model whose trainable parameters will be snapshotted as ``w^t``.
    proximal_mu:
        FedProx proximal-term weight (0.0 = vanilla FedAvg).

    Returns
    -------
    Callable with the same signature as ``original_loss_fn``.
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
