"""Generic federated client for all tasks.

A single concrete class that receives loss_fn and eval_dataloader_fn
via constructor injection — no ABC, no subclassing needed.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.common.logger import log

from models.cinema.cinema.convvit import param_groups_lr_decay
from models.cinema.cinema.device import get_amp_dtype_and_device
from models.cinema.cinema.optim import GradScaler
from models.cinema.cinema.train import train_one_epoch
from flwr_cinema.scripts.lora import get_lora_policy

logger = logging.getLogger(__name__)


def _to_flwr_metric_value(value: Any) -> int | float | list[int] | list[float] | None:
    """Convert metric values to Flower-supported MetricRecord types."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value.item() if isinstance(value, (np.integer, np.floating)) else value
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return float(value.item())
    if isinstance(value, (list, tuple)):
        converted: list[int] | list[float] = []
        for v in value:
            if isinstance(v, bool):
                converted.append(int(v))
            elif isinstance(v, (int, float, np.integer, np.floating)):
                converted.append(v.item() if isinstance(v, (np.integer, np.floating)) else v)
            else:
                return None
        return converted
    return None


class FederatedClient:
    """Generic federated learning client for all CineMA tasks.

    Loss and evaluation functions are injected via constructor so the same
    class serves classification, regression, and segmentation without
    subclassing.

    Args:
        client_id: Unique client identifier.
        model: Local model (initialized same as global).
        train_loader: Local training data loader.
        loss_fn: Task-specific loss function.
        views: Views to use.
        local_epochs: Number of local training epochs per round.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        clip_grad: Gradient clipping value.
        use_lora: Whether using LoRA.
        freeze_pretrained: Whether pretrained backbone is frozen.
        aggregate_head: Whether the server aggregates prediction heads (fed only).
        aggregate_decoder: Whether the server aggregates decoder (fed only).
        val_loader: Local validation data loader.
        metrics_save_path: Path to save metrics.
        stats_save_path: Path to save statistics.
        run_dir: Directory to save run files.
        eval_dataloader_fn: Function to evaluate model on a dataloader.
        config: Configuration object.
        n_accum_steps: Number of gradient accumulation steps.
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: Callable,
        views: list[str],
        local_epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        clip_grad: float | None = None,
        use_lora: bool = True,
        freeze_pretrained: bool = False,
        aggregate_head: bool = True,
        aggregate_decoder: bool = True,
        val_loader: DataLoader | None = None,
        metrics_save_path: str | None = None,
        stats_save_path: str | None = None,
        run_dir: str | None = None,
        eval_dataloader_fn: Callable | None = None,
        config: object | None = None,
        n_accum_steps: int = 1,
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.views = views
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.use_lora = use_lora
        self.freeze_pretrained = freeze_pretrained
        self.aggregate_head = aggregate_head
        self.aggregate_decoder = aggregate_decoder
        self.val_loader = val_loader
        self.metrics_save_path = metrics_save_path
        self.stats_save_path = stats_save_path
        self.run_dir = run_dir
        self.metrics_history: list[dict] = []
        self.eval_dataloader_fn = eval_dataloader_fn
        self.config = config
        self.n_accum_steps = n_accum_steps

        # Cinema sets up amp_dtype, cudnn.benchmark, empty_cache
        self.amp_dtype, self.device = get_amp_dtype_and_device()
        self.model.to(self.device)
        log(logging.INFO, f"Client {self.client_id} - Initialized on device: {self.device}")

        # Optimizer: full fine-tuning uses layer-wise decay; LoRA uses two groups.
        if not self.use_lora:
            param_groups = param_groups_lr_decay(
                model,
                no_weight_decay_list=[],
                weight_decay=config.train.weight_decay,
                layer_decay=config.train.layer_decay,
            )
        else:
            lora_params = []
            head_and_other_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "lora" in name.lower():
                    lora_params.append(param)
                else:
                    head_and_other_params.append(param)

            param_groups = [
                {"params": head_and_other_params, "weight_decay": self.weight_decay,},
                # LoRA params get no weight decay — low-rank matrices are small.
                {"params": lora_params, "weight_decay": 0.0},
            ]
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=config.train.betas, #TODO: corregir esto y pasarlo por parametro
        ) # TODO: verificar con centralized
        self.loss_scaler = GradScaler()
        self.n_samples = len(self.train_loader.dataset) if self.train_loader else 0
        self.n_val_samples = len(self.val_loader.dataset) if self.val_loader else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_patch_and_spacing_dicts(self) -> tuple[dict, dict]:
        """Compute spacing/patch size dicts for evaluation metrics."""
        views = (
            [self.config.model.views]
            if isinstance(self.config.model.views, str)
            else self.config.model.views
        )
        patch_size_dict = {
            v: self.config.data.sax.patch_size if v == "sax" else self.config.data.lax.patch_size
            for v in views
        }
        spacing_dict = {
            v: self.config.data.sax.spacing if v == "sax" else self.config.data.lax.spacing
            for v in views
        }
        return patch_size_dict, spacing_dict

    # ------------------------------------------------------------------
    # Model update / get
    # ------------------------------------------------------------------

    def update_model(self, server_params: dict) -> None:
        """Update local model with parameters from server."""
        if not self.use_lora and not self.freeze_pretrained:
            # Full fine-tuning: direct state_dict copy including buffers.
            for name, param in self.model.named_parameters():
                if name in server_params:
                    param.data.copy_(server_params[name].to(param.device))
            for name, buffer in self.model.named_buffers():
                if name in server_params:
                    buffer.data.copy_(server_params[name].to(buffer.device))
            return

        agg_head = self.aggregate_head
        policy = get_lora_policy(
            self.model,
            use_lora=self.use_lora,
            freeze_pretrained=self.freeze_pretrained,
            aggregate_head=agg_head,
        )
        policy.load(self.model, server_params)

    def get_model_update(self) -> dict:
        """Get model parameters to send to server.
        
        Always includes head so the server can archive per-client heads
        (the server strips them before aggregation when aggregate_head=False).
        """
        if self.use_lora or self.freeze_pretrained:
            policy = get_lora_policy(
                self.model,
                use_lora=self.use_lora,
                freeze_pretrained=self.freeze_pretrained,
                aggregate_head=True,  # always send head for archival
            )
            return policy.get(self.model)
        # Full fine-tuning
        return self.model.state_dict()

    # ------------------------------------------------------------------
    # Local training / validation (concrete, task injected)
    # ------------------------------------------------------------------

    def train_local(self, round_idx: int) -> int:
        """Local training loop using the injected loss_fn."""
        self.model.train()
        n_samples_trained = 0

        for local_epoch in range(self.local_epochs):
            global_epoch = round_idx * self.local_epochs + local_epoch
            self.optimizer.zero_grad(set_to_none=True)

            n_samples_trained = train_one_epoch(
                model=self.model,
                train_dataloader=self.train_loader,
                optimizer=self.optimizer,
                loss_scaler=self.loss_scaler,
                amp_dtype=self.amp_dtype,
                device=self.device,
                epoch=global_epoch,
                n_accum_steps=self.n_accum_steps,
                n_samples=n_samples_trained,
                config=self.config,
                wandb_run=None,
                loss_fn=self.loss_fn,
            )

        return self.n_samples

    def validate_local(self) -> tuple[int, dict]:
        """Local validation using the injected eval_dataloader_fn."""
        if self.val_loader is None or self.eval_dataloader_fn is None:
            return 0, {}

        patch_size_dict, spacing_dict = self.get_patch_and_spacing_dicts()

        self.model.eval()
        metrics = self.eval_dataloader_fn(
            self.model,
            self.val_loader,
            patch_size_dict,
            spacing_dict,
            self.amp_dtype,
            self.device,
        )

        return self.n_val_samples, metrics

    # ------------------------------------------------------------------
    # FedCE computation
    # ------------------------------------------------------------------

    def _compute_fedce(
        self,
        message: Message,
        model_state_dict: dict,
        last_state_dict: dict,
        updated_state_dict: dict,
        round_idx: int,
    ) -> dict | None:
        """Compute FedCE gamma metrics for the client."""
        from flwr_cinema.utils.fedCE_utils import (
            calculate_excluded_model,
            compute_fedCE_gamma_cos,
        )

        in_strategy_data = message.content.config_records.get(
            "strategy_data", ConfigRecord()
        )
        log(
            logging.INFO,
            f"Client {self.client_id} - Starting FedCE calculations for "
            f"round {round_idx} with prev_rho: "
            f"{in_strategy_data.get('prev_fedce_rho', 'N/A')}",
        )
        prev_rho = float(in_strategy_data.get("prev_fedce_rho", 0.0))
        primary_metric_name = in_strategy_data.get("primary_metric_name", None)
        primary_metric_direction = in_strategy_data.get(
            "primary_metric_direction", "max"
        )

        in_global_gradient = message.content.array_records.get(
            "global_gradient", ArrayRecord()
        )
        if in_global_gradient:
            global_gradient = in_global_gradient.to_torch_state_dict(keep_input=False)

            excluded_model = calculate_excluded_model(
                global_weights=model_state_dict,
                local_weights=last_state_dict,
                prev_rho=prev_rho,
            )
            del last_state_dict

            gamma_err = 1e-4
            # Temporarily load excluded_model to evaluate error — restore on finally.
            try:
                # Strip BN stats from excluded_model — interpolated values can be invalid.
                excluded_params_only = {
                    k: v for k, v in excluded_model.items()
                    if "running_" not in k and "num_batches_tracked" not in k
                }
                self.update_model(excluded_params_only)

                _, metrics = self.validate_local()
                primary_metric_value = (
                    metrics.get(primary_metric_name, None)
                    if primary_metric_name
                    else None
                )
                log(
                    logging.INFO,
                    f"Client {self.client_id} - FedCE primary metric value "
                    f"for round {round_idx}: {primary_metric_value}",
                )
                if primary_metric_value is not None:
                    # gamma_err: distance from perfect score (1.0 for max, 0.0 for min).
                    if primary_metric_direction == "max":
                        gamma_err = 1.0 - primary_metric_value
                    else:
                        gamma_err = primary_metric_value

                gamma_err = max(gamma_err, 1e-4)
            finally:
                self.update_model(updated_state_dict)
                del excluded_model

            gamma_cos = compute_fedCE_gamma_cos(
                global_weights=model_state_dict,
                local_weights=updated_state_dict,
                global_gradient=global_gradient,
                prev_rho=prev_rho,
            )

            log(
                logging.INFO,
                f"Client {self.client_id} - Completed FedCE calculations for "
                f"round {round_idx} with gamma_cos: {gamma_cos}, gamma_err: {gamma_err}",
            )
            return {
                "fedce_gamma_cos": float(gamma_cos),
                "fedce_gamma_err": float(gamma_err),
            }
        return None

    # ------------------------------------------------------------------
    # Flower Message API
    # ------------------------------------------------------------------

    def train(self, message: Message, context: Context) -> Message:
        """Handle train message from server."""
        in_configs = message.content.config_records.get("config", ConfigRecord())
        round_idx = in_configs.get("server-round", 0)
        strategy = in_configs.get("strategy", False)
        model_state_dict = None

        in_params = message.content.array_records.get(
            "model_params", ArrayRecord()
        )
        if in_params:
            model_state_dict = in_params.to_torch_state_dict(keep_input=False)
            self.update_model(model_state_dict)

        if strategy == "fedprox":
            # Wrap loss_fn with proximal term (after update_model, before dual LoRA).
            proximal_mu = float(in_configs.get("proximal-mu", 0.0))
            _original_loss_fn = self.loss_fn
            if proximal_mu > 0:
                from flwr_cinema.utils.fedProx_utils import create_fedprox_loss_wrapper

                self.loss_fn = create_fedprox_loss_wrapper(
                    _original_loss_fn, self.model, proximal_mu
                )

        num_samples = self.train_local(round_idx)

        if strategy == "fedprox":
            # Restore original loss_fn after training.
            if proximal_mu > 0:
                self.loss_fn = _original_loss_fn

        updated_state_dict = self.get_model_update()

        out_record = RecordDict()
        out_metrics = MetricRecord()
        out_configs = ConfigRecord()

        # Estimate optimizer steps this round (only complete accumulation groups count).
        optimizer_steps_this_round = 0
        if self.train_loader is not None:
            batches_per_epoch = len(self.train_loader)
            steps_per_epoch = batches_per_epoch // self.n_accum_steps
            optimizer_steps_this_round = steps_per_epoch * self.local_epochs

        out_configs["optimizer_steps_this_round"] = int(optimizer_steps_this_round)
        out_configs["client_id"] = str(self.client_id)
        out_record["config"] = out_configs
        out_metrics["num_samples"] = num_samples

        agg_metrics = None
        if strategy == "fedce" and round_idx > 1:
            # Clone — _compute_fedce mutates model, original dict stays clean for out_params.
            updated_state_dict = {k: v.clone() for k, v in updated_state_dict.items()}
            last_state_dict = context.state.array_records["last_model_params"].to_torch_state_dict(keep_input=False)
            agg_metrics = self._compute_fedce(
                message, model_state_dict, last_state_dict, updated_state_dict, round_idx
            )

        if agg_metrics is not None:
            out_metrics.update(agg_metrics)

        out_params = ArrayRecord.from_torch_state_dict(
            updated_state_dict, keep_input=False
        )
        out_record["model_params"] = out_params
        out_record["metrics"] = out_metrics

        context.state.array_records["last_model_params"] = out_params

        return Message(content=out_record, reply_to=message)

    def evaluate(self, message: Message, context: Context) -> Message:
        """Handle evaluate message from server."""
        in_configs = message.content.config_records.get("config", ConfigRecord())

        in_params = message.content.array_records.get(
            "model_params", ArrayRecord()
        )
        if in_params:
            model_state_dict = in_params.to_torch_state_dict(keep_input=False)
            self.update_model(model_state_dict)

        num_val_samples, metrics = self.validate_local()

        # Drop metrics whose values can't be converted to Flower types.
        safe_metrics = {
            k: converted
            for k, v in metrics.items()
            if (converted := _to_flwr_metric_value(v)) is not None
        }

        out_record = RecordDict()
        out_metrics = MetricRecord(safe_metrics)
        out_metrics["num_samples"] = num_val_samples
        out_config = ConfigRecord()
        out_config["client_id"] = str(self.client_id)
        out_record["config"] = out_config
        out_record["metrics"] = out_metrics

        return Message(content=out_record, reply_to=message)
