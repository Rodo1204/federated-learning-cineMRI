import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.app import (
    Message, 
    Context, 
    RecordDict, 
    ArrayRecord, 
    ConfigRecord, 
    MetricRecord
)

from models.cinema.cinema.device import get_amp_dtype_and_device
from models.cinema.cinema.optim import GradScaler
from flwr_cinema.scripts.lora import (
    get_lora_and_head_state_dict, load_lora_and_head_state_dict,
    get_global_lora_and_head_state_dict, load_global_lora_and_head_state_dict,
    get_local_lora_state_dict, get_trainable_state_dict,
    load_trainable_state_dict, has_dual_lora, load_local_lora_state_dict
)

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

class BaseClient(ABC):
    """Abstract Base Class for Federated Learning Clients using Flower's Message API."""

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
        val_loader: DataLoader = None,
        metrics_save_path: str = None,
        stats_save_path: str = None,
        run_dir: str = None,
        eval_dataloader_fn: Callable = None,
        config: object = None,
        n_accum_steps: int = 1,
    ) -> None:
        """Initialize federated client.
        
        Args:
            client_id: Unique client identifier.
            model: Local model (initialized same as global).
            train_loader: Local training data loader.
            loss_fn: Loss function for training.
            views: List of views to use.
            local_epochs: Number of local training epochs per round.
            lr: Learning rate.
            weight_decay: Weight decay for optimizer.
            clip_grad: Gradient clipping value.
            use_lora: Whether using LoRA.
            freeze_pretrained: Whether pretrained backbone is frozen.
            val_loader: Local validation data loader.
            metrics_save_path: Path to save metrics.
            stats_save_path: Path to save statistics.
            run_dir: Directory to save run files.
            eval_dataloader_fn: Function to create evaluation data loader.
            config: Configuration object.
            n_accum_steps: Number of gradient accumulation steps.
        """
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
        self.val_loader = val_loader
        self.metrics_save_path = metrics_save_path
        self.stats_save_path = stats_save_path
        self.run_dir = run_dir
        self.metrics_history = []  # Store metrics for each round
        self.eval_dataloader_fn = eval_dataloader_fn
        self.config = config  # Store config for train_one_epoch (now DictConfig)
        self.n_accum_steps = n_accum_steps
        
        self.amp_dtype, self.device = get_amp_dtype_and_device()
        self.model.to(self.device)

        # Optimizer for local training
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay, # Better for LoRA 0.0
        ) # TODO: revisar si el optimizador es correcto con adamw en lugar de sgd
        self.loss_scaler = GradScaler()
        self.n_samples = len(self.train_loader.dataset) if self.train_loader else 0
        self.n_val_samples = len(self.val_loader.dataset) if self.val_loader else 0

    def get_patch_and_spacing_dicts(self) -> tuple[dict, dict]:
        """Helper to compute specific dicts for evaluation metrics."""
        views = [self.config.model.views] if isinstance(self.config.model.views, str) else self.config.model.views
        patch_size_dict = {
            v: self.config.data.sax.patch_size if v == "sax" else self.config.data.lax.patch_size
            for v in views
        }
        spacing_dict = {
            v: self.config.data.sax.spacing if v == "sax" else self.config.data.lax.spacing
            for v in views
        }
        return patch_size_dict, spacing_dict

    # Model Update Management
    def update_model(self, server_params: dict) -> None:
        """Update local model with parameters from server."""
        if self.use_lora:
            if has_dual_lora(self.model):
                load_global_lora_and_head_state_dict(self.model, server_params)
            else:
                load_lora_and_head_state_dict(self.model, server_params)
        elif self.freeze_pretrained:
            load_trainable_state_dict(self.model, server_params)
        else:
            for name, param in self.model.named_parameters():
                if name in server_params:
                    param.data.copy_(server_params[name].to(param.device))
            for name, buffer in self.model.named_buffers():
                if name in server_params:
                    buffer.data.copy_(server_params[name].to(buffer.device))
        # logger.debug("Updated local model with server parameters") TODO: Add logging

    def get_model_update(self) -> dict:
        """Get model parameters to send to server."""
        if self.use_lora:
            if has_dual_lora(self.model):
                return get_global_lora_and_head_state_dict(self.model)
                # update["local_lora"] = get_local_lora_state_dict(self.model)
                # return update
            else:
                return get_lora_and_head_state_dict(self.model)
        elif self.freeze_pretrained:
            return get_trainable_state_dict(self.model)
        else:
            return self.model.state_dict()

    # Abstract Methods
    @abstractmethod
    def train_local(self, round_idx: int) -> int:
        """Local training loop. Must be implemented by subclasses.
        Returns:
            num_samples: Number of training samples processed.
        """
        pass

    @abstractmethod
    def validate_local(self) -> tuple[int, dict]:
        """Local validation loop. Must be implemented by subclasses.
        Returns:
            num_val_samples: Number of validation samples processed.
            metrics: Dictionary of validation metrics.
        """
        pass

    # Flower Message API Implementations
    def train(self, message: Message, context: Context) -> Message:
        """Message handler method for training to be called directly from ClientApp decorators."""
        in_configs = message.content.config_records.get("config", ConfigRecord())
        round_idx = in_configs.get("round", 0)

        in_params = message.content.array_records.get("model_params", ArrayRecord())
        if in_params:
            model_state_dict = in_params.to_torch_state_dict(keep_input=False)
            self.update_model(model_state_dict)

        if self.use_lora and has_dual_lora(self.model):
            if "local_lora" in context.state.array_records:
                local_lora_record = context.state.array_records["local_lora"]
                local_lora_state_dict = local_lora_record.to_torch_state_dict(keep_input=False)
                load_local_lora_state_dict(self.model, local_lora_state_dict)

        num_samples = self.train_local(round_idx)

        updated_state_dict = self.get_model_update()
        out_params = ArrayRecord.from_torch_state_dict(updated_state_dict, keep_input=False)

        out_record = RecordDict()
        out_record["model_params"] = out_params

        if self.use_lora and has_dual_lora(self.model):
            local_lora_dict = get_local_lora_state_dict(self.model)
            local_lora_record = ArrayRecord.from_torch_state_dict(local_lora_dict, keep_input=False)
            context.state.array_records["local_lora"] = local_lora_record  # Store local LoRA here to avoid potential issues with flower 
            out_record["local_lora_params"] = local_lora_record # TODO: only for debugging and testing, need to be removed later

        out_configs = ConfigRecord()
        out_configs["round"] = round_idx
        
        # Calculate an estimate of the number of optimizer steps taken this round. Maybe changing this to count in the real optimizer step would be better
        optimizer_steps_this_round = 0 
        if self.train_loader is not None:
            batches_per_epoch = len(self.train_loader)
            steps_per_epoch = (batches_per_epoch + self.n_accum_steps - 1) // self.n_accum_steps
            optimizer_steps_this_round = steps_per_epoch * self.local_epochs
        out_configs["optimizer_steps_this_round"] = int(optimizer_steps_this_round)
        
        out_record["config"] = out_configs

        out_metrics = MetricRecord()
        out_metrics["num_samples"] = num_samples
        out_record["metrics"] = out_metrics

        return message.create_reply(content=out_record)

    def evaluate(self, message: Message, context: Context) -> Message:
        """Message handler method for evaluation to be called directly from ClientApp decorators."""
        in_params = message.content.array_records.get("model_params", ArrayRecord())
        if in_params:
            model_state_dict = in_params.to_torch_state_dict(keep_input=False)
            self.update_model(model_state_dict)

        if self.use_lora and has_dual_lora(self.model):
            if "local_lora" in context.state.array_records:
                local_lora_record = context.state.array_records["local_lora"]
                local_lora_state_dict = local_lora_record.to_torch_state_dict(keep_input=False)
                load_local_lora_state_dict(self.model, local_lora_state_dict)

        num_val_samples, metrics = self.validate_local()

        out_record = RecordDict()
        safe_metrics = {
            k: converted
            for k, v in metrics.items()
            if (converted := _to_flwr_metric_value(v)) is not None
        }
        out_metrics = MetricRecord(safe_metrics)
        out_metrics["num_samples"] = num_val_samples
        out_record["metrics"] = out_metrics
        return message.create_reply(content=out_record)