"""Task adapters: TaskSpec, runtime adapters, and auto-discovery registry.

Architecture
------------
Task-specific factories (loss, eval, transforms, dataset) live in
``tasks/<name>/functions.py``. The registry discovers them via duck typing
so adding a new task means creating a new ``functions.py`` — no changes to
this file needed.

The two runtime adapters — ``FederatedTaskAdapter`` and
``CentralizedTaskAdapter`` — wrap the same ``TaskSpec`` for their respective
execution modes. Federated is the only runtime allowed to enable Dual LoRA.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch
from omegaconf import DictConfig

from models.cinema.cinema.convvit import ConvViT
from models.cinema.cinema.segmentation.convunetr import ConvUNetR

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

LossFn = Callable[..., Any]
EvalFn = Callable[..., Any]
TransformPair = tuple[Any, Any]
DatasetFactory = Callable[[DictConfig, Path, Any, Any, Any], Any]
ModelFactory = Callable[[DictConfig], torch.nn.Module]
RuntimeModelFactory = Callable[[DictConfig, "TaskSpec"], torch.nn.Module]

_TaskFunctions = ModuleType  # duck-typed: has get_loss_fn, get_eval_fn, etc.

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """Single source of truth for task semantics across runtimes."""

    name: str
    base_model_factory: ModelFactory
    loss_fn_factory: Callable[[], LossFn]
    eval_fn_factory: Callable[[], EvalFn]
    transform_factory: Callable[[DictConfig], TransformPair]
    dataset_factory: DatasetFactory


@dataclass(frozen=True)
class FederatedTaskAdapter:
    """Thin federated runtime facade over a shared :class:`TaskSpec`."""

    task_spec: TaskSpec
    client_class_factory: Callable[[], type]
    federated_model_factory: RuntimeModelFactory

    def build_model(self, config: DictConfig) -> torch.nn.Module:
        return self.federated_model_factory(config, self.task_spec)


@dataclass(frozen=True)
class CentralizedTaskAdapter:
    """Thin centralized runtime facade over a shared :class:`TaskSpec`."""

    task_spec: TaskSpec
    centralized_model_factory: RuntimeModelFactory

    def build_model(self, config: DictConfig) -> torch.nn.Module:
        return self.centralized_model_factory(config, self.task_spec)


# ---------------------------------------------------------------------------
# Shared model construction
# ---------------------------------------------------------------------------


def _build_convvit_model(config: DictConfig) -> torch.nn.Module:
    """Build ConvViT model for classification/regression before runtime overrides."""
    freeze_backbone = config.model.get("freeze_pretrained", True) or config.train.use_lora
    return ConvViT.from_pretrained(config=config, freeze=freeze_backbone)


def _build_convunetr_model(config: DictConfig) -> torch.nn.Module:
    """Build ConvUNetR model for segmentation before runtime overrides."""
    from models.cinema.cinema.segmentation.convunetr import ConvUNetR

    freeze_backbone = config.model.get("freeze_pretrained", True) or config.train.use_lora
    return ConvUNetR.from_pretrained(config=config, freeze=freeze_backbone)


def _apply_standard_lora_to_convvit(config: DictConfig, model: torch.nn.Module) -> None:
    """Apply standard LoRA adapters (shared by centralized and federated)."""
    from flwr_cinema.scripts.lora import apply_lora_to_convvit

    apply_lora_to_convvit(
        model=model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        blocks_to_lora=getattr(config.lora, "blocks_to_lora", None),
    )

def _apply_standard_lora_to_convunetr(config: DictConfig, model: torch.nn.Module) -> None:
    """Apply standard LoRA adapters to ConvUNetR (shared by centralized and federated)."""
    from flwr_cinema.scripts.lora import apply_lora_to_convunetr

    apply_lora_to_convunetr(
        model=model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        blocks_to_lora=getattr(config.lora, "blocks_to_lora", None),
    )

def _apply_standard_lora(config: DictConfig, model: torch.nn.Module) -> None:
    """Apply standard LoRA adapters to the model based on its type."""
    if config.task == "segmentation" and config.model.name == "convunetr":
        _apply_standard_lora_to_convunetr(config, model)
    else:
        _apply_standard_lora_to_convvit(config, model)

def _enable_head_only_training(config: DictConfig, model: torch.nn.Module) -> None:
    """Unfreeze prediction heads (and decoder if applicable) for frozen-backbone head-only mode."""
    if not config.model.get("freeze_pretrained", False):
        return  # No freezing, so no need to unfreeze heads

    if isinstance(model, ConvViT) and hasattr(model, "pred_head_dict"):
        for head_param in model.pred_head_dict.parameters():
            head_param.requires_grad = True

    elif isinstance(model, ConvUNetR): # TODO: duplicated logic in apply_lora_to_convunetr, can be unified?
        for view in model.views:
            if hasattr(model, 'dec_image_conv_block_dict') and view in model.dec_image_conv_block_dict:
                for param in model.dec_image_conv_block_dict[view].parameters():
                    param.requires_grad = True
            if hasattr(model, 'dec_down_blocks_dict') and view in model.dec_down_blocks_dict:
                for param in model.dec_down_blocks_dict[view].parameters():
                    param.requires_grad = True
            if hasattr(model, 'dec_conv_blocks_dict') and view in model.dec_conv_blocks_dict:
                for param in model.dec_conv_blocks_dict[view].parameters():
                    param.requires_grad = True
            if hasattr(model, 'decoder_dict') and view in model.decoder_dict:
                for param in model.decoder_dict[view].parameters():
                    param.requires_grad = True
            if hasattr(model, 'pred_head_dict'):
                for param in model.pred_head_dict[view].parameters():
                    param.requires_grad = True


def _apply_convvit_dropout(config: DictConfig, model: torch.nn.Module) -> None:
    """Inject MC Dropout rate from config into all ``nn.Dropout`` layers.

    Cinema models (ConvViT, ConvUNetR) already contain dropout modules in
    Attention, Mlp, and ConvMlp, but they default to ``p=0.0``. This sets
    a non-zero rate at runtime so the model can be trained with stochastic
    dropout and used for Monte Carlo Dropout uncertainty — without
    modifying cinema source code.

    Called BEFORE LoRA application so LoRA dropout (controlled by its own
    config) is not overwritten.
    """
    import logging

    from flwr_cinema.models.dropout import inject_mc_dropout

    # Resolve dropout rate from the correct model config section.
    model_name = str(config.model.name).lower()
    if model_name == "convunetr":
        dropout_rate = float(config.model.convunetr.get("dropout", 0.0))
    else:
        dropout_rate = float(config.model.convvit.get("dropout", 0.0))

    if dropout_rate > 0.0:
        n = inject_mc_dropout(model, dropout_rate)
        logging.getLogger(__name__).info(
            "Injected dropout rate %.3f into %d nn.Dropout modules",
            dropout_rate,
            n,
        )


def build_federated_model(config: DictConfig, task_spec: TaskSpec) -> torch.nn.Module:
    """Build the model for federated execution (Dual LoRA allowed)."""
    model = task_spec.base_model_factory(config)

    # Inject MC Dropout rate into all nn.Dropout layers BEFORE LoRA so
    # LoRA dropout (controlled by config.lora.dropout / config.dual_lora.dropout)
    # is not overwritten. Cinema models ship with p=0.0 by default.
    _apply_convvit_dropout(config, model)

    if config.train.use_lora:
        
        if getattr(config, "dual_lora", None) and getattr(config.dual_lora, "enabled", False):
            if config.task == "segmentation":
                raise NotImplementedError("Dual LoRA is not yet implemented for segmentation models.")
            
            from flwr_cinema.scripts.lora import apply_dual_lora_to_convvit

            apply_dual_lora_to_convvit(
                model=model,
                global_rank=config.dual_lora.global_rank,
                local_rank=config.dual_lora.local_rank,
                alpha=config.dual_lora.alpha,
                dropout=config.dual_lora.dropout,
                target_modules=config.dual_lora.target_modules,
                blocks_to_lora=getattr(config.lora, "blocks_to_lora", None),
            )
        else:
            _apply_standard_lora(config, model)
    else:
        _enable_head_only_training(config, model)

    return model


def build_centralized_model(config: DictConfig, task_spec: TaskSpec) -> torch.nn.Module:
    """Build the model for centralized execution (Dual LoRA excluded)."""
    model = task_spec.base_model_factory(config)

    # Inject MC Dropout rate BEFORE LoRA (see build_federated_model).
    _apply_convvit_dropout(config, model)

    if config.train.use_lora:
        _apply_standard_lora(config, model)
    else:
        _enable_head_only_training(config, model)

    return model


# ---------------------------------------------------------------------------
# Task auto-discovery
# ---------------------------------------------------------------------------

AVAILABLE_TASKS = ("classification", "regression", "segmentation")

# Contract that every tasks/<name>/functions.py must fulfill
_REQUIRED_FUNCTIONS = ("get_loss_fn", "get_eval_fn", "get_transforms", "get_dataset")

# Per-task model factory mapping (shared models use ConvViT, segmentation uses ConvUNetR)
_TASK_MODEL_FACTORIES: dict[str, ModelFactory] = {
    "classification": _build_convvit_model,
    "regression": _build_convvit_model,
    "segmentation": _build_convunetr_model,
}


def _load_task_functions(task_name: str) -> _TaskFunctions:
    """Import and validate a task's ``functions`` module.

    Raises ``AttributeError`` immediately if the module is missing any
    required function, with a clear message pointing to the broken file.
    """
    mod = importlib.import_module(f"flwr_cinema.tasks.{task_name}.functions")
    missing = [f for f in _REQUIRED_FUNCTIONS if not hasattr(mod, f)]
    if missing:
        raise AttributeError(
            f"Task '{task_name}' functions.py lacks: {', '.join(missing)}. "
            f"See tasks/classification/functions.py for the required contract."
        )
    return mod


def _build_task_spec(task_name: str) -> TaskSpec:
    """Build a TaskSpec from a task's functions module."""
    funcs = _load_task_functions(task_name)
    return TaskSpec(
        name=task_name,
        base_model_factory=_TASK_MODEL_FACTORIES[task_name],
        loss_fn_factory=funcs.get_loss_fn,
        eval_fn_factory=funcs.get_eval_fn,
        transform_factory=funcs.get_transforms,
        dataset_factory=funcs.get_dataset,
    )


def _resolve_task_name(config: DictConfig) -> str:
    """Extract and validate the task name from config."""
    task = str(getattr(config, "task", "classification")).lower()
    if task not in AVAILABLE_TASKS:
        raise ValueError(f"Unknown task type: {task}")
    return task


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_task_spec(config: DictConfig) -> TaskSpec:
    """Return the shared task specification for the current config."""
    return _build_task_spec(_resolve_task_name(config))


def get_federated_task_adapter(config: DictConfig) -> FederatedTaskAdapter:
    """Return the federated runtime adapter for the current task."""
    from flwr_cinema.tasks.client import FederatedClient

    task = _resolve_task_name(config)
    return FederatedTaskAdapter(
        task_spec=_build_task_spec(task),
        client_class_factory=lambda: FederatedClient,
        federated_model_factory=build_federated_model,
    )


def get_centralized_task_adapter(config: DictConfig) -> CentralizedTaskAdapter:
    """Return the centralized runtime adapter for the current task."""
    task = _resolve_task_name(config)
    return CentralizedTaskAdapter(
        task_spec=_build_task_spec(task),
        centralized_model_factory=build_centralized_model,
    )


def get_task_adapter(config: DictConfig) -> FederatedTaskAdapter:
    """Backward-compatible alias for ``get_federated_task_adapter``."""
    return get_federated_task_adapter(config)


def resolve_regression_stats(
    config: DictConfig,
    train_meta_df: Any,
) -> tuple[Any, Any]:
    """Resolve regression normalization stats (re-exported from task module)."""
    from flwr_cinema.tasks.regression.functions import (
        resolve_regression_stats as _resolve,
    )

    return _resolve(config, train_meta_df)


__all__ = [
    "TaskSpec",
    "FederatedTaskAdapter",
    "CentralizedTaskAdapter",
    "get_task_spec",
    "get_federated_task_adapter",
    "get_centralized_task_adapter",
    "get_task_adapter",
    "build_federated_model",
    "build_centralized_model",
    "resolve_regression_stats",
]
