"""Task adapters shared across centralized and federated runtimes.

This module deliberately separates two concerns:

1. ``TaskSpec``
   Owns the semantic definition of a task (classification, regression, later
   segmentation): which model family it uses, which dataset builder applies,
   which transforms are needed, and which loss/evaluation callables should be
   injected into the common training loops.

2. Runtime adapters
   Wrap a ``TaskSpec`` for a concrete execution mode:
   - ``FederatedTaskAdapter`` adds Flower-specific concerns such as the client
     class and model runtime overrides, including Dual LoRA.
   - ``CentralizedTaskAdapter`` reuses the exact same task semantics for the
     centralized comparison pipeline and intentionally does NOT use Dual LoRA.

The key architectural rule is: task semantics must live in ONE place.
Centralized and federated experiments may differ in runtime mechanics, but they
should not drift in model/loss/eval/dataset definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig

from flwr_cinema.scripts.lora import apply_lora_to_convvit
from models.cinema.cinema.convvit import ConvViT

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

LossFn = Callable[..., Any]
EvalFn = Callable[..., Any]
TransformPair = tuple[Any, Any]
DatasetFactory = Callable[[DictConfig, Path, Any, Any, Any], Any]
ModelFactory = Callable[[DictConfig], torch.nn.Module]
RuntimeModelFactory = Callable[[DictConfig, "TaskSpec"], torch.nn.Module]
MetricDefaultsFactory = Callable[[], dict[str, Any]]
ClientClassFactory = Callable[[], type]


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSpec:
    """Single source of truth for task semantics across runtimes.

    Attributes:
        name: Task identifier (e.g. ``classification`` or ``regression``).
        base_model_factory: Builds the task model BEFORE runtime-specific
            concerns are applied. For class/reg today this is the same ConvViT
            builder, but the hook exists so future tasks can diverge cleanly.
        loss_fn_factory: Lazy loader for the task loss callable.
        eval_fn_factory: Lazy loader for the task evaluation callable.
        transform_factory: Returns ``(train_transform, val_transform)``.
        dataset_factory: Builds datasets for the task. The signature includes
            ``train_meta_df`` so tasks like regression can compute shared
            normalization stats from the training split.
        metric_defaults_factory: Optional extension point for future task-level
            metric metadata.
    """

    name: str
    base_model_factory: ModelFactory
    loss_fn_factory: Callable[[], LossFn]
    eval_fn_factory: Callable[[], EvalFn]
    transform_factory: Callable[[DictConfig], TransformPair]
    dataset_factory: DatasetFactory
    metric_defaults_factory: MetricDefaultsFactory | None = None


@dataclass(frozen=True)
class FederatedTaskAdapter:
    """Thin federated runtime facade over a shared :class:`TaskSpec`."""

    task_spec: TaskSpec
    client_class_factory: ClientClassFactory
    federated_model_factory: RuntimeModelFactory

    def build_model(self, config: DictConfig) -> torch.nn.Module:
        """Build the model for federated execution."""
        return self.federated_model_factory(config, self.task_spec)


@dataclass(frozen=True)
class CentralizedTaskAdapter:
    """Thin centralized runtime facade over a shared :class:`TaskSpec`."""

    task_spec: TaskSpec
    centralized_model_factory: RuntimeModelFactory

    def build_model(self, config: DictConfig) -> torch.nn.Module:
        """Build the model for centralized execution."""
        return self.centralized_model_factory(config, self.task_spec)


# ---------------------------------------------------------------------------
# Shared model construction
# ---------------------------------------------------------------------------

def _build_convvit_model(config: DictConfig) -> torch.nn.Module:
    """Build the task model for classification/regression before runtime-specific overrides.

    Today both classification and regression use ``ConvViT.from_pretrained``.
    Keeping this as a factory on ``TaskSpec`` avoids baking that assumption into
    the runtime adapters and leaves room for future tasks such as segmentation.
    """

    freeze_backbone = config.model.get("freeze_pretrained", True) or config.train.use_lora
    return ConvViT.from_pretrained(config=config, freeze=freeze_backbone)

def _build_convunetr_model(config: DictConfig) -> torch.nn.Module:
    """Build the task model for segmentation before runtime-specific overrides."""

    from models.cinema.cinema.segmentation.convunetr import ConvUNetR
    freeze_backbone = config.model.get("freeze_pretrained", True) or config.train.use_lora
    return ConvUNetR.from_pretrained(config=config, freeze=freeze_backbone)



def _apply_standard_lora(config: DictConfig, model: torch.nn.Module) -> None:
    """Apply standard LoRA adapters.

    This path is shared by centralized and federated execution.
    """

    apply_lora_to_convvit(
        model=model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        blocks_to_lora=getattr(config.lora, "blocks_to_lora", None),
    )


def _enable_head_only_training(config: DictConfig, model: torch.nn.Module) -> None:
    """Unfreeze prediction heads when using frozen-backbone head-only mode."""

    if config.model.get("freeze_pretrained", False) and hasattr(model, "pred_head_dict"):
        for head_param in model.pred_head_dict.parameters():
            head_param.requires_grad = True


def build_federated_model(config: DictConfig, task_spec: TaskSpec) -> torch.nn.Module:
    """Build the model used by the federated runtime.

    Federated execution is the ONLY runtime allowed to enable Dual LoRA.
    """

    model = task_spec.base_model_factory(config)

    if config.train.use_lora:
        if config.task == "segmentation":
            raise NotImplementedError("LoRA is not yet implemented for segmentation models.")
        if getattr(config, "dual_lora", None) and getattr(config.dual_lora, "enabled", False):
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
    """Build the model used by the centralized runtime.

    Centralized execution intentionally excludes Dual LoRA to keep this path
    simple and aligned with current experiment design.
    """

    model = task_spec.base_model_factory(config)

    if config.train.use_lora:
        if config.task == "segmentation":  # LoRA is currently only implemented for ConvViT #TODO: implement this xd
            raise NotImplementedError("LoRA is not yet implemented for segmentation models.")
        _apply_standard_lora(config, model)
    else:
        _enable_head_only_training(config, model)

    return model


# ---------------------------------------------------------------------------
# Lazy task-specific imports
# ---------------------------------------------------------------------------

def _classification_loss_fn() -> LossFn:
    """Return the classification training loss callable."""
    from models.cinema.cinema.classification.train import classification_loss

    return classification_loss


def _classification_eval_fn() -> EvalFn:
    """Return the classification dataloader evaluation callable."""
    from models.cinema.cinema.classification.train import classification_eval_dataloader

    return classification_eval_dataloader


def _classification_client_class() -> type:
    """Return the Flower client class for classification."""
    from flwr_cinema.tasks.classification.client import ClassificationClient

    return ClassificationClient


def _regression_loss_fn() -> LossFn:
    """Return the regression training loss callable."""
    from models.cinema.cinema.regression.train import regression_loss

    return regression_loss


def _regression_eval_fn() -> EvalFn:
    """Return the regression dataloader evaluation callable."""
    from models.cinema.cinema.regression.train import regression_eval_dataloader

    return regression_eval_dataloader


def _regression_client_class() -> type:
    """Return the Flower client class for regression."""
    from flwr_cinema.tasks.regression.client import RegressionClient

    return RegressionClient

def _segmentation_loss_fn() -> LossFn:
    """Return the segmentation training loss callable."""
    from models.cinema.cinema.segmentation.train import segmentation_loss

    return segmentation_loss

def _segmentation_eval_fn() -> EvalFn:
    """Return the segmentation dataloader evaluation callable."""
    from models.cinema.cinema.segmentation.train import segmentation_eval_dataloader

    return segmentation_eval_dataloader


# ---------------------------------------------------------------------------
# Shared transform and dataset builders
# ---------------------------------------------------------------------------

def _shared_transforms(config: DictConfig) -> TransformPair:
    """Return transforms shared by classification and regression.

    CineMA exposes ``get_image_transforms`` in the classification dataset
    module, but those transforms are intentionally shared by both tasks.
    """

    from models.cinema.cinema.classification.dataset import get_image_transforms

    return get_image_transforms(config)

def _segmentation_transforms(config: DictConfig) -> TransformPair:
    """Return transforms for segmentation."""

    from models.cinema.cinema.segmentation.dataset import get_segmentation_transforms

    return get_segmentation_transforms(config)


def _classification_dataset(
    config: DictConfig,
    data_dir: Path,
    meta_df: Any,
    transform: Any,
    train_meta_df: Any | None = None,  # noqa: ARG001 - kept for signature parity
) -> Any:
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


def resolve_regression_stats(
    config: DictConfig,
    train_meta_df: Any | None = None,
) -> tuple[Any, Any]:
    """Resolve regression normalization stats.

    Preference order:
    1. Use stats declared in config under ``config.data[reg_col].mean/std``.
    2. Fallback to computing them from ``train_meta_df``.
    """

    reg_col = config.data.regression_column
    stats_cfg = config.data.get(reg_col, None) if hasattr(config.data, "get") else None

    reg_mean = None
    reg_std = None
    if stats_cfg is not None:
        reg_mean = stats_cfg.get("mean", None) if hasattr(stats_cfg, "get") else getattr(stats_cfg, "mean", None)
        reg_std = stats_cfg.get("std", None) if hasattr(stats_cfg, "get") else getattr(stats_cfg, "std", None)

    if reg_mean is None or reg_std is None:
        if train_meta_df is None:
            raise ValueError("Regression stats are missing in config and train_meta_df was not provided.")
        reg_mean = train_meta_df[reg_col].mean()
        reg_std = train_meta_df[reg_col].std()

    if abs(float(reg_std)) < 1e-6:
        reg_std = 1.0

    return reg_mean, reg_std


def _regression_dataset(
    config: DictConfig,
    data_dir: Path,
    meta_df: Any,
    transform: Any,
    train_meta_df: Any | None = None,
) -> Any:
    """Build the regression dataset for a split.

    Regression needs normalization statistics derived from the TRAIN split, so
    callers must pass ``train_meta_df`` even when building validation/test
    datasets.
    """

    from models.cinema.cinema.regression.dataset import EndDiastoleEndSystoleDataset

    if train_meta_df is None:
        raise ValueError("Regression dataset factory requires train_meta_df for normalization stats.")

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

def _segmentation_dataset(
    config: DictConfig,
    data_dir: Path,
    meta_df: Any,
    transform: Any,
    train_meta_df: Any | None = None,
) -> Any:
    """Build the segmentation dataset for a split."""

    from models.cinema.cinema.segmentation.dataset import EndDiastoleEndSystoleDataset

    return EndDiastoleEndSystoleDataset(
        data_dir=data_dir,
        meta_df=meta_df,
        views=config.model.views,
        transform=transform,
    )



def _classification_metric_defaults() -> dict[str, Any]:
    """Return task-level metric defaults for classification.

    Reserved for future use. Kept as a hook so task-level metric metadata can
    be added without reshaping the adapter contract again.
    """

    return {}


def _regression_metric_defaults() -> dict[str, Any]:
    """Return task-level metric defaults for regression."""

    return {}

def _segmentation_metric_defaults() -> dict[str, Any]:
    """Return task-level metric defaults for segmentation."""

    return {}


# ---------------------------------------------------------------------------
# Task registries
# ---------------------------------------------------------------------------

TASK_SPECS: dict[str, TaskSpec] = {
    "classification": TaskSpec(
        name="classification",
        base_model_factory=_build_convvit_model,
        loss_fn_factory=_classification_loss_fn,
        eval_fn_factory=_classification_eval_fn,
        transform_factory=_shared_transforms,
        dataset_factory=_classification_dataset,
        metric_defaults_factory=_classification_metric_defaults,
    ),
    "regression": TaskSpec(
        name="regression",
        base_model_factory=_build_convvit_model,
        loss_fn_factory=_regression_loss_fn,
        eval_fn_factory=_regression_eval_fn,
        transform_factory=_shared_transforms,
        dataset_factory=_regression_dataset,
        metric_defaults_factory=_regression_metric_defaults,
    ),
    "segmentation": TaskSpec(
        name="segmentation",
        base_model_factory=_build_convunetr_model,
        loss_fn_factory=_segmentation_loss_fn,
        eval_fn_factory=_segmentation_eval_fn,
        transform_factory=_segmentation_transforms,
        dataset_factory=_segmentation_dataset,
        metric_defaults_factory=_segmentation_metric_defaults,
    ),
}


FEDERATED_CLIENT_CLASS_FACTORIES: dict[str, ClientClassFactory] = {
    "classification": _classification_client_class,
    "regression": _regression_client_class,
    "segmentation": _classification_client_class,
}


# ---------------------------------------------------------------------------
# Public lookup helpers
# ---------------------------------------------------------------------------

def _resolve_task_name(config: DictConfig) -> str:
    """Extract and validate the task name from config."""

    task = str(getattr(config, "task", "classification")).lower()
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task type: {task}")
    return task


def get_task_spec(config: DictConfig) -> TaskSpec:
    """Return the shared task specification for the current config."""

    return TASK_SPECS[_resolve_task_name(config)]


def get_federated_task_adapter(config: DictConfig) -> FederatedTaskAdapter:
    """Return the federated runtime adapter for the current task."""

    task = _resolve_task_name(config)
    return FederatedTaskAdapter(
        task_spec=TASK_SPECS[task],
        client_class_factory=FEDERATED_CLIENT_CLASS_FACTORIES[task],
        federated_model_factory=build_federated_model,
    )


def get_centralized_task_adapter(config: DictConfig) -> CentralizedTaskAdapter:
    """Return the centralized runtime adapter for the current task."""

    task = _resolve_task_name(config)
    return CentralizedTaskAdapter(
        task_spec=TASK_SPECS[task],
        centralized_model_factory=build_centralized_model,
    )


def get_task_adapter(config: DictConfig) -> FederatedTaskAdapter:
    """Backward-compatible alias.

    Prefer the explicit helpers ``get_federated_task_adapter`` and
    ``get_centralized_task_adapter`` in new code.
    """

    return get_federated_task_adapter(config)


__all__ = [
    "TaskSpec",
    "FederatedTaskAdapter",
    "CentralizedTaskAdapter",
    "TASK_SPECS",
    "get_task_spec",
    "get_federated_task_adapter",
    "get_centralized_task_adapter",
    "get_task_adapter",
    "build_federated_model",
    "build_centralized_model",
    "resolve_regression_stats",
]
