"""Federated checkpoint evaluator with unified result schema.

Supports classification/regression evaluation and explicit segmentation placeholder
until a concrete segmentation adapter is wired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler

from flwr_cinema.scripts.eval_utils import (
    compute_primary_metric_fairness,
    filter_metadata_by_client_config,
    find_best_federated_round,
    find_last_federated_round,
    get_task_adapter,
    load_checkpoint_and_config,
    load_metadata,
    load_test_metadata,
    resolve_federated_checkpoint_dir,
    resolve_data_dir_with_fallback,
    to_builtin,
)


def detect_model_variant(applied_head: bool, applied_local_lora: bool) -> str:
    """Derive model variant exclusively from applied private artifacts."""
    if applied_head and applied_local_lora:
        return "personalized_head_and_local_lora"
    if applied_head:
        return "personalized_head"
    if applied_local_lora:
        return "personalized_local_lora"
    return "global"


def has_private_artifacts(checkpoint: dict[str, Any]) -> bool:
    """Return whether checkpoint contains any client-private evaluation artifacts."""
    return bool(checkpoint.get("client_heads")) or bool(checkpoint.get("client_local_loras"))


def _extract_primary_metric_meta(config: dict[str, Any]) -> tuple[str, str]:
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    es_cfg = train_cfg.get("early_stopping", {}) if isinstance(train_cfg, dict) else {}
    metric = es_cfg.get("metric")
    mode = es_cfg.get("mode")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("Missing train.early_stopping.metric required for federated evaluation.")
    if not isinstance(mode, str) or mode.strip().lower() not in {"min", "max"}:
        raise ValueError("Missing or invalid train.early_stopping.mode (expected 'min' or 'max').")
    return metric.strip(), mode.strip().lower()


def _extract_round_from_checkpoint_name(checkpoint_path: Path) -> int | None:
    parts = checkpoint_path.stem.split("_")
    for idx, token in enumerate(parts):
        if token == "round" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def resolve_checkpoint_for_evaluation(
    checkpoint_dir: Path,
    checkpoint_path: Path | None,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Resolve checkpoint for evaluation using direct path or best-from-metrics."""
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists() or not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        return ckpt_path, {
            "mode": "direct",
            "selected_round": _extract_round_from_checkpoint_name(ckpt_path),
        }

    resolved_checkpoint_dir = resolve_federated_checkpoint_dir(checkpoint_dir)
    primary_metric, direction = _extract_primary_metric_meta(config)
    ckpt_path, round_num = find_best_federated_round(
        checkpoint_dir=resolved_checkpoint_dir,
        primary_metric=primary_metric,
        direction=direction,
    )
    return ckpt_path, {
        "mode": "best",
        "selected_round": round_num,
    }


def build_base_model(config: dict[str, Any], checkpoint: dict[str, Any]) -> torch.nn.Module:
    """Recreate model and load global/base checkpoint weights."""
    from flwr_cinema.centralized.combined.train import get_model
    from flwr_cinema.scripts.lora import load_global_lora_state_dict, load_lora_state_dict
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(config)
    model = get_model(cfg)

    model_state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint model payload is not a valid state_dict dictionary.")

    use_lora = bool(cfg.train.use_lora) if hasattr(cfg, "train") else False
    model.load_state_dict(model_state, strict=not use_lora)

    if use_lora and isinstance(checkpoint.get("lora_state_dict"), dict):
        load_lora_state_dict(model, checkpoint["lora_state_dict"])

    if isinstance(checkpoint.get("global_lora_state_dict"), dict):
        load_global_lora_state_dict(model, checkpoint["global_lora_state_dict"])

    return model


def apply_private_artifacts(model: torch.nn.Module, checkpoint: dict[str, Any], client_id: str) -> dict[str, bool]:
    """Apply client-private head/local LoRA artifacts when present."""
    from flwr_cinema.scripts.lora import load_local_lora_state_dict

    applied_head = False
    applied_local_lora = False

    client_heads = checkpoint.get("client_heads")
    if isinstance(client_heads, dict) and client_id in client_heads and isinstance(client_heads[client_id], dict):
        model_state = model.state_dict()
        for key, value in client_heads[client_id].items():
            if key in model_state:
                model_state[key] = value
        model.load_state_dict(model_state, strict=False)
        applied_head = True

    client_local_loras = checkpoint.get("client_local_loras")
    if (
        isinstance(client_local_loras, dict)
        and client_id in client_local_loras
        and isinstance(client_local_loras[client_id], dict)
    ):
        load_local_lora_state_dict(model, client_local_loras[client_id])
        applied_local_lora = True

    return {"local_head": applied_head, "local_lora": applied_local_lora}


def build_eval_dataloader(dataset: Any) -> DataLoader:
    """Build deterministic evaluation dataloader."""
    return DataLoader(
        dataset=dataset,
        sampler=SequentialSampler(dataset),
        batch_size=1,
        drop_last=False,
        pin_memory=True,
        num_workers=0,
    )


def evaluate_single_client(
    client_id: str,
    n_samples: int,
    dataset: Any,
    model: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    checkpoint: dict[str, Any],
    adapter: Any,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one client after resetting base state and applying private artifacts."""
    model.load_state_dict(base_state, strict=False)

    applied = apply_private_artifacts(model=model, checkpoint=checkpoint, client_id=client_id)
    model_variant = detect_model_variant(applied_head=applied["local_head"], applied_local_lora=applied["local_lora"])

    dataloader = build_eval_dataloader(dataset)
    model.to(device)
    model.eval()

    metrics = adapter.evaluate_fn(model=model, dataloader=dataloader, config=config, device=device)
    return {
        "client_id": str(client_id),
        "num_samples": int(n_samples),
        "model_variant": model_variant,
        "applied_private_artifacts": applied,
        "metrics": adapter.normalize_metrics_fn(metrics),
    }


def _compute_assertivity_metrics(client_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not client_results:
        return {"weighted_avg": {}, "macro_avg": {}}

    metric_names = sorted({k for result in client_results for k in result["metrics"].keys()})
    total_samples = sum(int(result["num_samples"]) for result in client_results)
    macro: dict[str, float] = {}
    weighted: dict[str, float] = {}

    for metric_name in metric_names:
        values = [float(result["metrics"][metric_name]) for result in client_results if metric_name in result["metrics"]]
        weights = [int(result["num_samples"]) for result in client_results if metric_name in result["metrics"]]
        macro[metric_name] = float(np.mean(values))
        weighted[metric_name] = float(np.average(values, weights=weights)) if total_samples > 0 else float(np.mean(values))

    return {"weighted_avg": weighted, "macro_avg": macro}


def _build_patch_size_dict(config: dict[str, Any]) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(config)
    views = cfg.model.views if isinstance(cfg.model.views, list) else [cfg.model.views]
    return {
        view: cfg.data.sax.patch_size if view == "sax" else cfg.data.lax.patch_size
        for view in views
    }


def _resolve_regression_normalization(config: dict[str, Any]) -> tuple[float, float]:
    data_dir = Path(config["data"]["combined"]["dir"]).expanduser()
    data_dir = resolve_data_dir_with_fallback(data_dir)
    train_meta_df = load_metadata(data_dir / "train_metadata.csv", metadata_type="train")
    train_meta_df = filter_metadata_by_client_config(config=config, meta_df=train_meta_df, metadata_type="train")
    reg_col = config["data"]["regression_column"]
    return float(train_meta_df[reg_col].mean()), float(train_meta_df[reg_col].std())


def _evaluate_personalized_global_pool_classification(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    datasets_by_client: dict[str, tuple[int, Any]],
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    import torch.nn.functional as F
    from sklearn.metrics import balanced_accuracy_score

    from models.cinema.cinema.classification.train import classification_eval, classification_metrics
    from models.cinema.cinema.device import get_amp_dtype_and_device

    amp_dtype, _ = get_amp_dtype_and_device()
    patch_size_dict = _build_patch_size_dict(config)

    pred_labels: list[torch.Tensor] = []
    true_labels: list[torch.Tensor] = []
    pred_logits: list[torch.Tensor] = []

    for client_id, (_, dataset) in sorted(datasets_by_client.items(), key=lambda item: str(item[0])):
        model.load_state_dict(base_state, strict=False)
        apply_private_artifacts(model=model, checkpoint=checkpoint, client_id=str(client_id))
        dataloader = build_eval_dataloader(dataset)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                logits, _ = classification_eval(
                    model=model,
                    batch=batch,
                    patch_size_dict=patch_size_dict,
                    amp_dtype=amp_dtype,
                    device=device,
                )
                pred_labels.append(torch.argmax(logits, dim=1))
                true_labels.append(batch["label"])
                pred_logits.append(logits)

    pred_labels_np = torch.cat(pred_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels_np = torch.cat(true_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    pred_logits_tensor = torch.cat(pred_logits, dim=0).cpu().to(dtype=torch.float32)
    pred_probs_np = F.softmax(pred_logits_tensor, dim=1).numpy()

    metrics = classification_metrics(
        true_labels=true_labels_np,
        pred_labels=pred_labels_np,
        pred_probs=pred_probs_np,
    )
    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y_true=true_labels_np, y_pred=pred_labels_np)
    )
    return {name: float(value) for name, value in metrics.items()}


def _evaluate_personalized_global_pool_regression(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    datasets_by_client: dict[str, tuple[int, Any]],
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    from models.cinema.cinema.device import get_amp_dtype_and_device
    from models.cinema.cinema.regression.train import regression_eval, regression_metrics

    amp_dtype, _ = get_amp_dtype_and_device()
    patch_size_dict = _build_patch_size_dict(config)
    reg_mean, reg_std = _resolve_regression_normalization(config)

    pred_labels: list[torch.Tensor] = []
    true_labels: list[torch.Tensor] = []

    for client_id, (_, dataset) in sorted(datasets_by_client.items(), key=lambda item: str(item[0])):
        model.load_state_dict(base_state, strict=False)
        apply_private_artifacts(model=model, checkpoint=checkpoint, client_id=str(client_id))
        dataloader = build_eval_dataloader(dataset)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                preds, _ = regression_eval(
                    model=model,
                    batch=batch,
                    patch_size_dict=patch_size_dict,
                    amp_dtype=amp_dtype,
                    device=device,
                )
                pred_labels.append(preds)
                true_labels.append(batch["label"])

    pred_labels_np = torch.cat(pred_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels_np = torch.cat(true_labels, dim=0).cpu().to(dtype=torch.float32).numpy()

    metrics = regression_metrics(true_labels=true_labels_np, pred_labels=pred_labels_np)
    restored_metrics = regression_metrics(
        true_labels=(true_labels_np * reg_std) + reg_mean,
        pred_labels=(pred_labels_np * reg_std) + reg_mean,
    )
    metrics.update({f"restored_{name}": float(value) for name, value in restored_metrics.items()})
    return {name: float(value) for name, value in metrics.items()}


def evaluate_global_pool(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    adapter: Any,
    config: dict[str, Any],
    task_type: str,
    datasets_by_client: dict[str, tuple[int, Any]],
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[str, dict[str, float]]:
    """Evaluate pooled test data using global model or personalized artifacts."""
    if has_private_artifacts(checkpoint):
        if task_type == "classification":
            return "personalized", _evaluate_personalized_global_pool_classification(
                model=model,
                checkpoint=checkpoint,
                config=config,
                datasets_by_client=datasets_by_client,
                base_state=base_state,
                device=device,
            )
        if task_type == "regression":
            return "personalized", _evaluate_personalized_global_pool_regression(
                model=model,
                checkpoint=checkpoint,
                config=config,
                datasets_by_client=datasets_by_client,
                base_state=base_state,
                device=device,
            )
        raise ValueError(f"Unsupported task type '{task_type}' for personalized pooled evaluation.")

    model.load_state_dict(base_state, strict=False)
    pooled_dataset = ConcatDataset([dataset for _, dataset in datasets_by_client.values()])
    pooled_dataloader = build_eval_dataloader(pooled_dataset)
    model.to(device)
    model.eval()
    metrics = adapter.evaluate_fn(model=model, dataloader=pooled_dataloader, config=config, device=device)
    return "global", adapter.normalize_metrics_fn(metrics)


def build_per_client_payload(client_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_client: dict[str, dict[str, Any]] = {}
    for result in client_results:
        client_id = str(result["client_id"])
        per_client[client_id] = {
            "n_samples": int(result["num_samples"]),
            "model_variant": str(result["model_variant"]),
            "applied_private_artifacts": result["applied_private_artifacts"],
            "metrics": result["metrics"],
        }
    return per_client


def load_client_datasets(config: dict[str, Any], task_type: str) -> tuple[Any, dict[str, tuple[int, Any]]]:
    adapter = get_task_adapter(task_type)
    test_meta_df = load_test_metadata(config)
    datasets_by_client = adapter.build_datasets_fn(config, test_meta_df)
    if not datasets_by_client:
        raise ValueError("No client test datasets found for federated evaluation.")
    return adapter, datasets_by_client


def evaluate_clients(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    adapter: Any,
    config: dict[str, Any],
    datasets_by_client: dict[str, tuple[int, Any]],
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate each client with strict reset semantics."""
    results: list[dict[str, Any]] = []
    for client_id, (n_samples, dataset) in sorted(datasets_by_client.items(), key=lambda kv: str(kv[0])):
        results.append(
            evaluate_single_client(
                client_id=str(client_id),
                n_samples=int(n_samples),
                dataset=dataset,
                model=model,
                base_state=base_state,
                checkpoint=checkpoint,
                adapter=adapter,
                config=config,
                device=device,
            )
        )
    return results


def build_result_payload(
    checkpoint_name: str,
    selection: dict[str, Any],
    config: dict[str, Any],
    global_pool_metrics: dict[str, float],
    global_pool_model_variant: str,
    client_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build federated result payload aligned with centralized evaluation output."""
    primary_metric, direction = _extract_primary_metric_meta(config)
    fairness = compute_primary_metric_fairness(client_results, primary_metric=primary_metric)
    assertivity = _compute_assertivity_metrics(client_results)

    payload = {
        "config": {
            "checkpoint": checkpoint_name,
            "checkpoint_selection_mode": selection.get("mode"),
            "checkpoint_round": selection.get("selected_round"),
            "use_lora": bool(config.get("train", {}).get("use_lora", False)),
            "primary_metric": primary_metric,
            "primary_metric_direction": direction,
            "task": str(config.get("task", "regression")),
            "global_pool_model_variant": global_pool_model_variant,
        },
        "global_pool": global_pool_metrics,
        "per_client": build_per_client_payload(client_results),
        "assertivity": assertivity,
        "fairness": fairness,
    }
    return to_builtin(payload)


def validate_result_payload(payload: dict[str, Any]) -> None:
    required_top_level = {
        "config",
        "global_pool",
        "per_client",
        "assertivity",
        "fairness",
    }
    missing = required_top_level.difference(payload.keys())
    if missing:
        raise ValueError(f"Result payload missing required keys: {sorted(missing)}")


def run_evaluation(
    checkpoint_dir: Path,
    checkpoint_path: Path | None,
    output_name: str,
    output_dir: Path | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    resolved_checkpoint_dir: Path | None = None

    if checkpoint_path is None:
        # Metrics-based checkpoint selection depends on federated output structure.
        resolved_checkpoint_dir = resolve_federated_checkpoint_dir(checkpoint_dir)
        last_ckpt, _ = find_last_federated_round(resolved_checkpoint_dir)
        _, provisional_config = load_checkpoint_and_config(last_ckpt, output_dir=resolved_checkpoint_dir)
        selected_checkpoint_path, selection = resolve_checkpoint_for_evaluation(
            checkpoint_dir=resolved_checkpoint_dir,
            checkpoint_path=None,
            config=provisional_config,
        )
    else:
        # Direct checkpoint selection must work without fragile directory assumptions.
        selected_checkpoint_path, selection = resolve_checkpoint_for_evaluation(
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
            config={},
        )

    effective_checkpoint_dir = (
        resolved_checkpoint_dir if resolved_checkpoint_dir is not None else selected_checkpoint_path.parent
    )

    # Reload exact checkpoint selected
    checkpoint, config = load_checkpoint_and_config(selected_checkpoint_path, output_dir=selected_checkpoint_path.parent)
    task_type = str(config.get("task", "regression"))
    if task_type not in {"classification", "regression", "segmentation"}:
        raise ValueError(f"Unsupported task type '{task_type}' in config.")

    model = build_base_model(config=config, checkpoint=checkpoint)
    adapter, datasets_by_client = load_client_datasets(config=config, task_type=task_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    client_results = evaluate_clients(
        model=model,
        checkpoint=checkpoint,
        adapter=adapter,
        config=config,
        datasets_by_client=datasets_by_client,
        base_state=base_state,
        device=device,
    )
    global_pool_model_variant, global_pool_metrics = evaluate_global_pool(
        model=model,
        checkpoint=checkpoint,
        adapter=adapter,
        config=config,
        task_type=task_type,
        datasets_by_client=datasets_by_client,
        base_state=base_state,
        device=device,
    )
    payload = build_result_payload(
        checkpoint_name=selected_checkpoint_path.name,
        selection=selection,
        config=config,
        global_pool_metrics=global_pool_metrics,
        global_pool_model_variant=global_pool_model_variant,
        client_results=client_results,
    )
    validate_result_payload(payload)

    results_dir = Path(output_dir) if output_dir is not None else effective_checkpoint_dir / "test_evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / output_name
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate federated checkpoints with unified schema.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="Directory containing federated checkpoints")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Direct checkpoint path. If set, metrics-based selection is skipped.")
    parser.add_argument("--output-name", default="federated_eval.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for results (defaults to checkpoint_dir/test_evaluation)")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
        output_name=args.output_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
