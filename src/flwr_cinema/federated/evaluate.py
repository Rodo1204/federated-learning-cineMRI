"""Federated checkpoint evaluator with unified result schema.

Thin wrapper around the unified evaluation orchestrator.
Keeps federated-specific helpers for use by the orchestrator.

Supports classification, regression, and segmentation evaluation.
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
    build_shared_argument_parser,
    compute_primary_metric_fairness,
    filter_metadata_by_client_config,
    find_best_federated_round,
    find_last_federated_round,
    get_federated_eval_fn,
    get_test_datasets_by_client,
    load_checkpoint_and_config,
    load_metadata,
    load_test_metadata,
    normalize_metric_values,
    resolve_federated_eval_task,
    resolve_federated_checkpoint_dir,
    resolve_data_dir_with_fallback,
    run_unified_evaluation,
    to_builtin,
)


def detect_model_variant(applied_head: bool, applied_decoder: bool = False) -> str:
    """Derive model variant exclusively from applied private artifacts."""
    parts = []
    if applied_head:
        parts.append("head")
    if applied_decoder:
        parts.append("decoder")
    if parts:
        return "personalized_" + "_and_".join(parts)
    return "global"


def has_private_artifacts(checkpoint: dict[str, Any]) -> bool:
    """Return whether checkpoint contains any client-private evaluation artifacts."""
    return (
        bool(checkpoint.get("client_heads"))
        or bool(checkpoint.get("client_decoders"))
    )


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
    from flwr_cinema.tasks.adapters import get_federated_task_adapter
    from flwr_cinema.scripts.lora import load_lora_state_dict
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(config)
    model = get_federated_task_adapter(cfg).build_model(cfg)

    model_state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint model payload is not a valid state_dict dictionary.")

    use_lora = bool(cfg.train.use_lora) if hasattr(cfg, "train") else False
    model.load_state_dict(model_state, strict=not use_lora)

    if use_lora and isinstance(checkpoint.get("lora_state_dict"), dict):
        load_lora_state_dict(model, checkpoint["lora_state_dict"])

    return model


def apply_private_artifacts(model: torch.nn.Module, checkpoint: dict[str, Any], client_id: str) -> dict[str, bool]:
    """Apply client-private head / decoder artifacts when present."""
    applied_head = False
    applied_decoder = False

    # --- per-client head ---
    client_heads = checkpoint.get("client_heads")
    if isinstance(client_heads, dict) and client_id in client_heads and isinstance(client_heads[client_id], dict):
        model_state = model.state_dict()
        for key, value in client_heads[client_id].items():
            if key in model_state:
                model_state[key] = value
        model.load_state_dict(model_state, strict=False)
        applied_head = True

    # --- per-client decoder (ConvUNetR with aggregate_decoder=False) ---
    client_decoders = checkpoint.get("client_decoders")
    if isinstance(client_decoders, dict) and client_id in client_decoders and isinstance(client_decoders[client_id], dict):
        model_state = model.state_dict()
        for key, value in client_decoders[client_id].items():
            if key in model_state:
                model_state[key] = value.clone()
        model.load_state_dict(model_state, strict=False)
        applied_decoder = True

    return {"local_head": applied_head, "local_decoder": applied_decoder}


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
    evaluate_fn: Any,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one client after resetting base state and applying private artifacts."""
    model.load_state_dict(base_state, strict=False)

    applied = apply_private_artifacts(model=model, checkpoint=checkpoint, client_id=client_id)
    model_variant = detect_model_variant(
        applied_head=applied["local_head"],
        applied_decoder=applied.get("local_decoder", False),
    )

    dataloader = build_eval_dataloader(dataset)
    model.to(device)
    model.eval()

    metrics = evaluate_fn(model=model, dataloader=dataloader, config=config, device=device)
    return {
        "client_id": str(client_id),
        "num_samples": int(n_samples),
        "model_variant": model_variant,
        "applied_private_artifacts": applied,
        "metrics": normalize_metric_values(metrics),
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


def load_client_datasets(config: dict[str, Any], task_type: str) -> tuple[dict[str, tuple[int, Any]], Any]:
    resolve_federated_eval_task(task_type)
    test_meta_df = load_test_metadata(config)
    datasets_by_client = get_test_datasets_by_client(config, test_meta_df)
    if not datasets_by_client:
        raise ValueError("No client test datasets found for federated evaluation.")
    evaluate_fn = get_federated_eval_fn(task_type)
    return datasets_by_client, evaluate_fn


def evaluate_clients(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    evaluate_fn: Any,
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
                evaluate_fn=evaluate_fn,
                config=config,
                device=device,
            )
        )
    return results


def run_evaluation(
    checkpoint_dir: Path,
    checkpoint_path: Path | None,
    output_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Thin wrapper around run_unified_evaluation for backward compatibility."""
    return run_unified_evaluation(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=None,
        output_dir=output_dir,
        output_file_name=output_name,
        mc_dropout=False,
        mc_dropout_passes=30,
        is_federated=True,
    )


def main() -> None:
    parser = build_shared_argument_parser()
    args = parser.parse_args()

    run_unified_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
        checkpoint_epoch=None,
        output_dir=args.output_dir,
        output_file_name=args.output_file_name,
        mc_dropout=args.mc_dropout,
        mc_dropout_passes=args.mc_dropout_passes,
        is_federated=True,
    )


if __name__ == "__main__":
    main()
