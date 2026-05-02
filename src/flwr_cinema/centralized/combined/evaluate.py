#!/usr/bin/env python3
"""Evaluate centralized classification model on test set with per-client breakdown.

Produces per-client metrics, global pool metrics, Assertivity (weighted & macro
averages), and Fairness analysis (σ, gap, min/max client).

Usage:
 python -m flwr_cinema.centralized.combined.evaluate --checkpoint-dir <dir>
 python -m flwr_cinema.centralized.combined.evaluate --checkpoint-dir <dir> --checkpoint-epoch 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, SequentialSampler

from flwr_cinema.scripts.eval_utils import (
    find_best_checkpoint,
    load_checkpoint_and_config,
    load_metadata,
    resolve_data_dir_with_fallback,
)
from flwr_cinema.scripts.lora import load_lora_state_dict
from models.cinema.cinema.classification.dataset import EndDiastoleEndSystoleDataset, get_image_transforms
from models.cinema.cinema.classification.train import classification_eval, classification_metrics
from models.cinema.cinema.device import get_amp_dtype_and_device
from models.cinema.cinema.log import get_logger
from omegaconf import OmegaConf

from flwr_cinema.centralized.combined.train import get_model

logger = get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        return super().default(obj)


def add_balanced_accuracy(
    metrics: dict[str, float], true_labels: np.ndarray, pred_labels: np.ndarray
) -> dict[str, float]:
    """Add balanced accuracy to metrics dictionary."""
    balanced_acc = balanced_accuracy_score(y_true=true_labels, y_pred=pred_labels)
    metrics["balanced_accuracy"] = float(balanced_acc)
    return metrics


def compute_fairness_metrics(
    per_client_metrics: dict[str, tuple[int, dict[str, float]]],
    primary_metric: str,
) -> dict[str, object]:
    """Compute fairness analysis metrics across clients.

    Args:
        per_client_metrics: Dict mapping client_id to (n_samples, metrics_dict).
        primary_metric: The metric key to use for fairness analysis
            (e.g., "mcc", "roc_auc", "accuracy").

    Returns:
        Dictionary with fairness statistics: std, gap, min/max client, per-client values.
    """
    per_client_primary = {}
    for client_id, (_, metrics) in per_client_metrics.items():
        if primary_metric in metrics:
            per_client_primary[client_id] = float(metrics[primary_metric])
        else:
            logger.warning(
                f"Primary metric '{primary_metric}' not found for client '{client_id}', "
                f"available: {list(metrics.keys())}. Using 0.0."
            )
            per_client_primary[client_id] = 0.0

    values = list(per_client_primary.values())
    clients = list(per_client_primary.keys())

    if len(values) < 2:
        logger.warning("Less than 2 clients - fairness metrics are degenerate.")
        return {
            "primary_metric": primary_metric,
            "std": 0.0,
            "gap": 0.0,
            "min_client": clients[0] if clients else None,
            "min_value": values[0] if values else 0.0,
            "max_client": clients[0] if clients else None,
            "max_value": values[0] if values else 0.0,
            "per_client_primary": per_client_primary,
        }

    std = float(np.std(values))
    gap = float(max(values) - min(values))
    min_idx = int(np.argmin(values))
    max_idx = int(np.argmax(values))

    return {
        "primary_metric": primary_metric,
        "std": std,
        "gap": gap,
        "min_client": clients[min_idx],
        "min_value": values[min_idx],
        "max_client": clients[max_idx],
        "max_value": values[max_idx],
        "per_client_primary": per_client_primary,
    }


def classification_eval_dataloader_with_balanced_accuracy(
    model,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate classification model and return metrics with balanced accuracy + predictions."""
    import torch.nn.functional as F

    pred_labels_list = []
    true_labels_list = []
    pred_logits_list = []

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            logits, _ = classification_eval(
                model=model,
                batch=batch,
                patch_size_dict=patch_size_dict,
                amp_dtype=amp_dtype,
                device=device,
            )
            pred_labels_list.append(torch.argmax(logits, dim=1))
            true_labels_list.append(batch["label"])
            pred_logits_list.append(logits)

    pred_labels_np = (
        torch.cat(pred_labels_list, dim=0).cpu().to(dtype=torch.float32).numpy()
    )
    true_labels_np = (
        torch.cat(true_labels_list, dim=0).cpu().to(dtype=torch.float32).numpy()
    )
    pred_logits = torch.cat(pred_logits_list, dim=0).cpu().to(dtype=torch.float32)
    pred_probs = F.softmax(pred_logits, dim=1).numpy()

    metrics = classification_metrics(
        true_labels=true_labels_np,
        pred_labels=pred_labels_np,
        pred_probs=pred_probs,
    )

    metrics = add_balanced_accuracy(metrics, true_labels_np, pred_labels_np)

    return metrics, true_labels_np, pred_labels_np


def evaluate_centralized(
    checkpoint_dir: Path,
    checkpoint_epoch: int | None = None,
    output_dir: Path | None = None,
) -> None:
    """Evaluate centralized classification model on test set with per-client breakdown."""
    checkpoint_dir = Path(checkpoint_dir)

    # Find checkpoint
    if checkpoint_epoch is not None:
        ckpt_files = sorted(checkpoint_dir.glob("**/*.pt"))
        matching = [
            p
            for p in ckpt_files
            if f"epoch_{checkpoint_epoch}" in p.name or f"_{checkpoint_epoch}" in p.name
        ]
        if matching:
            ckpt_path = matching[0]
            logger.info(f"Selected checkpoint for epoch {checkpoint_epoch}")
        else:
            logger.warning(
                f"No checkpoint found for epoch {checkpoint_epoch}, using best instead"
            )
            ckpt_path, _ = find_best_checkpoint(checkpoint_dir)
    else:
        ckpt_path, _ = find_best_checkpoint(checkpoint_dir)

    # Load checkpoint and config
    checkpoint, config = load_checkpoint_and_config(ckpt_path, checkpoint_dir)

    # Setup output directory
    if output_dir is None:
        output_dir = checkpoint_dir / "test_evaluation"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}")

    # Load model using flwr_cinema's get_model (supports LoRA, head-only)
    cfg = OmegaConf.create(config)
    use_lora = getattr(cfg.train, "use_lora", False)

    # get_model() already applies LoRA if use_lora=True (see train.py get_model function)
    model = get_model(cfg)

    # Detect if checkpoint has LoRA weights
    checkpoint_model_state = checkpoint.get("model", checkpoint)
    has_lora_in_checkpoint = (
        "lora_state_dict" in checkpoint
        or any("lora" in key.lower() for key in checkpoint_model_state.keys())
    )

    if not use_lora and has_lora_in_checkpoint:
        raise ValueError(
            "Checkpoint contains LoRA weights but config.train.use_lora=False. "
            "Enable LoRA in config (train.use_lora=true) or evaluate with a non-LoRA checkpoint."
        )

    # Load checkpoint (handles both centralized and federated formats)
    # When use_lora=True: strict=False because LoRA params are saved separately
    # When use_lora=False: strict=True to ensure exact architecture match
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=not use_lora)
        if use_lora and "lora_state_dict" in checkpoint:
            load_lora_state_dict(model, checkpoint["lora_state_dict"])
            logger.info("Loaded LoRA weights from checkpoint")
        elif use_lora:
            logger.warning(
                "use_lora=True but no 'lora_state_dict' found in checkpoint — "
                "LoRA adapters will use their initial (zero) values"
            )
    else:
        model.load_state_dict(checkpoint, strict=not use_lora)

    amp_dtype, device = get_amp_dtype_and_device()
    model.to(device)
    model.eval()
    logger.info(f"Loaded model on device {device} (use_lora={use_lora})")

    # Extract primary metric from config (for Assertivity / Fairness)
    primary_metric = getattr(cfg.train.early_stopping, "metric", "mcc")
    logger.info(f"Primary metric for analysis: {primary_metric}")

    # Load test metadata
    data_dir = Path(cfg.data.combined.dir).expanduser()
    data_dir = resolve_data_dir_with_fallback(data_dir)
    class_col = cfg.data.class_column

    train_meta_df = load_metadata(data_dir / "train_metadata.csv", metadata_type="train")
    test_meta_df = load_metadata(data_dir / "test_metadata.csv", metadata_type="test")

    # Filter by client if specified
    client_name = getattr(cfg.data, "client_name", None)
    if client_name and client_name.lower() != "all":
        logger.info(f"Evaluating on client-specific data: {client_name}")
        train_meta_df = train_meta_df[
            train_meta_df["client"].str.lower() == client_name.lower()
        ].reset_index(drop=True)
        test_meta_df = test_meta_df[
            test_meta_df["client"].str.lower() == client_name.lower()
        ].reset_index(drop=True)
        if len(test_meta_df) == 0:
            logger.warning(f"No test samples found for client: {client_name}")
    elif hasattr(cfg.federated, "client_ids") and cfg.federated.client_ids is not None:
        logger.info(f"Filtering datasets for clients: {cfg.federated.client_ids}")
        train_meta_df = train_meta_df[
            train_meta_df["client"].isin(cfg.federated.client_ids)
        ].reset_index(drop=True)
        test_meta_df = test_meta_df[
            test_meta_df["client"].isin(cfg.federated.client_ids)
        ].reset_index(drop=True)

    # Extract classes from config
    try:
        classes = list(cfg.data[class_col]) if cfg.data[class_col] else None
    except (KeyError, TypeError):
        classes = None

    if not classes:
        logger.warning(
            f"No explicit class list found in config.data.{class_col}, using sorted unique values"
        )
        classes = sorted(train_meta_df[class_col].unique().tolist())
    logger.info(f"Classes: {classes}")

    # Get image transforms
    _, val_transform = get_image_transforms(cfg)

    # Setup patch size
    views = [cfg.model.views] if isinstance(cfg.model.views, str) else cfg.model.views
    patch_size_dict = {
        v: cfg.data.sax.patch_size if v == "sax" else cfg.data.lax.patch_size for v in views
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # GLOBAL POOL EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("GLOBAL POOL EVALUATION (all test data)")
    logger.info("=" * 80)

    overall_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "test",
        meta_df=test_meta_df,
        class_col=class_col,
        classes=classes,
        views=cfg.model.views,
        transform=val_transform,
    )

    overall_dataloader = DataLoader(
        dataset=overall_dataset,
        sampler=SequentialSampler(overall_dataset),
        batch_size=1,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.train.n_workers,
    )

    global_pool_metrics, _, _ = classification_eval_dataloader_with_balanced_accuracy(
        model=model,
        dataloader=overall_dataloader,
        patch_size_dict=patch_size_dict,
        amp_dtype=amp_dtype,
        device=device,
    )

    logger.info(f"Global pool metrics: {global_pool_metrics}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PER-CLIENT EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("PER-CLIENT EVALUATION")
    logger.info("=" * 80)

    per_client_metrics: dict[str, tuple[int, dict[str, float]]] = {}
    total_samples = 0

    for client_id in sorted(test_meta_df["client"].unique()):
        client_meta = test_meta_df[test_meta_df["client"] == client_id]
        n_samples = len(client_meta)

        logger.info(f"\nClient: {client_id} ({n_samples} samples)")

        client_dataset = EndDiastoleEndSystoleDataset(
            data_dir=data_dir / "test",
            meta_df=client_meta.reset_index(drop=True),
            class_col=class_col,
            classes=classes,
            views=cfg.model.views,
            transform=val_transform,
        )

        client_dataloader = DataLoader(
            dataset=client_dataset,
            sampler=SequentialSampler(client_dataset),
            batch_size=1,
            drop_last=False,
            pin_memory=True,
            num_workers=cfg.train.n_workers,
        )

        metrics, _, _ = classification_eval_dataloader_with_balanced_accuracy(
            model=model,
            dataloader=client_dataloader,
            patch_size_dict=patch_size_dict,
            amp_dtype=amp_dtype,
            device=device,
        )

        per_client_metrics[client_id] = (n_samples, metrics)
        total_samples += n_samples

        metrics_str = {k: f"{float(v):.4f}" if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}
        logger.info(f"  Metrics: {metrics_str}")

    # ═══════════════════════════════════════════════════════════════════════════
    # ASSERTIVITY (Weighted & Macro Averages)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("ASSERTIVITY METRICS")
    logger.info("=" * 80)

    weighted_metrics: dict[str, float] = {}
    macro_metrics: dict[str, float] = {}

    if per_client_metrics:
        first_client_metrics = list(per_client_metrics.values())[0][1]
        n_clients = len(per_client_metrics)

        for metric_key in first_client_metrics.keys():
            # Weighted average (Assertivity — sample-weighted)
            weighted_sum = sum(
                n_samples * per_client_metrics[client_id][1][metric_key]
                for client_id, (n_samples, _) in per_client_metrics.items()
            )
            weighted_metrics[metric_key] = weighted_sum / total_samples

            # Macro average (Assertivity — unweighted)
            macro_sum = sum(
                per_client_metrics[client_id][1][metric_key]
                for client_id in per_client_metrics
            )
            macro_metrics[metric_key] = macro_sum / n_clients

        w_str = {k: f"{float(v):.4f}" if isinstance(v, (float, np.floating)) else v for k, v in weighted_metrics.items()}
        m_str = {k: f"{float(v):.4f}" if isinstance(v, (float, np.floating)) else v for k, v in macro_metrics.items()}
        logger.info(f"  Weighted Average: {w_str}")
        logger.info(f"  Macro Average:    {m_str}")

    # ═══════════════════════════════════════════════════════════════════════════
    # FAIRNESS METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("FAIRNESS METRICS")
    logger.info("=" * 80)

    fairness = compute_fairness_metrics(per_client_metrics, primary_metric)

    logger.info(f"  Primary metric:  {fairness['primary_metric']}")
    logger.info(f"  Std (std):       {fairness['std']:.4f}")
    logger.info(f"  Fairness Gap:    {fairness['gap']:.4f}")
    logger.info(f"  Min client:      {fairness.get('min_client')} ({fairness.get('min_value', 0):.4f})")
    logger.info(f"  Max client:      {fairness.get('max_client')} ({fairness.get('max_value', 0):.4f})")
    logger.info(f"  Per-client:      {fairness['per_client_primary']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    header = f"{'Client':<15} {'N':>6} {primary_metric:>12}  {'accuracy':>12}  {'balanced_acc':>12}"
    logger.info(header)
    logger.info("-" * len(header))
    for client_id, (n_samples, metrics) in per_client_metrics.items():
        pm_val = metrics.get(primary_metric, 0.0)
        acc_val = metrics.get("accuracy", 0.0)
        ba_val = metrics.get("balanced_accuracy", 0.0)
        logger.info(f"{client_id:<15} {n_samples:>6} {pm_val:>12.4f}  {acc_val:>12.4f}  {ba_val:>12.4f}")
    logger.info("-" * len(header))
    gp_pm = global_pool_metrics.get(primary_metric, 0.0)
    gp_acc = global_pool_metrics.get("accuracy", 0.0)
    gp_ba = global_pool_metrics.get("balanced_accuracy", 0.0)
    logger.info(f"{'Global Pool':<15} {total_samples:>6} {gp_pm:>12.4f}  {gp_acc:>12.4f}  {gp_ba:>12.4f}")
    w_pm = weighted_metrics.get(primary_metric, 0.0)
    w_acc = weighted_metrics.get("accuracy", 0.0)
    w_ba = weighted_metrics.get("balanced_accuracy", 0.0)
    logger.info(f"{'Weighted Avg':<15} {'':>6} {w_pm:>12.4f}  {w_acc:>12.4f}  {w_ba:>12.4f}")
    m_pm = macro_metrics.get(primary_metric, 0.0)
    m_acc = macro_metrics.get("accuracy", 0.0)
    m_ba = macro_metrics.get("balanced_accuracy", 0.0)
    logger.info(f"{'Macro Avg':<15} {'':>6} {m_pm:>12.4f}  {m_acc:>12.4f}  {m_ba:>12.4f}")
    logger.info(f"\nFairness std={fairness['std']:.4f}  |  Gap={fairness['gap']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS JSON
    # ═══════════════════════════════════════════════════════════════════════════
    results = {
        "config": {
            "checkpoint": ckpt_path.name,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "use_lora": use_lora,
            "primary_metric": primary_metric,
            "task": getattr(cfg, "task", "classification"),
            "class_column": class_col,
            "classes": classes,
        },
        "global_pool": global_pool_metrics,
        "per_client": {
            client_id: {"n_samples": n_samples, "metrics": metrics}
            for client_id, (n_samples, metrics) in per_client_metrics.items()
        },
        "assertivity": {
            "weighted_avg": weighted_metrics,
            "macro_avg": macro_metrics,
        },
        "fairness": fairness,
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nSaved results to {results_file}")

    logger.info("=" * 80)
    logger.info("Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate centralized classification model on test set with per-client breakdown"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Specific epoch to evaluate (overrides best selection)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (auto-generated if not specified)",
    )
    args = parser.parse_args()

    evaluate_centralized(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_epoch=args.checkpoint_epoch,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
