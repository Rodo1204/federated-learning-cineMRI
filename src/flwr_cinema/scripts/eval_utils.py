"""Shared evaluation utilities for centralized and federated models.

Covers metadata loading, checkpoint selection, federated round selection,
client-split dataset creation, per-client evaluation, fairness metrics,
Monte Carlo Dropout uncertainty estimation, and clinical metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler

from flwr_cinema.tasks.adapters import get_task_spec

try:  # pragma: no cover - import availability depends on local environment
    from models.cinema.cinema.classification.dataset import (
        get_image_transforms,
        EndDiastoleEndSystoleDataset as ClassificationDataset,
    )
    from models.cinema.cinema.device import get_amp_dtype_and_device
    from models.cinema.cinema.regression.dataset import EndDiastoleEndSystoleDataset as RegressionDataset
    from models.cinema.cinema.regression.train import regression_eval, regression_metrics
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    get_image_transforms = None
    get_amp_dtype_and_device = None
    ClassificationDataset = Any
    RegressionDataset = Any
    regression_eval = None
    regression_metrics = None

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_metadata(metadata_path: Path, metadata_type: str = "test") -> pd.DataFrame:
    """Load metadata CSV with fallback to cache directory.
    
    Attempts to load metadata from the specified path. If not found, falls back to
    a hardcoded cache directory at ~/.cache/cinema_datasets/acdc_mnms_combined/processed/.
    
    Args:
        metadata_path: Primary path to metadata CSV file.
        metadata_type: Type of metadata (train/val/test) for logging/error messages.
    
    Returns:
        Loaded metadata DataFrame.
    
    Raises:
        FileNotFoundError: If metadata not found in either location.
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        # Fall back to hardcoded cache directory
        cache_path = Path.home() / ".cache" / "cinema_datasets" / "acdc_mnms_combined" / "processed" / metadata_path.name
        if cache_path.exists():
            metadata_path = cache_path
            logger.info(f"{metadata_type.capitalize()} metadata not found in config directory, using cache: {metadata_path}")
        else:
            logger.error(f"{metadata_type.capitalize()} metadata not found at {metadata_path} or {cache_path}. "
                        f"Please ensure {metadata_path.name} exists in either location.")
            raise FileNotFoundError(f"{metadata_type.capitalize()} metadata not found at {metadata_path} or {cache_path}")
    
    df = pd.read_csv(metadata_path, dtype={"pid": str})
    logger.info(f"Loaded {metadata_type} metadata with {len(df)} samples from {metadata_path}")
    return df


def resolve_data_dir_with_fallback(data_dir: Path) -> Path:
    """Resolve data directory with fallback to cache.
    
    Attempts to use the specified data_dir. If it doesn't exist, falls back to
    a hardcoded cache directory at ~/.cache/cinema_datasets/acdc_mnms_combined/processed/.
    
    Args:
        data_dir: Primary data directory path.
    
    Returns:
        Resolved data directory path (either original or fallback).
    
    Raises:
        FileNotFoundError: If neither location exists.
    """
    data_dir = Path(data_dir)
    
    if data_dir.exists():
        return data_dir
    
    # Try fallback cache directory
    cache_data_dir = Path.home() / ".cache" / "cinema_datasets" / "acdc_mnms_combined" / "processed"
    if cache_data_dir.exists():
        logger.info(f"Data directory not found at {data_dir}, using cache: {cache_data_dir}")
        return cache_data_dir
    
    logger.error(f"Data directory not found at {data_dir} or {cache_data_dir}. "
                f"Please ensure data exists in either location.")
    raise FileNotFoundError(f"Data directory not found at {data_dir} or {cache_data_dir}")


def find_best_checkpoint(checkpoint_dir: Path) -> tuple[Path, int | None]:
    """Find the best checkpoint by canonical validation metric history.

    Uses nearby validation history (validation_metrics.jsonl) and early-stopping
    config (config.yaml / .hydra/config.yaml). Exact mapping reads only
    primary_validation_metrics. Legacy histories without this field are treated
    as not reliably mappable and trigger heuristic fallback.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Path to best checkpoint and its epoch number (if found).
    """
    def _extract_epoch_from_name(path: Path) -> int | None:
        name = path.stem
        parts = name.split("_")
        for i, part in enumerate(parts):
            if part == "epoch" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    checkpoint_dir = Path(checkpoint_dir)

    # Search for all .pt files in checkpoint directory and subdirectories.
    pt_files = sorted(checkpoint_dir.glob("**/*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint files found in {checkpoint_dir}")

    # Map epoch -> checkpoint path, preserving deterministic order.
    epoch_to_ckpt: dict[int, Path] = {}
    for ckpt in pt_files:
        epoch = _extract_epoch_from_name(ckpt)
        if epoch is not None and epoch not in epoch_to_ckpt:
            epoch_to_ckpt[epoch] = ckpt

    # Search nearby directories for config and validation history.
    search_dirs = [
        checkpoint_dir,
        checkpoint_dir.parent,
        checkpoint_dir.parent.parent,
        checkpoint_dir.parent.parent.parent,
    ]

    config_path = next(
        (
            p
            for d in search_dirs
            for p in (d / "config.yaml", d / ".hydra" / "config.yaml")
            if p.exists()
        ),
        None,
    )
    metrics_path = next(
        (
            (d / "validation_metrics.jsonl")
            for d in search_dirs
            if (d / "validation_metrics.jsonl").exists()
        ),
        None,
    )

    if config_path is not None and metrics_path is not None:
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(config_path)
            metric_name = str(cfg.train.early_stopping.metric)
            mode = str(cfg.train.early_stopping.mode).lower()

            if mode not in {"min", "max"}:
                logger.warning(
                    f"Invalid early_stopping.mode='{mode}' in {config_path}; expected 'min' or 'max'."
                )
            else:
                best_ckpt: Path | None = None
                best_epoch: int | None = None
                best_value = float("inf") if mode == "min" else float("-inf")

                history_reliably_mappable = True

                with open(metrics_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        epoch = record.get("epoch")
                        if not isinstance(epoch, int):
                            continue

                        metric_block = record.get("primary_validation_metrics")
                        if not isinstance(metric_block, dict):
                            history_reliably_mappable = False
                            logger.warning(
                                "validation_metrics.jsonl is missing 'primary_validation_metrics' "
                                f"for epoch {epoch}; falling back to heuristic checkpoint selection "
                                "(NOT true best)."
                            )
                            break

                        # Primary lookup uses configured key; fallback supports prefixed variants.
                        metric_value = metric_block.get(metric_name)
                        if metric_value is None and isinstance(metric_name, str):
                            prefixed_name = (
                                metric_name[4:]
                                if metric_name.startswith("val_")
                                else f"val_{metric_name}"
                            )
                            metric_value = metric_block.get(prefixed_name)

                        if not isinstance(metric_value, (int, float)):
                            continue

                        ckpt = epoch_to_ckpt.get(epoch)
                        if ckpt is None:
                            continue

                        is_better = (
                            metric_value < best_value
                            if mode == "min"
                            else metric_value > best_value
                        )
                        if is_better:
                            best_value = float(metric_value)
                            best_epoch = epoch
                            best_ckpt = ckpt

                if history_reliably_mappable and best_ckpt is not None:
                    logger.info(
                        f"Selected best checkpoint from validation history: {best_ckpt.name} "
                        f"(epoch: {best_epoch}, {metric_name}: {best_value:.6g}, mode: {mode})"
                    )
                    return best_ckpt, best_epoch

                if history_reliably_mappable:
                    logger.warning(
                        "Could not map validation history to saved checkpoints; "
                        "falling back to heuristic checkpoint selection (NOT true best)."
                    )
        except Exception as e:
            logger.warning(
                "Failed to select checkpoint from validation history "
                f"(config: {config_path}, metrics: {metrics_path}): {e}. "
                "Falling back to heuristic checkpoint selection (NOT true best)."
            )
    else:
        logger.warning(
            "Missing nearby config.yaml/.hydra/config.yaml or validation_metrics.jsonl; "
            "falling back to heuristic checkpoint selection (NOT true best)."
        )

    # Fallback heuristic: prefer explicitly named best checkpoint, else latest.
    best_ckpts = [p for p in pt_files if "best" in p.name.lower()]
    best_ckpt = (
        best_ckpts[0]
        if best_ckpts
        else max(pt_files, key=lambda p: p.stat().st_mtime)
    )
    best_epoch = _extract_epoch_from_name(best_ckpt)

    logger.info(
        f"Selected fallback checkpoint: {best_ckpt.name} (epoch: {best_epoch})"
    )
    return best_ckpt, best_epoch


def resolve_federated_checkpoint_dir(checkpoint_dir: Path) -> Path:
    """Resolve federated checkpoints directory.

    Contract:
    - preferred input is direct .../federated_server/checkpoints
    - experiment root containing federated_server/checkpoints is accepted
    """
    candidate = Path(checkpoint_dir)
    if candidate.exists() and candidate.is_dir() and candidate.name == "checkpoints":
        return candidate

    nested = candidate / "federated_server" / "checkpoints"
    if nested.exists() and nested.is_dir():
        return nested

    raise FileNotFoundError(
        f"Federated checkpoint directory not found from '{checkpoint_dir}'. "
        "Expected a checkpoints dir or an experiment root containing federated_server/checkpoints."
    )


def _resolve_federated_server_and_checkpoints_dirs(checkpoint_dir: Path) -> tuple[Path, Path]:
    checkpoints_dir = resolve_federated_checkpoint_dir(checkpoint_dir)
    server_dir = checkpoints_dir.parent
    return server_dir, checkpoints_dir


def _extract_round_number_from_checkpoint_name(path: Path) -> int | None:
    stem = path.stem
    if "round" not in stem:
        return None
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if part == "round" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _candidate_checkpoint_paths_for_round(server_dir: Path, checkpoints_dir: Path, round_num: int) -> list[Path]:
    _ = server_dir
    suffix = f"{round_num:03d}.pt"
    return [
        checkpoints_dir / f"global_model_round_{suffix}",
        checkpoints_dir / f"global_model_final_round_{suffix}",
    ]


def _find_ckpt_round_steps_file(checkpoints_dir: Path, round_num: int) -> Path | None:
    candidates = sorted(checkpoints_dir.glob(f"ckpt_round_{round_num}_steps_*.pt"))
    if not candidates:
        return None

    def _extract_steps(path: Path) -> int:
        parts = path.stem.split("_")
        for i, part in enumerate(parts):
            if part == "steps" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return -1
        return -1

    return max(candidates, key=lambda p: (_extract_steps(p), p.stat().st_mtime))


def find_last_federated_round(checkpoint_dir: Path) -> tuple[Path, int]:
    """Find the last (most recent) federated global model checkpoint."""
    _, checkpoints_dir = _resolve_federated_server_and_checkpoints_dirs(checkpoint_dir)

    # Prefer true per-round checkpoints first
    round_candidates = sorted(
        list(checkpoints_dir.glob("global_model_round_*.pt"))
        + list(checkpoints_dir.glob("ckpt_round_*_steps_*.pt"))
    )

    best_with_round: tuple[int, Path] | None = None
    for candidate in round_candidates:
        round_num = _extract_round_number_from_checkpoint_name(candidate)
        if round_num is None:
            continue
        if best_with_round is None or round_num > best_with_round[0]:
            best_with_round = (round_num, candidate)

    if best_with_round is not None:
        round_num, last_ckpt = best_with_round
        logger.info(f"Selected last federated round: {last_ckpt.name} (round: {round_num})")
        return last_ckpt, round_num

    # If no per-round checkpoints exist, fall back to final round snapshots
    final_round_candidates = sorted(list(checkpoints_dir.glob("global_model_final_round_*.pt")))
    best_final_with_round: tuple[int, Path] | None = None
    for candidate in final_round_candidates:
        round_num = _extract_round_number_from_checkpoint_name(candidate)
        if round_num is None:
            continue
        if best_final_with_round is None or round_num > best_final_with_round[0]:
            best_final_with_round = (round_num, candidate)

    if best_final_with_round is not None:
        round_num, final_ckpt = best_final_with_round
        logger.info(
            f"Selected fallback final federated checkpoint: {final_ckpt.name} (round: {round_num})"
        )
        return final_ckpt, round_num

    fallback_candidates = [checkpoints_dir / "global_model_final.pt"]
    for fallback_ckpt in fallback_candidates:
        if fallback_ckpt.exists():
            logger.info(f"Selected last federated checkpoint without round metadata: {fallback_ckpt.name}")
            return fallback_ckpt, 0

    raise FileNotFoundError(
        f"No global model checkpoints found in {checkpoints_dir}"
    )


def _resolve_checkpoint_path_for_round(server_dir: Path, checkpoints_dir: Path, round_num: int) -> Path | None:
    ckpt_round_steps = _find_ckpt_round_steps_file(checkpoints_dir, round_num)
    if ckpt_round_steps is not None:
        return ckpt_round_steps

    for candidate in _candidate_checkpoint_paths_for_round(server_dir, checkpoints_dir, round_num):
        if candidate.exists():
            return candidate
    return None


def _load_federated_metrics_history(server_dir: Path) -> list[dict[str, Any]]:
    metrics_file = server_dir / "metrics_history.json"
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Missing metrics history under {server_dir}; expected metrics_history.json"
        )

    try:
        with open(metrics_file, encoding="utf-8") as f:
            metrics_history = json.load(f)
    except Exception as e:  # pragma: no cover - defensive
        raise ValueError(f"Unable to parse metrics_history.json: {e}") from e

    # New format (v2): {"rounds": [{ "round": N, "evaluate_metrics": {...}, ...}, ...]}
    if isinstance(metrics_history, dict) and "rounds" in metrics_history:
        rounds = metrics_history["rounds"]
        if not isinstance(rounds, list) or not rounds:
            raise ValueError("metrics_history.json has empty or invalid 'rounds' array")

        normalized_entries: list[dict[str, Any]] = []
        for round_data in rounds:
            if not isinstance(round_data, dict):
                continue

            evaluate_metrics = round_data.get("evaluate_metrics")
            # Skip rounds without evaluation (no evaluate_metrics or empty dict)
            if not isinstance(evaluate_metrics, dict) or not evaluate_metrics:
                continue

            round_num = round_data.get("round")
            if not isinstance(round_num, int):
                continue

            # Normalize to the legacy fedavg_metrics contract: wrap flat metrics under
            # fedavg_metrics so that find_best_federated_round works transparently.
            normalized_entries.append({"round": round_num, "fedavg_metrics": evaluate_metrics})

        if not normalized_entries:
            raise ValueError("metrics_history.json has no validation rounds with evaluate_metrics")
        return normalized_entries

    # Legacy format (v1): dict with evaluate_metrics keyed by round.
    if isinstance(metrics_history, dict):
        evaluate_metrics = metrics_history.get("evaluate_metrics")
        if not isinstance(evaluate_metrics, dict) or not evaluate_metrics:
            raise ValueError("metrics_history.json missing or empty 'evaluate_metrics'")

        normalized_entries: list[dict[str, Any]] = []
        for round_key, round_payload in evaluate_metrics.items():
            try:
                round_num = int(round_key)
            except (TypeError, ValueError):
                continue

            if not isinstance(round_payload, dict):
                continue

            if isinstance(round_payload.get("fedavg_metrics"), dict):
                fedavg_metrics = round_payload["fedavg_metrics"]
            else:
                # Accept flat metric maps as best-effort compatibility.
                fedavg_metrics = round_payload

            normalized_entries.append({"round": round_num, "fedavg_metrics": fedavg_metrics})

        if not normalized_entries:
            raise ValueError("metrics_history.json has no usable evaluate_metrics round entries")
        return normalized_entries

    # Keep compatibility if old list-based metrics were produced.
    if isinstance(metrics_history, list) and metrics_history:
        return metrics_history

    raise ValueError("metrics_history.json must be a dict with non-empty 'evaluate_metrics'")


def find_best_federated_round(
    checkpoint_dir: Path,
    primary_metric: str,
    direction: str,
) -> tuple[Path, int]:
    """Select best federated round from configured primary metric.

    Args:
        checkpoint_dir: Federated checkpoint directory or experiment root.
        primary_metric: Metric key expected inside each fedavg_metrics payload.
        direction: Optimization direction, one of min or max.

    Returns:
        Tuple of checkpoint path and round number.

    Raises:
        ValueError: If config metadata is invalid, metric values are missing/non-numeric,
            history is empty, or no valid round can be selected.
        FileNotFoundError: If required files/directories/checkpoints are missing.
    """
    try:
        mode = direction.lower().strip()
        if mode not in {"min", "max"}:
            raise ValueError(
                f"Invalid direction '{direction}' for best-round selection; expected 'min' or 'max'."
            )

        federated_server_dir, resolved_ckpt_dir = _resolve_federated_server_and_checkpoints_dirs(checkpoint_dir)
        metrics_history = _load_federated_metrics_history(federated_server_dir)

        best_round: int | None = None
        best_metric: float | None = None

        for entry in metrics_history:
            if not isinstance(entry, dict):
                continue

            round_num = entry.get("round")
            if not isinstance(round_num, int):
                raise ValueError(f"Invalid round entry in metrics history: {entry}")

            fedavg_metrics = entry.get("fedavg_metrics")
            if not isinstance(fedavg_metrics, dict):
                raise ValueError(
                    f"Round {round_num} missing 'fedavg_metrics'; cannot evaluate primary metric '{primary_metric}'."
                )

            metric_value = fedavg_metrics.get(primary_metric)
            if not isinstance(metric_value, (int, float)):
                raise ValueError(
                    f"Round {round_num} has missing/non-numeric primary metric '{primary_metric}' in fedavg_metrics."
                )

            value = float(metric_value)
            if best_metric is None:
                best_metric = value
                best_round = round_num
                continue

            is_better = value < best_metric if mode == "min" else value > best_metric
            if is_better:
                best_metric = value
                best_round = round_num

        if best_round is None:
            raise ValueError(
                f"Could not select best round using primary metric '{primary_metric}' ({mode})."
            )

        ckpt_path = _resolve_checkpoint_path_for_round(
            server_dir=federated_server_dir,
            checkpoints_dir=resolved_ckpt_dir,
            round_num=best_round,
        )
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Selected best round {best_round} by metric '{primary_metric}' ({mode}) "
                "but no matching checkpoint file was found in checkpoints or server directory."
            )

        logger.info(
            "Selected best federated round %s using %s=%s (%s)",
            best_round,
            primary_metric,
            f"{best_metric:.6g}",
            mode,
        )
        return ckpt_path, best_round
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "Best-round selection unavailable (%s). Falling back to last-round checkpoint.",
            exc,
        )
        return find_last_federated_round(checkpoint_dir)


def _load_config_near_checkpoint(
    checkpoint_path: Path,
    output_dir: Path | None = None,
) -> dict:
    """Find and load the config YAML near a checkpoint, without loading the model.

    Searches parent directories of both output_dir and checkpoint_path
    for config.yaml (including .hydra/config.yaml).  Falls back to the
    project default configs/cinema_fed.yaml when nothing else is found.

    Args:
        checkpoint_path: Path to a checkpoint (.pt) file (used to anchor the search).
        output_dir: Optional output directory to also search from.

    Returns:
        Parsed config as a plain dict (via OmegaConf.to_container).
    """
    config_paths: list[Path] = []
    if output_dir:
        config_paths.extend([
            output_dir / "config.yaml",
            output_dir.parent / "config.yaml",
            output_dir.parent.parent / "config.yaml",
            output_dir / ".hydra" / "config.yaml",
            output_dir.parent / ".hydra" / "config.yaml",
        ])

    config_paths.extend([
        checkpoint_path.parent / "config.yaml",
        checkpoint_path.parent.parent / "config.yaml",
        checkpoint_path.parent.parent.parent / "config.yaml",
        checkpoint_path.parent.parent / ".hydra" / "config.yaml",
        checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml",
    ])

    for config_path in config_paths:
        if config_path.exists():
            try:
                from omegaconf import OmegaConf
                config_dict = OmegaConf.to_container(OmegaConf.load(config_path))
                logger.info("Loaded config from %s", config_path)
                return config_dict  # type: ignore[return-value]
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", config_path, e)

    # Fall back to default config from configs/ directory
    try:
        from omegaconf import OmegaConf
        repo_root = Path(__file__).parent.parent
        default_config_path = repo_root / "configs" / "cinema_fed.yaml"
        if default_config_path.exists():
            logger.warning("Using default config from %s", default_config_path)
            return OmegaConf.to_container(OmegaConf.load(default_config_path))  # type: ignore[return-value]
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not find config.yaml for checkpoint {checkpoint_path}"
    )


def load_checkpoint_and_config(
    checkpoint_path: Path, 
    output_dir: Path | None = None
) -> tuple[dict, dict]:
    """Load checkpoint state dict and associated config.
    
    Args:
        checkpoint_path: Path to checkpoint (.pt file).
        output_dir: Output directory containing config.yaml (optional, will search parents).
    
    Returns:
        Tuple of (checkpoint_dict, config_dict)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    config_dict = _load_config_near_checkpoint(checkpoint_path, output_dir)
    
    return checkpoint, config_dict


def _detect_client_column(meta_df: pd.DataFrame) -> str | None:
    if "client" in meta_df.columns:
        return "client"
    if "vendor" in meta_df.columns:
        return "vendor"
    return None


def filter_metadata_by_client_config(
    config: dict,
    meta_df: pd.DataFrame,
    metadata_type: str,
) -> pd.DataFrame:
    """Apply centralized-style client filtering to metadata when configured."""
    if meta_df.empty:
        return meta_df

    client_col = _detect_client_column(meta_df)
    if client_col is None:
        logger.info(f"Skipping {metadata_type} client filtering: no client/vendor column present.")
        return meta_df

    filtered_df = meta_df
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    federated_cfg = config.get("federated", {}) if isinstance(config, dict) else {}

    client_name = data_cfg.get("client_name")
    if isinstance(client_name, str) and client_name.strip() and client_name.lower() != "all":
        logger.info(f"Filtering {metadata_type} metadata for client: {client_name}")
        filtered_df = filtered_df[
            filtered_df[client_col].astype(str).str.lower() == client_name.lower()
        ].reset_index(drop=True)
        return filtered_df

    client_ids = federated_cfg.get("client_ids")
    if client_ids is not None:
        if isinstance(client_ids, str):
            normalized_client_ids = {client_ids}
        else:
            normalized_client_ids = {str(client_id) for client_id in client_ids}
        logger.info(f"Filtering {metadata_type} metadata for clients: {sorted(normalized_client_ids)}")
        filtered_df = filtered_df[
            filtered_df[client_col].astype(str).isin(normalized_client_ids)
        ].reset_index(drop=True)

    return filtered_df


def load_test_metadata(config: dict) -> pd.DataFrame:
    """Load test metadata from combined dataset.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        DataFrame with test metadata including client column.
    """
    data_dir = Path(config["data"]["combined"]["dir"]).expanduser()
    data_dir = resolve_data_dir_with_fallback(data_dir)
    metadata_path = data_dir / "test_metadata.csv"
    test_meta_df = load_metadata(metadata_path, metadata_type="test")
    return filter_metadata_by_client_config(config=config, meta_df=test_meta_df, metadata_type="test")


def resolve_federated_eval_task(task: str | dict[str, Any] | Any) -> str:
    """Resolve and validate task name for federated evaluation."""
    task_name = str(task.get("task") if isinstance(task, dict) else task).lower() if not isinstance(task, str) else task.lower()

    if task_name not in {"classification", "regression", "segmentation"}:
        raise ValueError(f"Unsupported task type for federated evaluation: {task_name}")
    return task_name


def get_test_datasets_by_client(
    config: dict,
    test_meta_df: pd.DataFrame,
) -> dict[str, tuple[int, ClassificationDataset | RegressionDataset]]:
    """Create test datasets split by client for the configured task.
    
    Args:
        config: Configuration dictionary.
        test_meta_df: Test metadata DataFrame.
    
    Returns:
        Dictionary mapping client_id to (n_samples, dataset).
    """
    if get_task_spec is None:
        raise RuntimeError("Task adapter helpers are unavailable in this environment.")

    from omegaconf import OmegaConf
    
    data_dir = Path(config["data"]["combined"]["dir"]).expanduser()
    data_dir = resolve_data_dir_with_fallback(data_dir)
    cfg = OmegaConf.create(config)
    
    task = resolve_federated_eval_task(config)
    task_spec = get_task_spec(cfg)
    
    _, transform = task_spec.transform_factory(cfg)
    train_meta_df = load_metadata(data_dir / "train_metadata.csv", metadata_type="train")
    train_meta_df = filter_metadata_by_client_config(config=config, meta_df=train_meta_df, metadata_type="train")
    
    # Identify clients
    client_col = _detect_client_column(test_meta_df)
    
    datasets_by_client = {}
    
    if client_col is None:
        # Single client case
        client_id = test_meta_df["dataset"].iloc[0] if "dataset" in test_meta_df.columns else "unknown"
        dataset = task_spec.dataset_factory(
            config=cfg,
            data_dir=data_dir / "test",
            meta_df=test_meta_df,
            transform=transform,
            train_meta_df=train_meta_df,
        )
        datasets_by_client[client_id] = (len(test_meta_df), dataset)
    else:
        # Multi-client case
        for client_id in sorted(test_meta_df[client_col].unique()):
            client_meta_df = test_meta_df[test_meta_df[client_col] == client_id]
            dataset = task_spec.dataset_factory(
                config=cfg,
                data_dir=data_dir / "test",
                meta_df=client_meta_df,
                transform=transform,
                train_meta_df=train_meta_df,
            )
            datasets_by_client[client_id] = (len(client_meta_df), dataset)
    
    logger.info(f"Created test datasets ({task}) for {len(datasets_by_client)} clients: {list(datasets_by_client.keys())}")
    return datasets_by_client


def compute_metrics_on_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    reg_mean: float,
    reg_std: float,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    """Compute regression metrics on a dataloader.
    
    Args:
        model: Regression model.
        dataloader: DataLoader to evaluate.
        reg_mean: Normalization mean for denormalization.
        reg_std: Normalization std for denormalization.
        config: Configuration dictionary.
        device: Device to use.
    
    Returns:
        Dictionary with normalized and restored metrics.
    """
    if get_amp_dtype_and_device is None or regression_eval is None or regression_metrics is None:
        raise RuntimeError("Regression evaluation helpers are unavailable in this environment.")

    from omegaconf import OmegaConf
    cfg = OmegaConf.create(config)
    amp_dtype, _ = get_amp_dtype_and_device()
    
    # Build patch size dict - handle missing views gracefully
    views = cfg.model.views if isinstance(cfg.model.views, list) else [cfg.model.views]
    patch_size_dict = {}
    spacing_dict = {}
    
    for v in views:
        if v == "sax" and "sax" in cfg.data:
            patch_size_dict[v] = cfg.data.sax.patch_size
            spacing_dict[v] = cfg.data.sax.spacing
        elif v == "lax" and "lax" in cfg.data:
            patch_size_dict[v] = cfg.data.lax.patch_size
            spacing_dict[v] = cfg.data.lax.spacing
        else:
            # Default to sax if view config missing
            logger.warning(f"View {v} not found in config, using default sax patch size")
            patch_size_dict[v] = cfg.data.sax.patch_size if "sax" in cfg.data else [192, 192, 16]
            spacing_dict[v] = cfg.data.sax.spacing if "sax" in cfg.data else [1.0, 1.0, 10.0]
    
    pred_labels = []
    true_labels = []
    
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
    
    pred_labels = torch.cat(pred_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels = torch.cat(true_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    
    # Compute metrics on normalized scale
    metrics = regression_metrics(true_labels=true_labels, pred_labels=pred_labels)

    # R² (scale-invariant — computed here instead of regression_metrics to keep cinema untouched)
    ss_res = float(np.sum((true_labels - pred_labels) ** 2))
    ss_tot = float(np.sum((true_labels - np.mean(true_labels)) ** 2))
    metrics["r2"] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Denormalize and compute metrics on original scale
    restored_pred_labels = (pred_labels * reg_std) + reg_mean
    restored_true_labels = (true_labels * reg_std) + reg_mean
    restored_metrics = regression_metrics(
        true_labels=restored_true_labels,
        pred_labels=restored_pred_labels,
    )
    restored_metrics = {f"restored_{k}": v for k, v in restored_metrics.items() if k != "r2"}
    metrics.update(restored_metrics)
    
    return metrics


def to_builtin(obj: Any) -> Any:
    """Recursively convert numpy/torch values to JSON-serializable builtins."""
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return to_builtin(obj.detach().cpu().tolist())
    return obj


def compute_primary_metric_fairness(
    client_results: list[dict[str, Any]],
    primary_metric: str,
) -> dict[str, Any]:
    """Compute fairness metrics using one configured primary metric only.

    Raises:
        ValueError: If a client misses the metric, metric is non-numeric, or no clients provided.
    """
    if not client_results:
        raise ValueError("Cannot compute fairness: client_results is empty.")

    per_client_primary: dict[str, float] = {}
    for result in client_results:
        client_id = str(result.get("client_id"))
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Client '{client_id}' has invalid metrics payload: {metrics}")
        if primary_metric not in metrics:
            raise ValueError(
                f"Client '{client_id}' is missing required primary metric '{primary_metric}' for fairness."
            )
        metric_value = metrics[primary_metric]
        if not isinstance(metric_value, (int, float)):
            raise ValueError(
                f"Client '{client_id}' has non-numeric primary metric '{primary_metric}': {metric_value}"
            )
        per_client_primary[client_id] = float(metric_value)

    values = list(per_client_primary.values())
    clients = list(per_client_primary.keys())
    min_idx = int(np.argmin(values))
    max_idx = int(np.argmax(values))

    return {
        "primary_metric": primary_metric,
        "std": float(np.std(values)),
        "gap": float(max(values) - min(values)),
        "min_client": clients[min_idx],
        "min_value": values[min_idx],
        "max_client": clients[max_idx],
        "max_value": values[max_idx],
        "per_client_primary": per_client_primary,
    }


def normalize_metric_values(metrics: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            normalized[str(name)] = float(value)
    return normalized


def classification_eval_dataloader_with_balanced_accuracy(
    model,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate classification model and return metrics with balanced accuracy + predictions."""
    import torch.nn.functional as F
    from sklearn.metrics import balanced_accuracy_score

    from models.cinema.cinema.classification.train import classification_eval, classification_metrics

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

    balanced_acc = balanced_accuracy_score(y_true=true_labels_np, y_pred=pred_labels_np)
    metrics["balanced_accuracy"] = float(balanced_acc)

    return metrics, true_labels_np, pred_labels_np


def _evaluate_classification(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    from omegaconf import OmegaConf

    if get_amp_dtype_and_device is None:
        raise RuntimeError("Classification evaluation dependencies are unavailable in this environment.")

    cfg = OmegaConf.create(config)
    amp_dtype, _ = get_amp_dtype_and_device()
    views = cfg.model.views if isinstance(cfg.model.views, list) else [cfg.model.views]
    patch_size_dict = {
        v: cfg.data.sax.patch_size if v == "sax" else cfg.data.lax.patch_size
        for v in views
    }
    metrics, _, _ = classification_eval_dataloader_with_balanced_accuracy(
        model=model,
        dataloader=dataloader,
        patch_size_dict=patch_size_dict,
        amp_dtype=amp_dtype,
        device=device,
    )
    return normalize_metric_values(metrics)


def _evaluate_regression(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    from omegaconf import OmegaConf

    if get_amp_dtype_and_device is None:
        raise RuntimeError("Regression evaluation dependencies are unavailable in this environment.")

    _ = resolve_federated_eval_task(config)
    cfg = OmegaConf.create(config)
    task_spec = get_task_spec(cfg)
    eval_dataloader_fn = task_spec.eval_fn_factory()
    amp_dtype, _ = get_amp_dtype_and_device()
    views = cfg.model.views if isinstance(cfg.model.views, list) else [cfg.model.views]
    patch_size_dict = {
        v: cfg.data.sax.patch_size if v == "sax" else cfg.data.lax.patch_size
        for v in views
    }
    spacing_dict = {
        v: cfg.data.sax.spacing if v == "sax" else cfg.data.lax.spacing
        for v in views
    }

    # Use eval_utils compute_metrics_on_dataloader for R² support (cinema's
    # regression_eval_dataloader is equivalent but doesn't include R²).
    reg_mean = getattr(dataloader.dataset, "reg_mean", 0.0)
    reg_std = getattr(dataloader.dataset, "reg_std", 1.0)
    metrics = compute_metrics_on_dataloader(
        model=model,
        dataloader=dataloader,
        reg_mean=reg_mean,
        reg_std=reg_std,
        config=config,
        device=device,
    )
    return normalize_metric_values(metrics)


def _evaluate_segmentation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    """Wrap segmentation_eval_dataloader with the same interface as other _evaluate_* functions."""
    from omegaconf import OmegaConf

    from models.cinema.cinema.device import get_amp_dtype_and_device as _get_amp_device
    from models.cinema.cinema.segmentation.train import segmentation_eval_dataloader

    cfg = OmegaConf.create(config)
    amp_dtype, _ = _get_amp_device()
    views = cfg.model.views if isinstance(cfg.model.views, list) else [cfg.model.views]
    patch_size_dict = {
        v: cfg.data.sax.patch_size if v == "sax" else cfg.data.lax.patch_size
        for v in views
    }
    spacing_dict = {
        v: cfg.data.sax.spacing if v == "sax" else cfg.data.lax.spacing
        for v in views
    }

    metrics = segmentation_eval_dataloader(
        model=model,
        dataloader=dataloader,
        patch_size_dict=patch_size_dict,
        spacing_dict=spacing_dict,
        amp_dtype=amp_dtype,
        device=device,
    )
    return normalize_metric_values(metrics)


def get_federated_eval_fn(task: str | dict[str, Any] | Any) -> Callable[..., dict[str, float]]:
    """Return federated evaluation function for supported tasks."""
    task_name = resolve_federated_eval_task(task)
    if task_name == "classification":
        return _evaluate_classification
    if task_name == "segmentation":
        return _evaluate_segmentation
    return _evaluate_regression


# ============================================================================
# PR #2 — Monte Carlo Dropout Uncertainty Estimation
# ============================================================================

def enable_mc_dropout(model: torch.nn.Module) -> list[str]:
    """Set only nn.Dropout layers to train mode; leave all others in eval.

    Does NOT touch DropPath, BatchNorm, LayerNorm, or any timm stochastic layers.

    Args:
        model: PyTorch model already in eval mode.

    Returns:
        List of activated dropout module names (for logging).
    """
    activated: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            activated.append(name)
    return activated


def mc_dropout_predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    task_type: str,
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]] | None,
    amp_dtype: torch.dtype,
    device: torch.device,
    n_passes: int = 30,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray | None]:
    """Run N stochastic forward passes with dropout enabled.

    Args:
        model: Model with dropout layers already set to train mode.
        dataloader: Evaluation dataloader (batch_size=1 for patching support).
        task_type: One of "classification", "regression", "segmentation".
        patch_size_dict: Patch size per view.
        spacing_dict: Spacing per view (used for segmentation).
        amp_dtype: dtype for mixed precision.
        device: Device to run on.
        n_passes: Number of stochastic forward passes.

    Returns:
        Tuple of (mean, std, all_outputs, all_labels) where:
        - classification: mean/std are (n_samples, n_classes), all_outputs is list of (n_samples, n_classes) probs, labels is (n_samples,)
        - regression: mean/std are (n_samples,), all_outputs is list of (n_samples,) preds, labels is (n_samples,)
        - segmentation: mean/std are empty, all_outputs is list of dicts, labels is None (labels inside dicts)
    """
    import torch.nn.functional as F

    from models.cinema.cinema.classification.train import classification_eval
    from models.cinema.cinema.regression.train import regression_eval
    from models.cinema.cinema.segmentation.train import segmentation_eval

    all_outputs: list = []
    all_labels_list: list[np.ndarray] | None = None if task_type == "segmentation" else []

    for pass_idx in range(n_passes):
        pass_results: list = []
        if all_labels_list is not None and pass_idx == 0:
            all_labels_list = []
        for batch in dataloader:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                if task_type == "classification":
                    logits, _ = classification_eval(
                        model=model, batch=batch, patch_size_dict=patch_size_dict,
                        amp_dtype=amp_dtype, device=device,
                    )
                    probs = F.softmax(logits, dim=1).cpu().to(dtype=torch.float32).numpy()
                    pass_results.append(probs)
                    if pass_idx == 0:
                        all_labels_list.append(batch["label"].cpu().numpy())
                elif task_type == "regression":
                    preds, _ = regression_eval(
                        model=model, batch=batch, patch_size_dict=patch_size_dict,
                        amp_dtype=amp_dtype, device=device,
                    )
                    pass_results.append(preds.cpu().to(dtype=torch.float32).numpy())
                    if pass_idx == 0:
                        all_labels_list.append(batch["label"].cpu().numpy())
                elif task_type == "segmentation":
                    if spacing_dict is None:
                        raise ValueError("spacing_dict is required for segmentation MC Dropout.")
                    # metrics_fn=None: skip metrics during passes (computed later in clinical_metrics)
                    logits_dict, _ = segmentation_eval(
                        model=model, batch=batch,
                        patch_size_dict=patch_size_dict, spacing_dict=spacing_dict,
                        amp_dtype=amp_dtype, device=device,
                        metrics_fn=None,
                    )
                    # Crop labels to match cropped logits (same logic as segmentation_eval)
                    from models.cinema.cinema.transform import crop_start
                    cpu_logits = {}
                    labels_cpu = {}
                    for v in logits_dict:
                        width = int(batch[f"{v}_width"][0])
                        height = int(batch[f"{v}_height"][0])
                        label = batch[f"{v}_label"]
                        if len(patch_size_dict[v]) == 3:
                            n_slices = int(batch["n_slices"][0])
                            label = crop_start(label, (*label.shape[:2], width, height, n_slices))
                        elif len(patch_size_dict[v]) == 2:
                            label = crop_start(label, (*label.shape[:2], width, height))
                        cpu_logits[v] = logits_dict[v].cpu()
                        labels_cpu[v] = label.cpu()
                    pass_results.append({"logits": cpu_logits, "labels": labels_cpu})
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")

        # Offload pass results to CPU / numpy to free GPU memory
        if task_type == "classification":
            pass_array = np.concatenate(pass_results, axis=0)
            all_outputs.append(pass_array)
        elif task_type == "regression":
            pass_array = np.concatenate(pass_results, axis=0)
            all_outputs.append(pass_array)
        elif task_type == "segmentation":
            all_outputs.append(pass_results)

    # Stack and compute mean/std
    if task_type == "classification":
        stacked = np.stack(all_outputs, axis=0)  # (n_passes, n_samples, n_classes)
        labels = np.concatenate(all_labels_list, axis=0) if all_labels_list else None
        return stacked.mean(axis=0), stacked.std(axis=0), all_outputs, labels
    elif task_type == "regression":
        stacked = np.stack(all_outputs, axis=0)  # (n_passes, n_samples, 1)
        stacked = stacked.squeeze(-1)  # (n_passes, n_samples)
        labels = np.concatenate(all_labels_list, axis=0) if all_labels_list else None
        return stacked.mean(axis=0), stacked.std(axis=0), all_outputs, labels
    elif task_type == "segmentation":
        return np.array([]), np.array([]), all_outputs, None

    raise ValueError(f"Unknown task_type: {task_type}")


def compute_classification_uncertainty(
    all_probs: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Compute uncertainty metrics from MC Dropout classification outputs.

    Args:
        all_probs: Stacked softmax probabilities of shape (n_passes, n_samples, n_classes).

    Returns:
        Dict with:
        - mean_entropy: mean predictive entropy per sample
        - mutual_info: mutual information (epistemic uncertainty) per sample
        - per_class_std: std of predictions per class across passes
    """
    # Predictive entropy: H = -sum(p_mean * log(p_mean)) per sample
    mean_probs = all_probs.mean(axis=0)  # (n_samples, n_classes)
    mean_probs_clipped = np.clip(mean_probs, 1e-10, 1.0)
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs_clipped), axis=1)  # (n_samples,)
    mean_entropy = float(np.mean(predictive_entropy))
    # Normalize by ln(n_classes) → range [0, 1] (Jungo & Reyes 2019)
    n_classes = all_probs.shape[2]
    if n_classes > 1:
        mean_entropy = mean_entropy / np.log(n_classes)

    # Expected entropy: mean of per-pass entropies
    probs_clipped = np.clip(all_probs, 1e-10, 1.0)
    per_pass_entropy = -np.sum(all_probs * np.log(probs_clipped), axis=2)  # (n_passes, n_samples)
    expected_entropy = per_pass_entropy.mean(axis=0)  # (n_samples,)

    # Mutual information = predictive_entropy - expected_entropy
    mutual_info = predictive_entropy - expected_entropy
    mean_mutual_info = float(np.mean(mutual_info))

    # Per-class standard deviation across passes
    per_class_std = all_probs.std(axis=0).mean(axis=0)  # (n_classes,)

    return {
        "mean_entropy": mean_entropy,
        "mutual_info": mean_mutual_info,
        "per_class_std": per_class_std,
    }


def compute_regression_uncertainty(
    all_preds: np.ndarray,
    reg_mean: float,
    reg_std: float,
) -> dict[str, float | np.ndarray]:
    """Compute uncertainty metrics from MC Dropout regression outputs.

    Args:
        all_preds: Stacked predictions of shape (n_passes, n_samples).
        reg_mean: Normalization mean for restoring scales.
        reg_std: Normalization std for restoring scales.

    Returns:
        Dict with:
        - std: per-sample std (normalized scale)
        - ci_lower: lower bound of 95% CI (normalized)
        - ci_upper: upper bound of 95% CI (normalized)
        - restored_std: per-sample std (original scale)
        - restored_ci_lower: lower bound of 95% CI (original scale)
        - restored_ci_upper: upper bound of 95% CI (original scale)
    """
    sample_std = all_preds.std(axis=0)  # (n_samples,)
    sample_mean = all_preds.mean(axis=0)  # (n_samples,)

    ci_lower = sample_mean - 1.96 * sample_std
    ci_upper = sample_mean + 1.96 * sample_std

    restored_std = sample_std * reg_std
    restored_ci_lower = ci_lower * reg_std + reg_mean
    restored_ci_upper = ci_upper * reg_std + reg_mean

    return {
        "std": sample_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "restored_std": restored_std,
        "restored_ci_lower": restored_ci_lower,
        "restored_ci_upper": restored_ci_upper,
    }


def compute_regression_clinical_metrics_uncertainty(
    all_preds: np.ndarray,
    labels: np.ndarray,
    reg_mean: float,
    reg_std: float,
) -> dict[str, float]:
    """Compute flat-scalar clinical metrics for regression MC Dropout.

    Computes metrics on the predictive mean mu = mean(all_preds, axis=0),
    NOT per-pass averages. Returns flat floats only — no descriptive fields.

    Args:
        all_preds: Stacked predictions of shape (n_passes, n_samples).
        labels: Ground truth labels, shape (n_samples,).
        reg_mean: Normalization mean for restoring scales.
        reg_std: Normalization std for restoring scales.

    Returns:
        Dict with flat float scalars:
        rmse, restored_rmse, mae, restored_mae, r2,
        picp, mpiw, restored_mpiw, nll.
    """
    # Squeeze labels to (n_samples,) to prevent NumPy broadcasting bug:
    # labels from DataLoader are (n_samples, 1), mu is (n_samples,).
    # (n_samples, 1) - (n_samples,) broadcasts to (n_samples, n_samples).
    labels = labels.squeeze()

    mu = all_preds.mean(axis=0)  # predictive mean (n_samples,)
    sample_std = all_preds.std(axis=0)

    # RMSE / MAE on predictive mean
    abs_error = np.abs(labels - mu)
    rmse = float(np.sqrt(np.mean(abs_error ** 2)))
    mae = float(np.mean(abs_error))
    restored_rmse = rmse * reg_std
    restored_mae = mae * reg_std

    # R² on predictive mean
    ss_res = float(np.sum((labels - mu) ** 2))
    ss_tot = float(np.sum((labels - np.mean(labels)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # PICP
    ci_lower = mu - 1.96 * sample_std
    ci_upper = mu + 1.96 * sample_std
    within = (labels >= ci_lower) & (labels <= ci_upper)
    picp = float(np.mean(within))

    # MPIW
    mpiw = 3.92 * float(np.mean(sample_std))
    restored_mpiw = mpiw * reg_std

    # NLL (normalized scale only, no restored variant)
    eps = 1e-3
    sigma_clamped = np.maximum(sample_std, eps)
    nll = float(np.mean(
        0.5 * (np.log(2 * np.pi * sigma_clamped ** 2) + (labels - mu) ** 2 / sigma_clamped ** 2)
    ))

    return {
        "rmse": rmse,
        "restored_rmse": restored_rmse,
        "mae": mae,
        "restored_mae": restored_mae,
        "r2": float(r2),
        "picp": picp,
        "mpiw": mpiw,
        "restored_mpiw": restored_mpiw,
        "nll": nll,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "mean_y": float(np.mean(labels)),
    }


def compute_classification_clinical_metrics_uncertainty(
    all_outputs: list[np.ndarray],
    all_labels: np.ndarray,
) -> dict[str, dict[str, float] | dict[str, dict[str, float]]]:
    """Compute classification clinical metrics with consensus and per-pass sections.

    Consensus metrics are computed on argmax(mean_t(probs_t)) — the Bayesian
    model's averaged prediction.  Per-pass metrics capture variability across
    dropout realisations using {mean, std} envelopes.

    Args:
        all_outputs: List of per-pass probability arrays, each (n_samples, n_classes).
        all_labels: Ground truth labels, shape (n_samples,) or (n_samples, 1).

    Returns:
        Dict with two top-level keys:
        - consensus: flat float scalars (accuracy, balanced_accuracy, f1, mcc,
          roc_auc, cross_entropy, [specificity, sensitivity,] n_samples).
        - per_pass: {metric: {"mean": float, "std": float}} for each
          metric (same keys minus n_samples).
    """
    from models.cinema.cinema.classification.train import classification_metrics
    from sklearn.metrics import balanced_accuracy_score

    # Defensive squeeze — DataLoader yields (n_samples, 1).
    labels = all_labels.squeeze()

    stacked = np.stack(all_outputs, axis=0)  # (n_passes, n_samples, n_classes)
    n_passes = stacked.shape[0]

    # ── Consensus ───────────────────────────────────────────────────────
    consensus_probs = stacked.mean(axis=0)  # (n_samples, n_classes)
    consensus_labels = np.argmax(consensus_probs, axis=1)

    consensus_metrics = classification_metrics(
        true_labels=labels,
        pred_labels=consensus_labels,
        pred_probs=consensus_probs,
    )
    consensus_metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y_true=labels, y_pred=consensus_labels)
    )

    # Cross-entropy on consensus probs: -mean(sum(onehot * log(probs)))
    eps = 1e-6
    n_classes = consensus_probs.shape[1]
    onehot = np.eye(n_classes)[labels.astype(int)]
    cross_entropy = float(-np.mean(np.sum(onehot * np.log(consensus_probs + eps), axis=1)))

    # Discard the per-pass 'entropy' from classification_metrics; use our cross_entropy.
    consensus_metrics.pop("entropy", None)
    consensus_metrics["cross_entropy"] = cross_entropy
    consensus_metrics["n_samples"] = float(len(labels))

    consensus_flat = {k: float(v) for k, v in consensus_metrics.items()}

    # ── Per-pass ────────────────────────────────────────────────────────
    metrics_per_pass: list[dict[str, float]] = []
    for t in range(n_passes):
        pred_labels_t = np.argmax(stacked[t], axis=1)
        metrics_t = classification_metrics(
            true_labels=labels,
            pred_labels=pred_labels_t,
            pred_probs=stacked[t],
        )
        metrics_t["balanced_accuracy"] = float(
            balanced_accuracy_score(y_true=labels, y_pred=pred_labels_t)
        )
        # Rename entropy → cross_entropy per pass.
        metrics_t.pop("entropy", None)
        metrics_t["cross_entropy"] = float(
            -np.mean(np.sum(onehot * np.log(stacked[t] + eps), axis=1))
        )
        metrics_per_pass.append({k: float(v) for k, v in metrics_t.items()})

    # Aggregate: NaN-skipping mean/std per metric.
    per_pass_agg: dict[str, dict[str, float]] = {}
    if metrics_per_pass:
        for metric_name in metrics_per_pass[0]:
            if metric_name == "n_samples":
                continue
            values = np.array([m[metric_name] for m in metrics_per_pass], dtype=float)
            valid = values[~np.isnan(values)]
            if len(valid) == 0:
                per_pass_agg[metric_name] = {"mean": float("nan"), "std": float("nan")}
            else:
                per_pass_agg[metric_name] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                }

    return {"consensus": consensus_flat, "per_pass": per_pass_agg}


def compute_segmentation_uncertainty(
    all_outputs: list[list[dict[str, dict[str, torch.Tensor]]]],
    output_dir: Path | None = None,
    spacing_dict: dict[str, tuple[float, ...]] | None = None,
) -> dict[str, float | np.ndarray | str]:
    """Compute uncertainty metrics from MC Dropout segmentation outputs.

    Args:
        all_outputs: Per-pass lists of dicts [{logits, labels}_p0_s0, ...].
        output_dir: Directory (unused, kept for API compatibility).
        spacing_dict: Per-view voxel spacing for accurate Hausdorff metrics.

    Returns:
        Dict with:
        - mean_dice_std: mean std of Dice scores across passes
        - mean_predictive_variance: mean predictive variance across voxels
    """
    # Collect per-pass Dice scores. all_outputs is list of per-pass lists of dicts.
    dice_per_pass: list[dict[str, float]] = []       # one entry per pass
    all_logits_per_pass: list[list[torch.Tensor]] = []  # per-pass list of logit tensors

    for pass_outputs in all_outputs:  # each is list of dicts for one pass
        pass_logits: list[torch.Tensor] = []
        pass_dice: dict[str, list[float]] = {}
        for output in pass_outputs:
            logits_dict = output["logits"]
            labels_dict = output["labels"]
            first_view = next(iter(logits_dict))
            logits = logits_dict[first_view]
            labels = labels_dict[first_view]
            pass_logits.append(logits)

            sp = spacing_dict.get(first_view, (1.0, 1.0, 1.0)) if spacing_dict else (1.0, 1.0, 1.0)
            pass_metrics = _segmentation_clinical_metrics(logits, labels, spacing=sp)
            for k, v in pass_metrics.items():
                if "dice" in k:
                    pass_dice.setdefault(k, []).append(v)

        all_logits_per_pass.append(pass_logits)
        # Average dice scores across samples for this pass
        dice_per_pass.append({k: float(np.mean(v)) for k, v in pass_dice.items()})

    # Mean Dice std across classes + per-class breakdown
    dice_keys = sorted(dice_per_pass[0].keys())
    dice_array = np.array([[d.get(k, 0.0) for k in dice_keys] for d in dice_per_pass])
    dice_std_per_class = dice_array.std(axis=0)  # (n_dice_metrics,)
    mean_dice_std = float(np.mean(dice_std_per_class))
    per_class_dice_std = {k: float(v) for k, v in zip(dice_keys, dice_std_per_class)}

    # Mean predictive variance — per-sample, per-class + global
    n_passes_total = len(dice_per_pass)
    n_samples = len(all_logits_per_pass[0]) if all_logits_per_pass else 0
    per_sample_variance: list[np.ndarray] = []

    if n_samples > 0:
        for sample_idx in range(n_samples):
            sample_logits = [all_logits_per_pass[p][sample_idx] for p in range(n_passes_total)
                           if sample_idx < len(all_logits_per_pass[p])]
            if sample_logits:
                shapes = {tuple(s.shape) for s in sample_logits}
                if len(shapes) == 1:
                    stacked = torch.stack(sample_logits, dim=0)  # (n_passes, 1, n_classes, H, W, D)
                    softmax_passes = torch.softmax(stacked.float(), dim=2)
                    var_per_class = softmax_passes.var(dim=0, unbiased=False).mean(dim=(-1, -2, -3)).squeeze(0)  # (n_classes,)
                    per_sample_variance.append(var_per_class.float().cpu().numpy())
                else:
                    # Variable-sized volumes between passes — cannot compute inter-pass variance
                    per_sample_variance.append(np.full(sample_logits[0].shape[1], np.nan))

    if per_sample_variance:
        stacked_vars = np.stack(per_sample_variance, axis=0)  # (n_samples, n_classes)
        per_class_predictive_variance = stacked_vars.mean(axis=0)  # (n_classes,)
        mean_predictive_variance = float(np.mean(per_class_predictive_variance))
    else:
        per_class_predictive_variance = np.array([])
        mean_predictive_variance = 0.0

    # Mean predictive entropy (per-voxel, averaged) — Jungo & Reyes 2019, eq. (4)
    per_sample_entropy: list[float] = []
    per_sample_entropy_fg: list[float] = []  # foreground only (predicted labels > 0)
    if n_samples > 0:
        for sample_idx in range(n_samples):
            sample_logits = [all_logits_per_pass[p][sample_idx] for p in range(n_passes_total)
                           if sample_idx < len(all_logits_per_pass[p])]
            if sample_logits and len({tuple(s.shape) for s in sample_logits}) == 1:
                stacked = torch.stack(sample_logits, dim=0)
                mean_softmax = torch.softmax(stacked.float(), dim=2).mean(dim=0)
                mean_softmax_clipped = torch.clamp(mean_softmax, min=1e-10)
                entropy = -torch.sum(mean_softmax * torch.log(mean_softmax_clipped), dim=1)
                per_sample_entropy.append(float(entropy.mean().item()))
                # Foreground uncertainty should be prediction-derived, not label-derived.
                predicted_foreground = torch.argmax(mean_softmax, dim=1).squeeze(0) > 0
                fg_entropy = entropy.squeeze(0)[predicted_foreground]
                if fg_entropy.numel() > 0:
                    per_sample_entropy_fg.append(float(fg_entropy.mean().item()))
    mean_predictive_entropy = float(np.mean(per_sample_entropy)) if per_sample_entropy else 0.0
    mean_predictive_entropy_fg = float(np.mean(per_sample_entropy_fg)) if per_sample_entropy_fg else 0.0
    # Normalize by ln(n_classes) → range [0, 1]
    if all_logits_per_pass and all_logits_per_pass[0]:
        n_classes = all_logits_per_pass[0][0].shape[1]
        if n_classes > 1:
            mean_predictive_entropy = mean_predictive_entropy / np.log(n_classes)
            mean_predictive_entropy_fg = mean_predictive_entropy_fg / np.log(n_classes)

    return {
        "mean_dice_std": mean_dice_std,
        "per_class_dice_std": per_class_dice_std,
        "mean_predictive_variance": mean_predictive_variance,
        "per_class_predictive_variance": _label_predictive_variance(per_class_predictive_variance),
        "mean_predictive_entropy": mean_predictive_entropy,
        "mean_predictive_entropy_foreground": mean_predictive_entropy_fg,
    }


def _label_predictive_variance(values: np.ndarray) -> dict[str, float]:
    """Label per-class predictive variance with channel names.

    Channel 0 = background, then class_1, class_2, ...
    """
    n = len(values)
    if n == 0:
        return {}
    labels = ["background"] + [f"class_{i}" for i in range(1, n)]
    return {labels[i]: float(values[i]) for i in range(min(n, len(labels)))}


# ============================================================================
# Task 1.3 — Shared argument parser
# ============================================================================

def build_shared_argument_parser() -> argparse.ArgumentParser:
    """Build a shared argparse.ArgumentParser with unified evaluation CLI flags.

    Returns a parser pre-configured with:
    - --checkpoint-dir (type=Path, default=None)
    - --checkpoint-path (type=Path, default=None)
    - --output-dir (type=Path, default=None)
    - --output-file-name (type=str, default="evaluation_results.json")
    - --mc-dropout (action="store_true", default=False)
    - --mc-dropout-passes (type=int, default=30)

    Note: --checkpoint-epoch is centralized-only and added by the centralized wrapper.
    Validation: at least one of --checkpoint-dir or --checkpoint-path is required.
    """
    parser = argparse.ArgumentParser(
        description="Unified evaluation for centralized and federated models."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Direct checkpoint path. If set, directory-based selection is skipped.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (auto-generated if not specified)",
    )
    parser.add_argument(
        "--output-file-name",
        type=str,
        default="evaluation_results.json",
        help="Output filename for results JSON",
    )
    parser.add_argument(
        "--mc-dropout",
        action="store_true",
        default=False,
        help="Enable Monte Carlo Dropout uncertainty estimation",
    )
    parser.add_argument(
        "--mc-dropout-passes",
        type=int,
        default=30,
        help="Number of MC Dropout forward passes",
    )

    # Override parse_args to enforce checkpoint source validation
    _original_parse_args = parser.parse_args

    def _validated_parse_args(args=None, namespace=None):
        parsed = _original_parse_args(args, namespace)
        if parsed.checkpoint_dir is None and parsed.checkpoint_path is None:
            parser.error("At least one of --checkpoint-dir or --checkpoint-path is required.")
        return parsed

    parser.parse_args = _validated_parse_args  # type: ignore[assignment]
    return parser


# ============================================================================
# Task 1.7 — Unified result payload builder (schema)
# ============================================================================

def _compute_fairness_from_per_client(
    per_client_metrics: dict[str, tuple[int, dict[str, float]]],
    primary_metric: str,
) -> dict[str, Any]:
    """Compute fairness analysis metrics across clients.

    Args:
        per_client_metrics: Dict mapping client_id to (n_samples, metrics_dict).
        primary_metric: The metric key to use for fairness analysis.

    Returns:
        Dictionary with fairness statistics: std, gap, min/max client, per-client values.
    """
    per_client_primary: dict[str, float] = {}
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


def build_unified_result_payload(
    config: dict[str, Any],
    per_client_metrics: dict[str, tuple[int, dict[str, float]]],
    primary_metric: str,
    is_federated: bool = False,
    checkpoint_selection_mode: str | None = None,
    checkpoint_round: int | None = None,
) -> dict[str, Any]:
    """Build unified result payload with new schema.

    New schema:
    {
        "config": {...},
        "overall": {
            "weighted_avg": {...},
            "macro_avg": {...},
        },
        "per_client": {...},
        "fairness": {...},
    }

    Federated mode adds: checkpoint_selection_mode and checkpoint_round to config.
    """
    # Compute weighted and macro averages
    weighted_metrics: dict[str, float] = {}
    macro_metrics: dict[str, float] = {}
    total_samples = sum(n for n, _ in per_client_metrics.values())

    if per_client_metrics:
        first_client_metrics = list(per_client_metrics.values())[0][1]
        n_clients = len(per_client_metrics)

        for metric_key in first_client_metrics.keys():
            # Weighted average (sample-weighted)
            weighted_sum = sum(
                n_samples * per_client_metrics[client_id][1][metric_key]
                for client_id, (n_samples, _) in per_client_metrics.items()
            )
            weighted_metrics[metric_key] = weighted_sum / total_samples if total_samples > 0 else 0.0

            # Macro average (unweighted)
            macro_sum = sum(
                per_client_metrics[client_id][1][metric_key]
                for client_id in per_client_metrics
            )
            macro_metrics[metric_key] = macro_sum / n_clients

    # Build per_client output
    per_client_output: dict[str, dict[str, Any]] = {}
    for client_id, (n_samples, metrics) in per_client_metrics.items():
        entry: dict[str, Any] = {
            "n_samples": n_samples,
            "metrics": to_builtin(metrics),
        }
        per_client_output[client_id] = entry

    # Build config
    output_config: dict[str, Any] = to_builtin(config)
    if is_federated:
        if checkpoint_selection_mode is not None:
            output_config["checkpoint_selection_mode"] = checkpoint_selection_mode
        if checkpoint_round is not None:
            output_config["checkpoint_round"] = checkpoint_round

    # Build fairness
    fairness = _compute_fairness_from_per_client(per_client_metrics, primary_metric)

    payload = {
        "config": output_config,
        "overall": {
            "weighted_avg": to_builtin(weighted_metrics),
            "macro_avg": to_builtin(macro_metrics),
        },
        "per_client": per_client_output,
        "fairness": to_builtin(fairness),
    }
    return payload


# ============================================================================
# Task 1.4 — Unified evaluation orchestrator
# ============================================================================

def run_unified_evaluation(
    checkpoint_dir: Path | None,
    checkpoint_path: Path | None,
    checkpoint_epoch: int | None,
    output_dir: Path | None,
    output_file_name: str,
    mc_dropout: bool,
    mc_dropout_passes: int,
    is_federated: bool,
) -> Path:
    """Unified evaluation orchestrator for centralized and federated modes.

    Handles:
    a. Checkpoint resolution
    b. Model building via task adapters
    c. Weight loading
    d. Per-client evaluation
    e. Weighted average + macro average (overall)
    f. Fairness metrics
    g. JSON output with new schema

    MC Dropout: when mc_dropout=True, runs stochastic forward passes
    and appends uncertainty metrics under mc_dropout_uncertainty key.
    """
    from flwr_cinema.scripts.lora import load_lora_state_dict
    from models.cinema.cinema.device import get_amp_dtype_and_device
    from models.cinema.cinema.log import get_logger
    from omegaconf import OmegaConf

    orc_logger = get_logger(__name__)

    if mc_dropout:
        orc_logger.info(
            f"MC Dropout enabled with {mc_dropout_passes} passes. "
            "Uncertainty metrics will be included in output."
        )

    # ── a. Checkpoint resolution ──────────────────────────────────────────
    if is_federated:
        ckpt_path, selection_meta = _resolve_federated_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
        )
    else:
        ckpt_path, selection_meta = _resolve_centralized_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=checkpoint_epoch,
        )

    # Load checkpoint and config
    checkpoint, config = load_checkpoint_and_config(ckpt_path, output_dir=ckpt_path.parent)
    cfg = OmegaConf.create(config)
    task = str(getattr(cfg, "task", "classification"))
    task = resolve_federated_eval_task(task)

    # ── b. Model building ─────────────────────────────────────────────────
    if is_federated:
        from flwr_cinema.tasks.adapters import get_federated_task_adapter
        adapter = get_federated_task_adapter(cfg)
    else:
        from flwr_cinema.tasks.adapters import get_centralized_task_adapter
        adapter = get_centralized_task_adapter(cfg)

    model = adapter.build_model(cfg)

    # ── c. Weight loading ─────────────────────────────────────────────────
    use_lora = bool(getattr(cfg.train, "use_lora", False)) if hasattr(cfg, "train") else False
    checkpoint_model_state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=not use_lora)
        if use_lora and "lora_state_dict" in checkpoint:
            load_lora_state_dict(model, checkpoint["lora_state_dict"])
            orc_logger.info("Loaded LoRA weights from checkpoint")
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=not use_lora)
    else:
        model.load_state_dict(checkpoint_model_state, strict=not use_lora)

    amp_dtype, device = get_amp_dtype_and_device()
    model.to(device)
    model.eval()
    orc_logger.info(f"Loaded model on device {device} (use_lora={use_lora})")

    # ── d. Per-client evaluation ──────────────────────────────────────────
    if is_federated:
        per_client_metrics = _run_federated_per_client(
            model=model,
            checkpoint=checkpoint,
            config=config,
            task_type=task,
            device=device,
            base_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        )
    else:
        per_client_metrics = _run_centralized_per_client(
            model=model,
            config=config,
            task_type=task,
            device=device,
            amp_dtype=amp_dtype,
        )

    # ── e2. MC Dropout uncertainty (PR #2) ────────────────────────────────
    mc_dropout_results: dict[str, Any] = {}
    if mc_dropout:
        # Enable dropout layers for stochastic passes
        activated = enable_mc_dropout(model)
        orc_logger.info(f"MC Dropout activated {len(activated)} dropout layers")

        if is_federated:
            mc_dropout_results = _run_mc_dropout_federated(
                model=model,
                checkpoint=checkpoint,
                config=config,
                task_type=task,
                device=device,
                amp_dtype=amp_dtype,
                n_passes=mc_dropout_passes,
                base_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                output_dir=output_dir,
            )
        else:
            mc_dropout_results = _run_mc_dropout_centralized(
                model=model,
                config=config,
                task_type=task,
                device=device,
                amp_dtype=amp_dtype,
                n_passes=mc_dropout_passes,
                output_dir=output_dir,
            )

        # Set model back to eval mode
        model.eval()
        orc_logger.info("MC Dropout evaluation complete, model restored to eval mode")

    # ── f-h. Build payload and save ───────────────────────────────────────
    primary_metric = _extract_primary_metric(config)

    # Extract target_column and classes from config for experiment insights
    target_column = None
    classes = None
    if task == "classification":
        target_column = str(getattr(cfg.data, "class_column", "")) or None
        if target_column:
            try:
                raw_classes = getattr(cfg.data, target_column, None)
                if raw_classes and not isinstance(raw_classes, str):
                    classes = list(raw_classes)
            except Exception:
                classes = None
    elif task == "regression":
        target_column = str(getattr(cfg.data, "regression_column", "")) or None

    # Build mode-specific config dict — centralized gets checkpoint_epoch,
    # federated gets checkpoint_round / selection_mode / model_variant.
    # No nulls from the wrong mode.
    payload_config: dict[str, Any] = {
        "checkpoint": ckpt_path.name,
        "use_lora": use_lora,
        "primary_metric": primary_metric,
        "task": task,
        "target_column": target_column,
        "classes": classes,
    }
    if not is_federated:
        payload_config["checkpoint_epoch"] = checkpoint.get("epoch")

    payload = build_unified_result_payload(
        config=payload_config,
        per_client_metrics=per_client_metrics,
        primary_metric=primary_metric,
        is_federated=is_federated,
        checkpoint_selection_mode=selection_meta.get("mode"),
        checkpoint_round=selection_meta.get("selected_round"),
    )

    # Add MC Dropout uncertainty results to payload
    if mc_dropout and mc_dropout_results:
        payload["mc_dropout_uncertainty"] = to_builtin(mc_dropout_results)

    # Determine output directory
    if output_dir is None:
        if is_federated:
            effective_dir = selection_meta.get("checkpoint_dir", ckpt_path.parent)
        else:
            effective_dir = checkpoint_dir if checkpoint_dir else ckpt_path.parent
        output_dir = effective_dir / "test_evaluation"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / output_file_name
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2)
    orc_logger.info(f"Saved results to {result_path}")

    return result_path


# ── Internal helpers for run_unified_evaluation ────────────────────────────

def _extract_primary_metric(config: dict[str, Any]) -> str:
    """Extract primary metric name from config."""
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    es_cfg = train_cfg.get("early_stopping", {}) if isinstance(train_cfg, dict) else {}
    metric = es_cfg.get("metric", "mcc")
    return str(metric)


def _resolve_centralized_checkpoint(
    checkpoint_dir: Path | None,
    checkpoint_path: Path | None,
    checkpoint_epoch: int | None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve checkpoint for centralized evaluation."""
    # Direct path takes precedence
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists() or not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        return ckpt_path, {
            "mode": "direct",
            "selected_round": None,
            "checkpoint_dir": ckpt_path.parent,
        }

    if checkpoint_dir is None:
        raise ValueError(
            "At least one of --checkpoint-dir or --checkpoint-path is required "
            "for centralized evaluation."
        )

    checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_epoch is not None:
        ckpt_files = sorted(checkpoint_dir.glob("**/*.pt"))
        matching = [
            p for p in ckpt_files
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

    return ckpt_path, {"mode": "centralized", "selected_round": None, "checkpoint_dir": checkpoint_dir}


def _resolve_federated_checkpoint(
    checkpoint_dir: Path | None,
    checkpoint_path: Path | None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve checkpoint for federated evaluation."""
    from flwr_cinema.federated.evaluate import (
        resolve_checkpoint_for_evaluation,
    )

    if checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required for federated evaluation.")

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists() or not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        return ckpt_path, {
            "mode": "direct",
            "selected_round": _extract_round_from_path(ckpt_path),
            "checkpoint_dir": ckpt_path.parent,
        }

    resolved_checkpoint_dir = resolve_federated_checkpoint_dir(checkpoint_dir)
    # Load provisional config from last round to get primary metric (config only, no model)
    last_ckpt, _ = find_last_federated_round(resolved_checkpoint_dir)
    provisional_config = _load_config_near_checkpoint(last_ckpt, output_dir=resolved_checkpoint_dir)
    primary_metric, direction = _extract_primary_metric_meta(provisional_config)

    ckpt_path, round_num = find_best_federated_round(
        checkpoint_dir=resolved_checkpoint_dir,
        primary_metric=primary_metric,
        direction=direction,
    )
    return ckpt_path, {
        "mode": "best",
        "selected_round": round_num,
        "checkpoint_dir": resolved_checkpoint_dir,
    }


def _extract_round_from_path(path: Path) -> int | None:
    parts = path.stem.split("_")
    for idx, token in enumerate(parts):
        if token == "round" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


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


def _run_centralized_per_client(
    model: torch.nn.Module,
    config: dict[str, Any],
    task_type: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, tuple[int, dict[str, float]]]:
    """Run per-client evaluation for centralized mode."""
    from flwr_cinema.tasks.adapters import get_centralized_task_adapter
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(config)
    adapter = get_centralized_task_adapter(cfg)
    task_spec = adapter.task_spec
    eval_dataloader_fn = task_spec.eval_fn_factory()

    data_dir = Path(cfg.data.combined.dir).expanduser()
    data_dir = resolve_data_dir_with_fallback(data_dir)

    train_meta_df = load_metadata(data_dir / "train_metadata.csv", metadata_type="train")
    test_meta_df = load_metadata(data_dir / "test_metadata.csv", metadata_type="test")
    test_meta_df = filter_metadata_by_client_config(config=config, meta_df=test_meta_df, metadata_type="test")

    _, val_transform = task_spec.transform_factory(cfg)

    views = [cfg.model.views] if isinstance(cfg.model.views, str) else cfg.model.views
    patch_size_dict = {
        v: cfg.data.sax.patch_size if v == "sax" else cfg.data.lax.patch_size for v in views
    }
    spacing_dict = {
        v: cfg.data.sax.spacing if v == "sax" else cfg.data.lax.spacing for v in views
    }

    per_client_metrics: dict[str, tuple[int, dict[str, float]]] = {}

    for client_id in sorted(test_meta_df["client"].unique()):
        client_meta = test_meta_df[test_meta_df["client"] == client_id]
        n_samples = len(client_meta)

        client_dataset = task_spec.dataset_factory(
            config=cfg,
            data_dir=data_dir / "test",
            meta_df=client_meta.reset_index(drop=True),
            transform=val_transform,
            train_meta_df=train_meta_df,
        )

        client_dataloader = DataLoader(
            dataset=client_dataset,
            sampler=SequentialSampler(client_dataset),
            batch_size=1,
            drop_last=False,
            pin_memory=True,
            num_workers=cfg.train.n_workers,
        )

        if task_type == "classification":
            metrics, _, _ = classification_eval_dataloader_with_balanced_accuracy(
                model=model,
                dataloader=client_dataloader,
                patch_size_dict=patch_size_dict,
                amp_dtype=amp_dtype,
                device=device,
            )
        elif task_type == "segmentation":
            metrics = _evaluate_segmentation(
                model=model,
                dataloader=client_dataloader,
                config=config,
                device=device,
            )
        else:
            # Use eval_utils compute_metrics_on_dataloader for R² support (cinema's
            # regression_eval_dataloader is equivalent but doesn't include R²).
            reg_mean = getattr(client_dataset, "reg_mean", 0.0)
            reg_std = getattr(client_dataset, "reg_std", 1.0)
            metrics = compute_metrics_on_dataloader(
                model=model,
                dataloader=client_dataloader,
                reg_mean=reg_mean,
                reg_std=reg_std,
                config=config,
                device=device,
            )

        per_client_metrics[client_id] = (n_samples, normalize_metric_values(metrics))

    return per_client_metrics


def _run_federated_per_client(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    task_type: str,
    device: torch.device,
    base_state: dict[str, torch.Tensor],
) -> dict[str, tuple[int, dict[str, float]]]:
    """Run per-client evaluation for federated mode (reuses existing federated helpers)."""
    from flwr_cinema.federated.evaluate import evaluate_single_client

    datasets_by_client, evaluate_fn = _load_federated_client_datasets(config, task_type)

    per_client_metrics: dict[str, tuple[int, dict[str, float]]] = {}

    for client_id, (n_samples, dataset) in sorted(datasets_by_client.items(), key=lambda kv: str(kv[0])):
        result = evaluate_single_client(
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
        per_client_metrics[client_id] = (
            int(result["num_samples"]),
            result["metrics"],
        )

    return per_client_metrics


def _load_federated_client_datasets(
    config: dict[str, Any],
    task_type: str,
) -> tuple[dict[str, tuple[int, Any]], Callable]:
    """Load federated client datasets and evaluation function."""
    from flwr_cinema.federated.evaluate import load_client_datasets
    return load_client_datasets(config=config, task_type=task_type)


# ── MC Dropout helpers for run_unified_evaluation ─────────────────────────

def _build_centralized_dataloader(
    config: dict[str, Any],
    task_type: str,
    amp_dtype: torch.dtype,
    client_id: str | None = None,
    preloaded_train_meta: pd.DataFrame | None = None,
    preloaded_test_meta: pd.DataFrame | None = None,
) -> tuple[DataLoader, dict[str, tuple[int, ...]], dict[str, tuple[float, ...]] | None]:
    """Build a dataloader for centralized MC Dropout evaluation.

    Args:
        preloaded_train_meta: Pre-loaded train metadata (avoids reloading per client).
        preloaded_test_meta: Pre-loaded test metadata (avoids reloading per client).
    """
    from flwr_cinema.tasks.adapters import get_centralized_task_adapter
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(config)
    adapter = get_centralized_task_adapter(cfg)
    task_spec = adapter.task_spec

    data_dir = Path(cfg.data.combined.dir).expanduser()
    data_dir = resolve_data_dir_with_fallback(data_dir)

    train_meta_df = preloaded_train_meta if preloaded_train_meta is not None else load_metadata(data_dir / "train_metadata.csv", metadata_type="train")
    test_meta_df = preloaded_test_meta if preloaded_test_meta is not None else load_metadata(data_dir / "test_metadata.csv", metadata_type="test")

    if client_id is not None:
        test_meta_df = test_meta_df[test_meta_df["client"] == client_id].reset_index(drop=True)
    else:
        test_meta_df = filter_metadata_by_client_config(config=config, meta_df=test_meta_df, metadata_type="test")

    _, val_transform = task_spec.transform_factory(cfg)

    views = [cfg.model.views] if isinstance(cfg.model.views, str) else cfg.model.views
    patch_size_dict = {
        v: cfg.data.sax.patch_size if v == "sax" else cfg.data.lax.patch_size for v in views
    }
    spacing_dict = {
        v: cfg.data.sax.spacing if v == "sax" else cfg.data.lax.spacing for v in views
    }

    dataset = task_spec.dataset_factory(
        config=cfg,
        data_dir=data_dir / "test",
        meta_df=test_meta_df,
        transform=val_transform,
        train_meta_df=train_meta_df,
    )

    dataloader = DataLoader(
        dataset=dataset,
        sampler=SequentialSampler(dataset),
        batch_size=1,
        drop_last=False,
        pin_memory=False,  # MC dropout: avoid CUDA pinned memory exhaustion
        num_workers=0,
    )

    return dataloader, patch_size_dict, spacing_dict


def _run_mc_dropout_centralized(
    model: torch.nn.Module,
    config: dict[str, Any],
    task_type: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    n_passes: int,
    output_dir: Path | None,
) -> dict[str, Any]:
    """Run MC Dropout uncertainty for centralized per-client evaluation."""
    from omegaconf import OmegaConf
    from flwr_cinema.scripts.eval_utils import load_metadata, resolve_data_dir_with_fallback, filter_metadata_by_client_config

    cfg = OmegaConf.create(config)
    data_dir = Path(cfg.data.combined.dir).expanduser()
    data_dir = resolve_data_dir_with_fallback(data_dir)
    test_meta_df = load_metadata(data_dir / "test_metadata.csv", metadata_type="test")
    test_meta_df = filter_metadata_by_client_config(config=config, meta_df=test_meta_df, metadata_type="test")
    train_meta_df = load_metadata(data_dir / "train_metadata.csv", metadata_type="train")

    result: dict[str, Any] = {"task_type": task_type}

    # ── Per-client ─────────────────────────────────────────────────────
    if "client" in test_meta_df.columns:
        per_client: dict[str, Any] = {}
        for client_id in sorted(test_meta_df["client"].unique()):
            client_dl, client_ps, client_sp = _build_centralized_dataloader(
                config=config, task_type=task_type, amp_dtype=amp_dtype,
                client_id=client_id,
                preloaded_train_meta=train_meta_df,
                preloaded_test_meta=test_meta_df,
            )
            if task_type == "segmentation":
                per_client[client_id] = _run_segmentation_mc_dropout_streaming_entry(
                    model=model, dataloader=client_dl,
                    patch_size_dict=client_ps, spacing_dict=client_sp or {},
                    amp_dtype=amp_dtype, device=device, n_passes=n_passes,
                )
            else:
                _, _, c_outputs, c_labels = mc_dropout_predict(
                    model=model, dataloader=client_dl, task_type=task_type,
                    patch_size_dict=client_ps, spacing_dict=client_sp,
                    amp_dtype=amp_dtype, device=device, n_passes=n_passes,
                )
                per_client[client_id] = _build_mc_uncertainty_entry(
                    task_type=task_type, all_outputs=c_outputs, mean=np.array([]),
                    dataloader=client_dl, output_dir=output_dir,
                    client_id=client_id, all_labels=c_labels,
                    spacing_dict=client_sp,
                )
        result["per_client"] = per_client

        # ── Overall aggregation (regression only) ────────────────────────
        if task_type == "regression":
            per_client_agg = []
            for client_entry in per_client.values():
                cm = client_entry.get("clinical_metrics", {})
                prob = client_entry.get("probabilistic", {}).get("regression", {})
                n = client_entry.get("n_samples", 0)
                if n > 0 and cm:
                    per_client_agg.append({
                        "n_samples": n,
                        "rmse": cm["rmse"],
                        "restored_rmse": cm["restored_rmse"],
                        "mae": cm["mae"],
                        "restored_mae": cm["restored_mae"],
                        "picp": cm["picp"],
                        "mpiw": cm["mpiw"],
                        "restored_mpiw": cm["restored_mpiw"],
                        "nll": cm["nll"],
                        "ss_res": cm["ss_res"],
                        "ss_tot": cm["ss_tot"],
                        "mean_y": cm["mean_y"],
                        "r2": cm["r2"],
                        "mean_std": prob.get("mean_std", 0.0),
                        "restored_mean_std": prob.get("restored_mean_std", 0.0),
                    })
            overall = _compute_overall_mc_regression(per_client_agg)
            if overall is not None:
                result["weighted_avg"] = overall

    return result


def _build_mc_uncertainty_entry(
    task_type: str,
    all_outputs: list,
    mean: np.ndarray,
    dataloader: DataLoader,
    output_dir: Path | None,
    client_id: str | None = None,
    all_labels: np.ndarray | None = None,
    spacing_dict: dict[str, tuple[float, ...]] | None = None,
) -> dict[str, Any]:
    """Build MC uncertainty dict with probabilistic + clinical metrics sections.

    Returns a dict with two sub-keys:
    - probabilistic: entropy, MI, per-class std (classification), std+CI (regression), Dice std (seg)
    - clinical_metrics: mean and std of each clinical metric across passes
    """
    entry: dict[str, Any] = {"n_passes": len(all_outputs)}
    probabilistic: dict[str, Any] = {}

    if task_type == "classification":
        unc = compute_classification_uncertainty(all_probs=np.stack(all_outputs, axis=0))
        probabilistic["classification"] = {
            "mean_entropy": float(unc["mean_entropy"]),
            "mutual_info": float(unc["mutual_info"]),
            "per_class_std": to_builtin(unc["per_class_std"]),
        }
        if all_labels is not None:
            entry["n_samples"] = len(all_labels)
    elif task_type == "regression":
        reg_mean = getattr(dataloader.dataset, "reg_mean", 0.0)
        reg_std = getattr(dataloader.dataset, "reg_std", 1.0)
        stacked = np.stack(all_outputs, axis=0).squeeze(-1) if all_outputs[0].ndim == 2 else np.stack(all_outputs, axis=0)
        unc = compute_regression_uncertainty(
            all_preds=stacked, reg_mean=reg_mean, reg_std=reg_std,
        )
        probabilistic["regression"] = {
            "mean_std": float(np.mean(unc["std"])),
            "ci_lower": to_builtin(unc["ci_lower"]),
            "ci_upper": to_builtin(unc["ci_upper"]),
            "restored_mean_std": float(np.mean(unc["restored_std"])),
        }
        if all_labels is not None:
            entry["n_samples"] = len(all_labels)
    elif task_type == "segmentation":
        eff_output_dir = Path(output_dir) if output_dir else Path(".")
        eff_output_dir.mkdir(parents=True, exist_ok=True)
        unc = compute_segmentation_uncertainty(all_outputs=all_outputs, output_dir=eff_output_dir, spacing_dict=spacing_dict)
        probabilistic["segmentation"] = {
            "mean_dice_std": float(unc["mean_dice_std"]),
            "per_class_dice_std": to_builtin(unc["per_class_dice_std"]),
            "mean_predictive_variance": float(unc["mean_predictive_variance"]),
            "per_class_predictive_variance": to_builtin(unc["per_class_predictive_variance"]),
            "mean_predictive_entropy": float(unc["mean_predictive_entropy"]),
            "mean_predictive_entropy_foreground": float(unc["mean_predictive_entropy_foreground"]),
        }

    entry["probabilistic"] = probabilistic

    # ── Clinical metrics uncertainty (mean ± std per metric across passes) ─
    if task_type == "segmentation" and all_outputs:
        entry["clinical_metrics"] = _compute_segmentation_clinical_uncertainty(all_outputs, spacing_dict=spacing_dict)
    elif task_type == "regression" and all_labels is not None:
        reg_mean = getattr(dataloader.dataset, "reg_mean", 0.0)
        reg_std = getattr(dataloader.dataset, "reg_std", 1.0)
        stacked_cm = np.stack(all_outputs, axis=0).squeeze(-1) if all_outputs[0].ndim == 2 else np.stack(all_outputs, axis=0)
        entry["clinical_metrics"] = compute_regression_clinical_metrics_uncertainty(
            all_preds=stacked_cm, labels=all_labels,
            reg_mean=reg_mean, reg_std=reg_std,
        )
    elif all_labels is not None and task_type == "classification":
        entry["clinical_metrics"] = compute_classification_clinical_metrics_uncertainty(
            all_outputs=all_outputs,
            all_labels=all_labels,
        )

    return entry


def _segmentation_clinical_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    spacing: tuple[float, ...],
) -> dict[str, float]:
    """Lightweight segmentation metrics (Dice, IoU, Stability, Volume) — no Hausdorff.

    Mirrors segmentation_metrics from cinema but skips the expensive
    compute_hausdorff_distance to avoid memory fragmentation with many passes.
    """
    import torch.nn.functional as F

    from monai.metrics import compute_dice, compute_iou
    from cinema.metric import get_volumes, stability_score

    n_classes = logits.shape[1] - 1
    labels_1d = labels.squeeze(dim=1).long()

    pred_labels = torch.argmax(logits, dim=1)
    pred_mask = F.one_hot(pred_labels, n_classes + 1).moveaxis(-1, 1)
    true_mask = F.one_hot(labels_1d, n_classes + 1).moveaxis(-1, 1)

    dice = compute_dice(y_pred=pred_mask, y=true_mask, num_classes=n_classes + 1)
    iou = compute_iou(y_pred=pred_mask, y=true_mask)
    stability = stability_score(logits=logits)
    true_volumes = get_volumes(mask=true_mask, spacing=spacing)
    pred_volumes = get_volumes(mask=pred_mask, spacing=spacing)

    result: dict[str, float] = {}
    for i in range(n_classes):
        cls_idx = i + 1
        result[f"class_{cls_idx}_dice_score"] = float(dice[0, cls_idx].item())
        result[f"class_{cls_idx}_iou_score"] = float(iou[0, cls_idx].item())
        result[f"class_{cls_idx}_stability_score"] = float(stability[0, cls_idx].item())
        result[f"class_{cls_idx}_true_volume"] = float(true_volumes[0, cls_idx].item())
        result[f"class_{cls_idx}_pred_volume"] = float(pred_volumes[0, cls_idx].item())

    result["mean_dice_score"] = float(dice[0, 1:].mean().item())
    result["mean_iou_score"] = float(iou[0, 1:].mean().item())
    result["mean_stability_score"] = float(stability[0, 1:].mean().item())
    return result


def _compute_segmentation_clinical_uncertainty(
    all_outputs: list[list[dict[str, dict[str, torch.Tensor]]]],
    spacing_dict: dict[str, tuple[float, ...]] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute mean and std of segmentation clinical metrics across MC passes.

    Uses lightweight metrics (no Hausdorff) to avoid OOM.
    """

    metrics_per_pass: list[dict[str, float]] = []

    for pass_outputs in all_outputs:
        pass_metrics_accum: dict[str, list[float]] = {}
        for output in pass_outputs:
            logits_dict = output["logits"]
            labels_dict = output["labels"]
            first_view = next(iter(logits_dict))
            logits = logits_dict[first_view]
            labels = labels_dict[first_view]
            sp = spacing_dict.get(first_view, (1.0, 1.0, 1.0)) if spacing_dict else (1.0, 1.0, 1.0)
            m = _segmentation_clinical_metrics(logits, labels, spacing=sp)
            for k, v in m.items():
                pass_metrics_accum.setdefault(k, []).append(v)
        metrics_per_pass.append({k: float(np.mean(v)) for k, v in pass_metrics_accum.items()})

    if not metrics_per_pass:
        return {}

    result: dict[str, dict[str, float]] = {}
    for metric_name in metrics_per_pass[0].keys():
        values = [m[metric_name] for m in metrics_per_pass]
        result[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return result


def _run_segmentation_mc_dropout_streaming_entry(
    model: torch.nn.Module,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
    n_passes: int,
) -> dict[str, Any]:
    """Compute segmentation MC Dropout uncertainty without storing all pass logits."""
    from models.cinema.cinema.segmentation.train import segmentation_eval
    from models.cinema.cinema.transform import crop_start

    def _crop_labels(batch: dict[str, torch.Tensor], logits_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        labels_cpu: dict[str, torch.Tensor] = {}
        for view in logits_dict:
            width = int(batch[f"{view}_width"][0])
            height = int(batch[f"{view}_height"][0])
            label = batch[f"{view}_label"]
            if len(patch_size_dict[view]) == 3:
                n_slices = int(batch["n_slices"][0])
                label = crop_start(label, (*label.shape[:2], width, height, n_slices))
            elif len(patch_size_dict[view]) == 2:
                label = crop_start(label, (*label.shape[:2], width, height))
            labels_cpu[view] = label.cpu()
        return labels_cpu

    pass_metrics_accum: list[dict[str, list[float]]] = [dict() for _ in range(n_passes)]
    per_sample_variance: list[np.ndarray] = []
    per_sample_entropy: list[float] = []
    per_sample_entropy_fg: list[float] = []
    n_samples = 0

    for batch in dataloader:
        probs_sum: torch.Tensor | None = None
        probs_sq_sum: torch.Tensor | None = None
        first_view: str | None = None

        for pass_idx in range(n_passes):
            with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits_dict, _ = segmentation_eval(
                    model=model,
                    batch=batch,
                    patch_size_dict=patch_size_dict,
                    spacing_dict=spacing_dict,
                    amp_dtype=amp_dtype,
                    device=device,
                    metrics_fn=None,
                )

            cpu_logits = {view: logits.detach().cpu() for view, logits in logits_dict.items()}
            labels_cpu = _crop_labels(batch, logits_dict)

            first_view = next(iter(cpu_logits))
            logits = cpu_logits[first_view]
            labels = labels_cpu[first_view]
            sp = spacing_dict.get(first_view, (1.0, 1.0, 1.0))

            metrics = _segmentation_clinical_metrics(logits, labels, spacing=sp)
            for key, value in metrics.items():
                pass_metrics_accum[pass_idx].setdefault(key, []).append(value)

            probs = torch.softmax(logits.float(), dim=1)
            if probs_sum is None:
                probs_sum = torch.zeros_like(probs)
                probs_sq_sum = torch.zeros_like(probs)
            probs_sum += probs
            probs_sq_sum += probs * probs

        if probs_sum is None or probs_sq_sum is None or first_view is None:
            continue

        n_samples += 1
        mean_probs = probs_sum / n_passes
        var_probs = torch.clamp((probs_sq_sum / n_passes) - (mean_probs * mean_probs), min=0.0)
        var_per_class = var_probs.mean(dim=(-1, -2, -3)).squeeze(0)
        per_sample_variance.append(var_per_class.float().cpu().numpy())

        mean_probs_clipped = torch.clamp(mean_probs, min=1e-10)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs_clipped), dim=1)
        per_sample_entropy.append(float(entropy.mean().item()))

        predicted_foreground = torch.argmax(mean_probs, dim=1).squeeze(0) > 0
        fg_entropy = entropy.squeeze(0)[predicted_foreground]
        if fg_entropy.numel() > 0:
            per_sample_entropy_fg.append(float(fg_entropy.mean().item()))

    metrics_per_pass = [
        {key: float(np.mean(values)) for key, values in pass_accum.items()}
        for pass_accum in pass_metrics_accum
    ]

    clinical_metrics: dict[str, dict[str, float]] = {}
    if metrics_per_pass and metrics_per_pass[0]:
        for metric_name in metrics_per_pass[0].keys():
            values = [m[metric_name] for m in metrics_per_pass]
            clinical_metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    dice_std = {
        key: value["std"]
        for key, value in clinical_metrics.items()
        if "dice" in key
    }
    mean_dice_std = float(np.mean(list(dice_std.values()))) if dice_std else 0.0

    if per_sample_variance:
        stacked_vars = np.stack(per_sample_variance, axis=0)
        per_class_predictive_variance = stacked_vars.mean(axis=0)
        mean_predictive_variance = float(np.mean(per_class_predictive_variance))
    else:
        per_class_predictive_variance = np.array([])
        mean_predictive_variance = 0.0

    mean_predictive_entropy = float(np.mean(per_sample_entropy)) if per_sample_entropy else 0.0
    mean_predictive_entropy_fg = float(np.mean(per_sample_entropy_fg)) if per_sample_entropy_fg else 0.0
    if per_sample_variance:
        n_classes = len(per_sample_variance[0])
        if n_classes > 1:
            mean_predictive_entropy = mean_predictive_entropy / np.log(n_classes)
            mean_predictive_entropy_fg = mean_predictive_entropy_fg / np.log(n_classes)

    return {
        "n_passes": n_passes,
        "probabilistic": {
            "segmentation": {
                "mean_dice_std": mean_dice_std,
                "per_class_dice_std": to_builtin(dice_std),
                "mean_predictive_variance": mean_predictive_variance,
                "per_class_predictive_variance": to_builtin(_label_predictive_variance(per_class_predictive_variance)),
                "mean_predictive_entropy": mean_predictive_entropy,
                "mean_predictive_entropy_foreground": mean_predictive_entropy_fg,
            }
        },
        "clinical_metrics": clinical_metrics,
    }


def _compute_overall_mc_regression(
    per_client_data: list[dict[str, float]],
) -> dict[str, float] | None:
    """Compute overall weighted-average metrics and global R² across federated clients.

    Uses SS decomposition for global R²:
      global_ss_res = sum(client ss_res)
      global_ss_tot = within_ss_tot + between_variance
      global_r2 = 1 - global_ss_res / global_ss_tot

    Args:
        per_client_data: List of dicts, each with keys:
            n_samples, rmse, restored_rmse, mae, restored_mae,
            picp, mpiw, restored_mpiw, nll,
            ss_res, ss_tot, mean_y, r2,
            mean_std, restored_mean_std.

    Returns:
        Dict with weighted_avg section, or None if no clients.
    """
    import math as _math

    if not per_client_data:
        return None

    total_samples = sum(c["n_samples"] for c in per_client_data)
    if total_samples == 0:
        return None

    # Weighted RMSE (quadratic mean — standard for RMSE aggregation)
    weighted_rmse = _math.sqrt(
        sum(c["n_samples"] * c["rmse"] ** 2 for c in per_client_data) / total_samples
    )
    weighted_restored_rmse = _math.sqrt(
        sum(c["n_samples"] * c["restored_rmse"] ** 2 for c in per_client_data) / total_samples
    )

    # Weighted MAE (linear)
    weighted_mae = sum(c["n_samples"] * c["mae"] for c in per_client_data) / total_samples
    weighted_restored_mae = sum(c["n_samples"] * c["restored_mae"] for c in per_client_data) / total_samples

    # Weighted PICP, MPIW, NLL (linear)
    weighted_picp = sum(c["n_samples"] * c["picp"] for c in per_client_data) / total_samples
    weighted_mpiw = sum(c["n_samples"] * c["mpiw"] for c in per_client_data) / total_samples
    weighted_restored_mpiw = sum(c["n_samples"] * c["restored_mpiw"] for c in per_client_data) / total_samples
    weighted_nll = sum(c["n_samples"] * c["nll"] for c in per_client_data) / total_samples

    # Weighted predictive std (linear — mean of per-sample stds across passes)
    weighted_mean_std = sum(c["n_samples"] * c.get("mean_std", 0.0) for c in per_client_data) / total_samples
    weighted_restored_mean_std = sum(c["n_samples"] * c.get("restored_mean_std", 0.0) for c in per_client_data) / total_samples

    # Global R² via SS decomposition
    global_ss_res = sum(c["ss_res"] for c in per_client_data)
    within_variance = sum(c["ss_tot"] for c in per_client_data)
    global_mean_y = sum(c["mean_y"] * c["n_samples"] for c in per_client_data) / total_samples
    between_variance = sum(
        c["n_samples"] * (c["mean_y"] - global_mean_y) ** 2 for c in per_client_data
    )
    global_ss_tot = within_variance + between_variance
    global_r2 = 1.0 - global_ss_res / global_ss_tot if global_ss_tot > 0 else 0.0

    return {
        "rmse": weighted_rmse,
        "restored_rmse": weighted_restored_rmse,
        "mae": weighted_mae,
        "restored_mae": weighted_restored_mae,
        "picp": weighted_picp,
        "mpiw": weighted_mpiw,
        "restored_mpiw": weighted_restored_mpiw,
        "nll": weighted_nll,
        "mean_std": weighted_mean_std,
        "restored_mean_std": weighted_restored_mean_std,
        "r2": float(global_r2),
    }


def _run_mc_dropout_federated(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    task_type: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    n_passes: int,
    base_state: dict[str, torch.Tensor],
    output_dir: Path | None,
) -> dict[str, Any]:
    """Run MC Dropout for federated per-client evaluation."""
    from omegaconf import OmegaConf
    from flwr_cinema.federated.evaluate import apply_private_artifacts, build_eval_dataloader

    datasets_by_client, _ = _load_federated_client_datasets(config, task_type)

    cfg = OmegaConf.create(config)
    views = [cfg.model.views] if isinstance(cfg.model.views, str) else cfg.model.views
    patch_size_dict = {
        v: cfg.data.sax.patch_size if v == "sax" else cfg.data.lax.patch_size for v in views
    }
    spacing_dict = {
        v: cfg.data.sax.spacing if v == "sax" else cfg.data.lax.spacing for v in views
    }

    result: dict[str, Any] = {"task_type": task_type}

    # ── Per-client (with private artifacts) ─────────────────────────────
    per_client_uncertainty: dict[str, Any] = {}
    for client_id, (n_samples, dataset) in sorted(datasets_by_client.items(), key=lambda kv: str(kv[0])):
        model.load_state_dict(base_state, strict=False)
        apply_private_artifacts(model=model, checkpoint=checkpoint, client_id=str(client_id))
        model.to(device)

        dataloader = build_eval_dataloader(dataset)
        if task_type == "segmentation":
            per_client_uncertainty[client_id] = _run_segmentation_mc_dropout_streaming_entry(
                model=model, dataloader=dataloader,
                patch_size_dict=patch_size_dict, spacing_dict=spacing_dict,
                amp_dtype=amp_dtype, device=device, n_passes=n_passes,
            )
        else:
            _, _, all_outputs, all_labels = mc_dropout_predict(
                model=model, dataloader=dataloader, task_type=task_type,
                patch_size_dict=patch_size_dict, spacing_dict=spacing_dict,
                amp_dtype=amp_dtype, device=device, n_passes=n_passes,
            )
            per_client_uncertainty[client_id] = _build_mc_uncertainty_entry(
                task_type=task_type, all_outputs=all_outputs, mean=np.array([]),
                dataloader=dataloader, output_dir=output_dir, client_id=client_id,
                all_labels=all_labels, spacing_dict=spacing_dict,
            )
    result["per_client"] = per_client_uncertainty

    # ── Overall aggregation (regression only) ────────────────────────────
    if task_type == "regression":
        per_client_agg = []
        for client_entry in per_client_uncertainty.values():
            cm = client_entry.get("clinical_metrics", {})
            prob = client_entry.get("probabilistic", {}).get("regression", {})
            n = client_entry.get("n_samples", 0)
            if n > 0 and cm:
                per_client_agg.append({
                    "n_samples": n,
                    "rmse": cm["rmse"],
                    "restored_rmse": cm["restored_rmse"],
                    "mae": cm["mae"],
                    "restored_mae": cm["restored_mae"],
                    "picp": cm["picp"],
                    "mpiw": cm["mpiw"],
                    "restored_mpiw": cm["restored_mpiw"],
                    "nll": cm["nll"],
                    "ss_res": cm["ss_res"],
                    "ss_tot": cm["ss_tot"],
                    "mean_y": cm["mean_y"],
                    "r2": cm["r2"],
                    "mean_std": prob.get("mean_std", 0.0),
                    "restored_mean_std": prob.get("restored_mean_std", 0.0),
                })
        overall = _compute_overall_mc_regression(per_client_agg)
        if overall is not None:
            result["weighted_avg"] = overall

    return result
