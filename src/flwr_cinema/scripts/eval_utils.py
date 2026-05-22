"""Shared evaluation utilities for centralized and federated regression models."""

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
    `primary_validation_metrics`. Legacy histories without this field are treated
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
    - preferred input is direct `.../federated_server/checkpoints`
    - experiment root containing `federated_server/checkpoints` is accepted
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

    # Real contract: metrics_history is a dict with evaluate_metrics keyed by round.
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
        primary_metric: Metric key expected inside each ``fedavg_metrics`` payload.
        direction: Optimization direction, one of ``min`` or ``max``.

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


def get_round_participants(output_dir: Path, round_num: int) -> list[str]:
    """Get list of clients that participated in a specific federated round.
    
    Infers from:
    1. Aggregated metrics history (client_steps keys)
    2. Client checkpoint directories
    3. Config file
    
    Args:
        output_dir: Federated training output directory.
        round_num: Round number.
    
    Returns:
        List of client IDs (e.g., ["acdc", "GE"]).
    """
    federated_server_dir = output_dir / "federated_server"
    
    # Try to extract from metrics history
    metrics_file = federated_server_dir / "metrics_history.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                metrics_history = json.load(f)
                for entry in metrics_history:
                    if entry.get("round") == round_num:
                        if "client_steps" in entry:
                            return list(entry["client_steps"].keys())
        except Exception as e:
            logger.warning(f"Failed to extract participants from metrics: {e}")
    
    # Fall back to client directories
    client_dirs = sorted(federated_server_dir.parent.glob("client_*/"))
    if client_dirs:
        participants = [d.name.replace("client_", "") for d in client_dirs]
        logger.info(f"Inferred participants from client directories: {participants}")
        return participants
    
    # Fall back to config
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        config_path = output_dir.parent / "config.yaml"
    if not config_path.exists():
        config_path = output_dir.parent.parent / "config.yaml"
    
    if config_path.exists():
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_path)
            if "federated" in cfg and "n_clients" in cfg.federated:
                # Default clients based on config
                if cfg.federated.n_clients >= 2:
                    return ["acdc", "GE", "Philips", "Siemens", "Canon"]  # Common default
        except Exception as e:
            logger.warning(f"Failed to read config: {e}")
    
    logger.warning("Could not determine round participants, assuming all clients")
    return ["acdc", "GE", "Philips", "Siemens", "Canon"]  # Default assumption


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
    
    # Find and load config
    config_paths = []
    if output_dir:
        config_paths.extend([
            output_dir / "config.yaml",
            output_dir.parent / "config.yaml",
            output_dir.parent.parent / "config.yaml",
            output_dir / ".hydra" / "config.yaml",
            output_dir.parent / ".hydra" / "config.yaml",
        ])
    
    # Also search from checkpoint directory
    config_paths.extend([
        checkpoint_path.parent / "config.yaml",
        checkpoint_path.parent.parent / "config.yaml",
        checkpoint_path.parent.parent.parent / "config.yaml",
        checkpoint_path.parent.parent / ".hydra" / "config.yaml",
        checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml",
    ])
    
    config_dict = None
    for config_path in config_paths:
        if config_path.exists():
            try:
                from omegaconf import OmegaConf
                config_dict = OmegaConf.to_container(OmegaConf.load(config_path))
                logger.info(f"Loaded config from {config_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
    
    if config_dict is None:
        # Fall back to default config from configs/ directory
        try:
            from omegaconf import OmegaConf
            repo_root = Path(__file__).parent.parent
            default_config_path = repo_root / "configs" / "cinema_fed.yaml"
            if default_config_path.exists():
                config_dict = OmegaConf.to_container(OmegaConf.load(default_config_path))
                logger.warning(f"Using default config from {default_config_path}")
            else:
                raise FileNotFoundError(f"Could not find config.yaml for checkpoint {checkpoint_path}")
        except Exception as e:
            raise FileNotFoundError(f"Could not find config.yaml for checkpoint {checkpoint_path}") from e
    
    return checkpoint, config_dict # type: ignore


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
    
    # Denormalize and compute metrics on original scale
    restored_pred_labels = (pred_labels * reg_std) + reg_mean
    restored_true_labels = (true_labels * reg_std) + reg_mean
    restored_metrics = regression_metrics(
        true_labels=restored_true_labels,
        pred_labels=restored_pred_labels,
    )
    restored_metrics = {f"restored_{k}": v for k, v in restored_metrics.items()}
    metrics.update(restored_metrics)
    
    return metrics


def compute_weighted_metrics(
    metrics_by_client: dict[str, dict],
    test_meta_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute weighted average metrics across clients.
    
    Args:
        metrics_by_client: Dictionary mapping client_id to metrics dict.
        test_meta_df: Test metadata to get client sample counts.
    
    Returns:
        Weighted average metrics.
    """
    if not metrics_by_client:
        return {}
    
    # Identify client column
    if "client" in test_meta_df.columns:
        client_col = "client"
    elif "vendor" in test_meta_df.columns:
        client_col = "vendor"
    else:
        # Single client, no weighting needed
        return list(metrics_by_client.values())[0]
    
    # Get sample counts per client
    client_counts = test_meta_df[client_col].value_counts().to_dict()
    total_samples = len(test_meta_df)
    
    weighted_metrics = {}
    for metric_name in metrics_by_client[list(metrics_by_client.keys())[0]].keys():
        weighted_value = 0.0
        for client_id, metrics in metrics_by_client.items():
            count = client_counts.get(client_id, 1)
            weight = count / total_samples
            weighted_value += metrics[metric_name] * weight
        weighted_metrics[metric_name] = weighted_value
    
    return weighted_metrics


def to_builtin(obj: Any) -> Any:
    """Recursively convert numpy/torch values to JSON-serializable builtins."""
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
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

    metrics = eval_dataloader_fn(
        model=model,
        dataloader=dataloader,
        patch_size_dict=patch_size_dict,
        spacing_dict=spacing_dict,
        amp_dtype=amp_dtype,
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


def compute_cross_center_metrics(
    model: torch.nn.Module,
    datasets_by_client: dict[str, tuple[int, ClassificationDataset | RegressionDataset]],
    test_meta_df: pd.DataFrame,
    reg_mean: float,
    reg_std: float,
    config: dict,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Compute cross-center validation metrics.
    
    For each client's model, evaluate on all other clients' test data.
    Automatically detects task type (classification vs regression) and uses appropriate metrics.
    
    Args:
        model: Global model.
        datasets_by_client: Datasets split by client.
        test_meta_df: Test metadata.
        reg_mean: Normalization mean (for regression only).
        reg_std: Normalization std (for regression only).
        config: Configuration dictionary.
        device: Device to use.
    
    Returns:
        Dictionary mapping "client_X_to_client_Y" to metrics dict.
    """
    from omegaconf import OmegaConf
    
    cross_center_metrics = {}
    clients = sorted(datasets_by_client.keys())
    
    # Determine task type
    cfg = OmegaConf.create(config)
    task = getattr(cfg, "task", "regression")
    is_classification = task == "classification"
    
    # For each source client, evaluate on all target clients
    for source_client in clients:
        for target_client in clients:
            if source_client == target_client:
                continue
            
            # Create dataloader for target client data
            n_samples, dataset = datasets_by_client[target_client]
            sampler = SequentialSampler(dataset)
            dataloader = DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=1,
                drop_last=False,
                pin_memory=True,
                num_workers=0,
            )
            
            # Compute metrics (task-specific)
            try:
                if is_classification:
                    # For classification, try to use the classification metrics function if available
                    # Otherwise fall back to regression metrics (which may fail gracefully and be caught)
                    try:
                        from cinema_classification.federated.evaluate import compute_metrics_on_dataloader_with_balanced_accuracy
                        metrics, _, _ = compute_metrics_on_dataloader_with_balanced_accuracy(
                            model=model,
                            dataloader=dataloader,
                            config=config,
                            device=device,
                            task="classification",
                            reg_mean=None,
                            reg_std=None,
                        )
                    except (ImportError, TypeError):
                        # Fall back to generic metrics computation
                        metrics = compute_metrics_on_dataloader(
                            model=model,
                            dataloader=dataloader,
                            reg_mean=0,
                            reg_std=1,
                            config=config,
                            device=device,
                        )
                else:
                    # Use regression metrics
                    metrics = compute_metrics_on_dataloader(
                        model=model,
                        dataloader=dataloader,
                        reg_mean=reg_mean,
                        reg_std=reg_std,
                        config=config,
                        device=device,
                    )
                
                cross_center_key = f"{source_client}_to_{target_client}"
                cross_center_metrics[cross_center_key] = metrics
                logger.debug(f"Cross-center evaluation {cross_center_key}: success")
            except Exception as e:
                cross_center_key = f"{source_client}_to_{target_client}"
                logger.debug(f"Failed to evaluate {cross_center_key}: {type(e).__name__}: {e}")
                # Don't fail entire cross-center evaluation for a single pair - just skip it
                continue
    
    return cross_center_metrics


def format_results_json_csv(
    overall_metrics: dict[str, float],
    per_client_metrics: dict[str, dict[str, float]],
    weighted_metrics: dict[str, float],
    cross_center_metrics: dict[str, dict[str, float]] | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, dict[str, float] | dict[str, dict[str, float]]], pd.DataFrame, pd.DataFrame | None]:
    """Format evaluation results into JSON and CSV format.
    
    Supports both classification and regression metrics automatically.
    
    Args:
        overall_metrics: Overall metrics dictionary.
        per_client_metrics: Per-client metrics dictionary.
        weighted_metrics: Weighted average metrics.
        cross_center_metrics: Cross-center evaluation results (optional).
        output_dir: Directory to save results (optional).
    
    Returns:
        Tuple of (results_dict, metrics_dataframe, cross_center_dataframe)
    """
    # Classification metrics
    classification_metrics = [
        "accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc", 
        "sensitivity", "specificity"
    ]
    
    # Regression metrics
    regression_metrics_list = [
        "rmse", "mae", "max_error", "min_error",
        "restored_rmse", "restored_mae", "restored_max_error", "restored_min_error"
    ]
    
    # Detect which metrics are present (classification vs regression)
    all_metrics_keys = set(overall_metrics.keys()) | set(weighted_metrics.keys())
    for metrics in per_client_metrics.values():
        all_metrics_keys.update(metrics.keys())
    
    is_classification = any(m in all_metrics_keys for m in classification_metrics)
    is_regression = any(m in all_metrics_keys for m in regression_metrics_list)
    
    # Choose which metrics to keep based on what's present
    if is_classification:
        metrics_to_keep = classification_metrics
    elif is_regression:
        metrics_to_keep = regression_metrics_list
    else:
        # Keep all metrics if we can't determine type
        metrics_to_keep = list(all_metrics_keys)
    
    # Helper to convert numpy types to Python floats for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (float, int)):
            return obj
        else:
            return obj
    
    # Filter to only selected metrics
    overall_filtered = {k: v for k, v in overall_metrics.items() if k in metrics_to_keep}
    weighted_filtered = {k: v for k, v in weighted_metrics.items() if k in metrics_to_keep}
    per_client_filtered = {
        client: {k: v for k, v in metrics.items() if k in metrics_to_keep}
        for client, metrics in per_client_metrics.items()
    }
    
    # Prepare JSON results
    results_dict = {
        "overall": overall_filtered,
        "weighted": weighted_filtered,
        "per_client": per_client_filtered,
    }
    
    if cross_center_metrics:
        cross_center_filtered = {
            k: {m: v for m, v in metrics.items() if m in metrics_to_keep}
            for k, metrics in cross_center_metrics.items()
        }
        results_dict["cross_center"] = cross_center_filtered
    
    # Convert to serializable format with explicit type casting
    results_dict_serializable: dict[str, dict[str, float] | dict[str, dict[str, float]]] = convert_to_serializable(results_dict)  # type: ignore
    
    # Prepare DataFrames for CSV
    metrics_data: dict[str, list] = {"metric": []}
    metrics_data["overall"] = []
    metrics_data["weighted"] = []
    for client in sorted(per_client_filtered.keys()):
        metrics_data[client] = []
    
    for metric in metrics_to_keep:
        metrics_data["metric"].append(metric)
        metrics_data["overall"].append(float(overall_filtered.get(metric, np.nan)))
        metrics_data["weighted"].append(float(weighted_filtered.get(metric, np.nan)))
        for client in sorted(per_client_filtered.keys()):
            metrics_data[client].append(float(per_client_filtered[client].get(metric, np.nan)))
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Prepare cross-center DataFrame
    cross_center_df = None
    if cross_center_metrics:
        cross_center_data: dict[str, list] = {"metric": []}
        for key in sorted(cross_center_metrics.keys()):
            cross_center_data[key] = []
        
        for metric in metrics_to_keep:
            cross_center_data["metric"].append(metric)
            for key in sorted(cross_center_metrics.keys()):
                metrics = cross_center_metrics[key]
                cross_center_data[key].append(float(metrics.get(metric, np.nan)))
        
        cross_center_df = pd.DataFrame(cross_center_data)
    
    # Save to files if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / "test_metrics.json"
        with open(json_path, "w") as f:
            json.dump(results_dict_serializable, f, indent=2)
        logger.info(f"Saved metrics to {json_path}")
        
        # Save CSV
        csv_path = output_dir / "test_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics to {csv_path}")
        
        # Save cross-center CSV if available
        if cross_center_df is not None:
            cross_center_csv_path = output_dir / "test_cross_center_metrics.csv"
            cross_center_df.to_csv(cross_center_csv_path, index=False)
            logger.info(f"Saved cross-center metrics to {cross_center_csv_path}")
    
    return results_dict_serializable, metrics_df, cross_center_df


# ============================================================================
# Task 1.3 — Shared argument parser
# ============================================================================

def build_shared_argument_parser() -> argparse.ArgumentParser:
    """Build a shared argparse.ArgumentParser with unified evaluation CLI flags.

    Returns a parser pre-configured with:
    - --checkpoint-dir (type=Path, default=None)
    - --checkpoint-path (type=Path, default=None)
    - --checkpoint-epoch (type=int, default=None)
    - --output-dir (type=Path, default=None)
    - --output-file-name (type=str, default="evaluation_results.json")
    - --mc-dropout (action="store_true", default=False)
    - --mc-dropout-passes (type=int, default=30)

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
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Specific epoch to evaluate (centralized only; overrides best selection)",
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
    global_pool_metrics: dict[str, float],
    per_client_metrics: dict[str, tuple[int, dict[str, float]]],
    primary_metric: str,
    is_federated: bool = False,
    checkpoint_selection_mode: str | None = None,
    checkpoint_round: int | None = None,
    global_pool_model_variant: str | None = None,
) -> dict[str, Any]:
    """Build unified result payload with new schema.

    New schema:
    {
        "config": {...},
        "overall": {
            "global_pool": {...},
            "weighted_avg": {...},
            "macro_avg": {...},
        },
        "per_client": {...},
        "fairness": {...},
    }

    Federated mode adds: checkpoint_selection_mode, checkpoint_round, global_pool_model_variant to config.
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
        if global_pool_model_variant is not None:
            output_config["global_pool_model_variant"] = global_pool_model_variant

    # Build fairness
    fairness = _compute_fairness_from_per_client(per_client_metrics, primary_metric)

    payload = {
        "config": output_config,
        "overall": {
            "global_pool": to_builtin(global_pool_metrics),
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
    d. Global pool evaluation
    e. Per-client evaluation
    f. Weighted average + macro average (overall)
    g. Fairness metrics
    h. JSON output with new schema

    MC Dropout integration is deferred to PR #2 — params accepted but no-op.
    """
    from flwr_cinema.scripts.lora import load_lora_state_dict
    from models.cinema.cinema.device import get_amp_dtype_and_device
    from models.cinema.cinema.log import get_logger
    from omegaconf import OmegaConf

    orc_logger = get_logger(__name__)

    # MC Dropout warning (deferred to PR #2)
    if mc_dropout:
        orc_logger.warning(
            "MC Dropout is requested but not yet implemented. "
            "Running deterministic evaluation only. MC Dropout will be available in a future release."
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

    # ── d. Global pool evaluation ─────────────────────────────────────────
    if is_federated:
        global_pool_metrics, global_pool_variant = _run_federated_global_pool(
            model=model,
            checkpoint=checkpoint,
            config=config,
            task_type=task,
            device=device,
            base_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        )
    else:
        global_pool_metrics = _run_centralized_global_pool(
            model=model,
            config=config,
            task_type=task,
            device=device,
            amp_dtype=amp_dtype,
        )
        global_pool_variant = None

    orc_logger.info(f"Global pool metrics: {global_pool_metrics}")

    # ── e. Per-client evaluation ──────────────────────────────────────────
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

    # ── f-h. Build payload and save ───────────────────────────────────────
    primary_metric = _extract_primary_metric(config)

    payload = build_unified_result_payload(
        config={
            "checkpoint": ckpt_path.name,
            "checkpoint_epoch": checkpoint.get("epoch") if not is_federated else None,
            "use_lora": use_lora,
            "primary_metric": primary_metric,
            "task": task,
        },
        global_pool_metrics=global_pool_metrics,
        per_client_metrics=per_client_metrics,
        primary_metric=primary_metric,
        is_federated=is_federated,
        checkpoint_selection_mode=selection_meta.get("mode"),
        checkpoint_round=selection_meta.get("selected_round"),
        global_pool_model_variant=global_pool_variant,
    )

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
    checkpoint_epoch: int | None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve checkpoint for centralized evaluation."""
    if checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required for centralized evaluation.")

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
    # Load provisional config from last round to get primary metric
    last_ckpt, _ = find_last_federated_round(resolved_checkpoint_dir)
    _, provisional_config = load_checkpoint_and_config(last_ckpt, output_dir=resolved_checkpoint_dir)
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


def _run_centralized_global_pool(
    model: torch.nn.Module,
    config: dict[str, Any],
    task_type: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    """Run global pool evaluation for centralized mode."""
    from flwr_cinema.tasks.adapters import get_centralized_task_adapter
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(config)
    adapter = get_centralized_task_adapter(cfg)
    task_spec = adapter.task_spec
    eval_dataloader_fn = task_spec.eval_fn_factory()

    data_dir = Path(cfg.data.combined.dir).expanduser()
    data_dir = resolve_data_dir_with_fallback(data_dir)

    if task_type == "classification":
        target_col = cfg.data.class_column
    elif task_type == "regression":
        target_col = cfg.data.regression_column
    else:
        target_col = None

    train_meta_df = load_metadata(data_dir / "train_metadata.csv", metadata_type="train")
    test_meta_df = load_metadata(data_dir / "test_metadata.csv", metadata_type="test")

    # Apply client filtering
    test_meta_df = filter_metadata_by_client_config(config=config, meta_df=test_meta_df, metadata_type="test")

    _, val_transform = task_spec.transform_factory(cfg)

    views = [cfg.model.views] if isinstance(cfg.model.views, str) else cfg.model.views
    patch_size_dict = {
        v: cfg.data.sax.patch_size if v == "sax" else cfg.data.lax.patch_size for v in views
    }
    spacing_dict = {
        v: cfg.data.sax.spacing if v == "sax" else cfg.data.lax.spacing for v in views
    }

    overall_dataset = task_spec.dataset_factory(
        config=cfg,
        data_dir=data_dir / "test",
        meta_df=test_meta_df,
        transform=val_transform,
        train_meta_df=train_meta_df,
    )

    overall_dataloader = DataLoader(
        dataset=overall_dataset,
        sampler=SequentialSampler(overall_dataset),
        batch_size=1,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.train.n_workers,
    )

    if task_type == "classification":
        metrics, _, _ = classification_eval_dataloader_with_balanced_accuracy(
            model=model,
            dataloader=overall_dataloader,
            patch_size_dict=patch_size_dict,
            amp_dtype=amp_dtype,
            device=device,
        )
    else:
        metrics = eval_dataloader_fn(
            model=model,
            dataloader=overall_dataloader,
            patch_size_dict=patch_size_dict,
            spacing_dict=spacing_dict,
            amp_dtype=amp_dtype,
            device=device,
        )

    return normalize_metric_values(metrics)


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
        else:
            metrics = eval_dataloader_fn(
                model=model,
                dataloader=client_dataloader,
                patch_size_dict=patch_size_dict,
                spacing_dict=spacing_dict,
                amp_dtype=amp_dtype,
                device=device,
            )

        per_client_metrics[client_id] = (n_samples, normalize_metric_values(metrics))

    return per_client_metrics


def _run_federated_global_pool(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    task_type: str,
    device: torch.device,
    base_state: dict[str, torch.Tensor],
) -> tuple[dict[str, float], str]:
    """Run global pool evaluation for federated mode (reuses existing federated helpers)."""
    from flwr_cinema.federated.evaluate import (
        evaluate_global_pool,
        has_private_artifacts,
    )

    datasets_by_client, evaluate_fn = _load_federated_client_datasets(config, task_type)

    variant, metrics = evaluate_global_pool(
        model=model,
        checkpoint=checkpoint,
        evaluate_fn=evaluate_fn,
        config=config,
        task_type=task_type,
        datasets_by_client=datasets_by_client,
        base_state=base_state,
        device=device,
    )
    return metrics, variant


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
