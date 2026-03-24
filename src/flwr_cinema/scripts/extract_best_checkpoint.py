"""Extract best checkpoint metrics and compute federated parameters.

This script:
1. Finds the latest centralized training checkpoint
2. Identifies the best epoch based on validation metrics
3. Extracts optimizer steps and configuration
4. Computes equivalent federated training parameters (n_rounds, local_epochs)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(base_dir: Path = Path("outputs/regression/centralized")) -> Optional[Path]:
    """Find the most recently created checkpoint directory.
    
    Args:
        base_dir: Base directory to search (outputs/regression/centralized)
        
    Returns:
        Path to the latest checkpoint directory, or None if not found
    """
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return None
    
    # Find all checkpoint directories (contains validation_metrics.jsonl)
    checkpoint_dirs = []
    for metrics_file in base_dir.rglob("validation_metrics.jsonl"):
        checkpoint_dir = metrics_file.parent
        checkpoint_dirs.append(checkpoint_dir)
    
    if not checkpoint_dirs:
        logger.error(f"No checkpoints found in {base_dir}")
        return None
    
    # Sort by modification time and return the latest
    latest = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest checkpoint: {latest}")
    return latest


def load_validation_metrics(metrics_file: Path) -> list[dict]:
    """Load validation metrics from JSONL file.
    
    Args:
        metrics_file: Path to validation_metrics.jsonl
        
    Returns:
        List of metric dictionaries (one per epoch)
    """
    metrics = []
    with open(metrics_file) as f:
        for line in f:
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line[:100]}")
    
    logger.info(f"Loaded {len(metrics)} epochs of metrics")
    return metrics


def load_dataset_size(checkpoint_path: Path) -> Optional[int]:
    """Load dataset size from combined dataset metadata.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Total dataset size, or None if not found
    """
    # Try to find combined dataset metadata
    possible_paths = [
        Path.home() / ".cache" / "cinema_datasets" / "acdc_mnms_combined" / "processed" / "train_metadata.csv",
        Path.home() / ".cache" / "cinema_datasets" / "acdc_mnms_combined" / "processed" / "metadata.csv",
    ]
    
    import pandas as pd
    
    for metadata_path in possible_paths:
        if metadata_path.exists():
            try:
                df = pd.read_csv(metadata_path)
                dataset_size = len(df)
                logger.info(f"Loaded dataset size from {metadata_path}: {dataset_size} samples")
                return dataset_size
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
    
    # Fallback: try to find any metadata.csv in the checkpoint parent directories
    search_dir = checkpoint_path.parent
    for _ in range(3):  # Search up to 3 directories up
        for metadata_file in search_dir.rglob("metadata.csv"):
            try:
                df = pd.read_csv(metadata_file)
                dataset_size = len(df)
                logger.info(f"Found dataset size from {metadata_file}: {dataset_size} samples")
                return dataset_size
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        search_dir = search_dir.parent
    
    logger.warning("Could not find dataset metadata to determine size, using default 469")
    return None


def find_saved_epochs(checkpoint_path: Path) -> set[int]:
    """Find which epochs have saved checkpoints.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Set of epoch numbers that have saved checkpoints
    """
    saved_epochs = set()
    
    # First, look in the "checkpoints" subdirectory with "ckpt_*.pt" pattern
    checkpoints_dir = checkpoint_path / "checkpoints"
    if checkpoints_dir.exists():
        for ckpt_file in checkpoints_dir.glob("ckpt_*.pt"):
            try:
                # Extract epoch number from filename: ckpt_309.pt → 309
                epoch_str = ckpt_file.stem.replace("ckpt_", "")
                if epoch_str.isdigit():
                    saved_epochs.add(int(epoch_str))
            except (ValueError, AttributeError):
                pass
    
    # Fallback: look in checkpoint directory itself with various patterns
    if not saved_epochs:
        # Look for ckpt_*.pt in checkpoint_path
        for ckpt_file in checkpoint_path.glob("ckpt_*.pt"):
            try:
                epoch_str = ckpt_file.stem.replace("ckpt_", "")
                if epoch_str.isdigit():
                    saved_epochs.add(int(epoch_str))
            except (ValueError, AttributeError):
                pass
        
        # Look for model_epoch_*.pt pattern
        if not saved_epochs:
            for ckpt_file in checkpoint_path.glob("model_epoch_*.pt"):
                try:
                    epoch_str = ckpt_file.stem.replace("model_epoch_", "")
                    if epoch_str.isdigit():
                        saved_epochs.add(int(epoch_str))
                except (ValueError, AttributeError):
                    pass
        
        # Look for checkpoint*.pt pattern
        if not saved_epochs:
            for ckpt_file in checkpoint_path.glob("checkpoint*.pt"):
                saved_epochs.add(-1)  # Unknown epoch, will filter differently
    
    if saved_epochs:
        logger.info(f"Found saved checkpoints for epochs: {sorted(saved_epochs)}")
    else:
        logger.warning("Could not find saved checkpoint files, will use best epoch from metrics")
    
    return saved_epochs


def find_best_saved_epoch(metrics: list[dict], saved_epochs: set[int], metric_key: str = "fedavg_metrics") -> tuple[int, dict]:
    """Find best epoch among those that have saved checkpoints.
    
    Args:
        metrics: List of metric dictionaries
        saved_epochs: Set of epochs that have saved checkpoints
        metric_key: Key for aggregated metrics ("fedavg_metrics" or "macro_metrics")
        
    Returns:
        Tuple of (best_epoch, best_metrics)
    """
    # If no saved epochs found, fall back to best epoch overall
    if not saved_epochs or -1 in saved_epochs:
        logger.warning("Could not determine saved epochs, using best epoch from metrics")
        return find_best_epoch(metrics, metric_key)
    
    best_epoch = -1
    best_mae = float('inf')
    best_metrics = None
    
    for metrics_entry in metrics:
        epoch = metrics_entry.get("epoch", -1)
        
        # Only consider epochs that have saved checkpoints
        if epoch not in saved_epochs:
            continue
        
        agg_metrics = metrics_entry.get(metric_key, {})
        mae = agg_metrics.get("mae", float('inf'))
        
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            best_metrics = metrics_entry
    
    if best_epoch < 0:
        logger.error("No metrics found for saved checkpoint epochs")
        return find_best_epoch(metrics, metric_key)
    
    logger.info(f"Best saved checkpoint: epoch {best_epoch} (MAE: {best_mae:.6f})")
    return best_epoch, best_metrics


def find_best_epoch(metrics: list[dict], metric_key: str = "fedavg_metrics") -> tuple[int, dict]:
    """Find best epoch based on MAE metric.
    
    Args:
        metrics: List of metric dictionaries
        metric_key: Key for aggregated metrics ("fedavg_metrics" or "macro_metrics")
        
    Returns:
        Tuple of (best_epoch, best_metrics)
    """
    best_epoch = -1
    best_mae = float('inf')
    best_metrics = None
    
    for metrics_entry in metrics:
        epoch = metrics_entry.get("epoch", -1)
        agg_metrics = metrics_entry.get(metric_key, {})
        mae = agg_metrics.get("mae", float('inf'))
        
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            best_metrics = metrics_entry
    
    logger.info(f"Best epoch: {best_epoch} (MAE: {best_mae:.6f})")
    return best_epoch, best_metrics


def extract_federated_params(best_metrics: dict, dataset_name: str = "acdc_mnms") -> dict:
    """Extract parameters for federated training.
    
    Args:
        best_metrics: Best epoch metrics dictionary
        dataset_name: Name of the dataset (for output)
        
    Returns:
        Dictionary with recommended federated parameters
    """
    params = {}
    
    # Extract step information
    optimizer_steps = best_metrics.get("optimizer_steps", 0)
    optimizer_steps_per_epoch = best_metrics.get("optimizer_steps_per_epoch", 1)
    forward_passes_per_epoch = best_metrics.get("forward_passes_per_epoch", 1)
    n_accum_steps = best_metrics.get("n_accum_steps", 1)
    epoch = best_metrics.get("epoch", 0)
    
    logger.info(f"\nExtracted from best epoch {epoch}:")
    logger.info(f"  Total optimizer steps: {optimizer_steps}")
    logger.info(f"  Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    logger.info(f"  Forward passes per epoch: {forward_passes_per_epoch}")
    logger.info(f"  Gradient accumulation steps: {n_accum_steps}")
    
    # Store extracted parameters
    params["best_epoch"] = epoch
    params["centralized_optimizer_steps"] = optimizer_steps
    params["optimizer_steps_per_epoch"] = optimizer_steps_per_epoch
    params["forward_passes_per_epoch"] = forward_passes_per_epoch
    params["n_accum_steps"] = n_accum_steps
    
    # Store validation metrics
    params["best_fedavg_metrics"] = best_metrics.get("fedavg_metrics", {})
    params["best_macro_metrics"] = best_metrics.get("macro_metrics", {})
    
    return params


def compute_federated_parameters(
    centralized_optimizer_steps: int,
    n_clients: int = 5,
    batch_size_per_device: int = 32,
    batch_size: int = 64,
    dataset_size: int = 469,  # acdc_mnms_combined training set size
) -> dict:
    """Compute federated parameters matching centralized training.
    
    Args:
        centralized_optimizer_steps: Total optimizer steps in centralized training
        n_clients: Number of federated clients
        batch_size_per_device: Batch size per client
        batch_size: Effective batch size (after gradient accumulation)
        dataset_size: Total dataset size (for per-client computation). 
                     Loaded from metadata; defaults to acdc_mnms_combined: 469
        
    Returns:
        Dictionary with recommended federated parameters
    """
    # Assume even distribution across clients (worst case)
    samples_per_client = dataset_size // n_clients
    
    # Forward passes per epoch per client
    forward_passes_per_client = (samples_per_client + batch_size_per_device - 1) // batch_size_per_device
    
    # Compute n_accum_steps for each client
    # In federated: each client trains independently with world_size=1
    # n_accum_steps = batch_size / batch_size_per_device
    n_accum_steps = batch_size // batch_size_per_device
    logger.info(f"Federated gradient accumulation: batch_size={batch_size}, batch_size_per_device={batch_size_per_device}, n_accum_steps={n_accum_steps}")
    
    # Optimizer steps per epoch per client (after gradient accumulation)
    optimizer_steps_per_epoch_per_client = (forward_passes_per_client + n_accum_steps - 1) // n_accum_steps
    
    logger.info(f"Per-client breakdown:")
    logger.info(f"  samples_per_client: {samples_per_client}")
    logger.info(f"  forward_passes_per_client: {forward_passes_per_client}")
    logger.info(f"  optimizer_steps_per_epoch_per_client: {optimizer_steps_per_epoch_per_client} (with n_accum_steps={n_accum_steps})")
    
    # Optimizer steps per round (all clients do local_epochs)
    # For fair comparison: optimizer_steps_per_round = n_clients * local_epochs * optimizer_steps_per_epoch_per_client
    # This formula ensures we step through approximately the same number of optimizer updates
    # as the centralized training, but distributed across federated rounds
    
    # Try different local_epochs values and compute corresponding n_rounds
    recommended_configs = []
    
    logger.info(f"Computing n_rounds for different local_epochs (targeting ≈{centralized_optimizer_steps} total steps):")
    for local_epochs in [1, 2, 3, 4, 5, 10, 15, 20]:
        optimizer_steps_per_round = n_clients * local_epochs * optimizer_steps_per_epoch_per_client
        n_rounds = (centralized_optimizer_steps + optimizer_steps_per_round - 1) // optimizer_steps_per_round
        total_optimizer_steps = n_rounds * optimizer_steps_per_round
        difference = abs(total_optimizer_steps - centralized_optimizer_steps)
        
        logger.info(f"  local_epochs={local_epochs:2d} → optimizer_steps_per_round={optimizer_steps_per_round:4d}, n_rounds={n_rounds:3d}, total_steps={total_optimizer_steps:5d}, diff={difference:4d}")
        
        recommended_configs.append({
            "local_epochs": local_epochs,
            "n_rounds": n_rounds,
            "optimizer_steps_per_round": optimizer_steps_per_round,
            "total_optimizer_steps": total_optimizer_steps,
            "difference": difference,
        })
    
    # Sort by difference (closest to centralized)
    recommended_configs.sort(key=lambda x: x["difference"])
    
    return {
        "samples_per_client": samples_per_client,
        "forward_passes_per_client": forward_passes_per_client,
        "optimizer_steps_per_epoch_per_client": optimizer_steps_per_epoch_per_client,
        "recommended_configs": recommended_configs,
    }


def print_config_recommendations(
    checkpoint_path: Path,
    extracted_params: dict,
    federated_params: dict,
) -> None:
    """Print recommended configuration parameters.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        extracted_params: Extracted parameters from centralized training
        federated_params: Computed federated parameters
    """
    print("\n" + "=" * 80)
    print("CHECKPOINT ANALYSIS: CENTRALIZED → FEDERATED PARAMETER EXTRACTION")
    print("=" * 80)
    
    print(f"\n📁 Checkpoint Directory:")
    print(f"   {checkpoint_path}")
    
    print(f"\n✅ Best Epoch: {extracted_params['best_epoch']}")
    print(f"   Total optimizer steps: {extracted_params['centralized_optimizer_steps']}")
    
    print(f"\n📊 Validation Metrics (Best Epoch):")
    for key, value in extracted_params.get("best_fedavg_metrics", {}).items():
        if isinstance(value, float):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n⚙️  Centralized Training Configuration:")
    print(f"   Optimizer steps per epoch: {extracted_params['optimizer_steps_per_epoch']}")
    print(f"   Forward passes per epoch: {extracted_params['forward_passes_per_epoch']}")
    print(f"   Gradient accumulation steps: {extracted_params['n_accum_steps']}")
    
    print(f"\n🌐 Federated Training Options:")
    print(f"   (Choose one based on computational budget)")
    print(f"\n   {'local_epochs':<15} {'n_rounds':<15} {'steps/round':<15} {'total_steps':<15} {'diff':<15}")
    print(f"   {'-'*75}")
    
    for config in federated_params["recommended_configs"][:10]:  # Show top 10
        local_e = config["local_epochs"]
        n_r = config["n_rounds"]
        steps_r = config["optimizer_steps_per_round"]
        total_s = config["total_optimizer_steps"]
        diff = config["difference"]
        marker = " ← RECOMMENDED" if config == federated_params["recommended_configs"][0] else ""
        print(f"   {local_e:<15} {n_r:<15} {steps_r:<15} {total_s:<15} {diff:<15}{marker}")
    
    best_config = federated_params["recommended_configs"][0]
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDED FEDERATED CONFIG UPDATE")
    print("=" * 80)
    
    print(f"\nUpdate your YAML config file with:")
    print(f"\nfederated:")
    print(f"  server_host: 'localhost'")
    print(f"  server_port: 8888")
    print(f"  n_rounds: {best_config['n_rounds']}")
    print(f"  local_epochs: {best_config['local_epochs']}")
    
    print(f"\nThis will:")
    print(f"  • Run {best_config['n_rounds']} communication rounds")
    print(f"  • Each client trains for {best_config['local_epochs']} local epochs per round")
    print(f"  • Total optimizer steps: {best_config['total_optimizer_steps']} (diff: {best_config['difference']})")
    print(f"  • Step info: {federated_params['optimizer_steps_per_epoch_per_client']} steps/epoch per client")
    
    print(f"\n" + "=" * 80)


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract best checkpoint metrics and compute federated parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest checkpoint and recommend federated parameters
  python scripts/extract_best_checkpoint.py
  
  # Specify custom checkpoint directory
  python scripts/extract_best_checkpoint.py --checkpoint_dir outputs/regression/centralized/acdc_mnms/2026-02-03/13-45-18_lv_ef_seed0
  
  # Specify number of federated clients
  python scripts/extract_best_checkpoint.py --n_clients 3
        """
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Path to specific checkpoint directory. If not provided, uses latest."
    )
    parser.add_argument(
        "--n_clients",
        type=int,
        default=5,
        help="Number of federated clients (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Effective batch size after gradient accumulation (default: 64)"
    )
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=32,
        help="Batch size per client device (default: 32)"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=None,
        help="Total dataset size. If not provided, will load from metadata (default: auto-detect)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="fedavg_metrics",
        choices=["fedavg_metrics", "macro_metrics"],
        help="Metric to use for best epoch selection (default: fedavg_metrics)"
    )
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir)
    else:
        checkpoint_path = find_latest_checkpoint()
    
    if not checkpoint_path:
        logger.error("Could not find checkpoint directory")
        sys.exit(1)
    
    # Load metrics
    metrics_file = checkpoint_path / "validation_metrics.jsonl"
    if not metrics_file.exists():
        logger.error(f"Metrics file not found: {metrics_file}")
        sys.exit(1)
    
    metrics = load_validation_metrics(metrics_file)
    if not metrics:
        logger.error("No metrics loaded from file")
        sys.exit(1)
    
    # Find which epochs have saved checkpoints
    saved_epochs = find_saved_epochs(checkpoint_path)
    
    # Load dataset size from metadata if not provided
    if args.dataset_size is None:
        loaded_size = load_dataset_size(checkpoint_path)
        args.dataset_size = loaded_size if loaded_size else 469
        logger.info(f"Using dataset size: {args.dataset_size}")
    
    # Find best epoch among saved checkpoints
    best_epoch, best_metrics = find_best_saved_epoch(metrics, saved_epochs, metric_key=args.metric)
    if best_epoch < 0:
        logger.error("Could not find best epoch")
        sys.exit(1)
    
    # Extract parameters
    extracted_params = extract_federated_params(best_metrics)
    
    # Compute federated parameters
    federated_params = compute_federated_parameters(
        centralized_optimizer_steps=extracted_params["centralized_optimizer_steps"],
        n_clients=args.n_clients,
        batch_size=args.batch_size,
        batch_size_per_device=args.batch_size_per_device,
        dataset_size=args.dataset_size,
    )
    
    # Print recommendations
    print_config_recommendations(checkpoint_path, extracted_params, federated_params)
    
    # Save results to file
    output_file = checkpoint_path / "federated_params_recommendation.json"
    results = {
        "checkpoint_path": str(checkpoint_path),
        "timestamp": datetime.now().isoformat(),
        "extracted_params": extracted_params,
        "federated_params": {
            "samples_per_client": federated_params["samples_per_client"],
            "forward_passes_per_client": federated_params["forward_passes_per_client"],
            "optimizer_steps_per_epoch_per_client": federated_params["optimizer_steps_per_epoch_per_client"],
            "recommended_configs": federated_params["recommended_configs"],
        },
        "command_to_run": f"./run_federated_monitored.sh configs/cinema_fed.yaml {federated_params['recommended_configs'][0]['n_rounds']} {args.n_clients}",
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
