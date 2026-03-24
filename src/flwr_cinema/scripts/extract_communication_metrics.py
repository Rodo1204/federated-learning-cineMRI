#!/usr/bin/env python3
"""
Extract communication budget and validation metrics from federated experiment runs.

This script processes multiple federated learning experiment runs and extracts:
- Per-round accumulated communication budget (total optimizer steps)
- Per-round validation metric (balanced accuracy or accuracy)
- Final test evaluation results (if available)

Results are saved as CSV and JSON in outputs/ablation_study_lora/

Usage:
    python scripts/extract_communication_metrics.py \\
        outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_False/seed0/2026-02-20/16-03-59/ \\
        outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/16-05-30/
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import csv
from datetime import datetime
import yaml


def load_config(exp_path: Path) -> Dict[str, Any]:
    """
    Load experiment configuration from config.yaml.
    
    Args:
        exp_path: Path to the experiment directory
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config.yaml not found
    """
    config_file = exp_path / "config.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def extract_seed_from_id(exp_id: str) -> int | None:
    """
    Extract seed number from experiment ID.
    
    Expected format: ...seed{number}...
    
    Args:
        exp_id: Experiment identifier string
    
    Returns:
        Seed number as integer, or None if seed not found
    """
    import re
    match = re.search(r'seed(\d+)', exp_id)
    if match:
        return int(match.group(1))
    return None


def extract_seed_from_config(config: Dict[str, Any]) -> int | None:
    """
    Extract seed number from experiment configuration file.
    
    Tries multiple common config key paths:
    - train.seed
    - train.random_seed
    - seed
    - random_seed
    
    Args:
        config: Configuration dictionary from config.yaml
    
    Returns:
        Seed number as integer, or None if seed not found
    """
    # Try nested keys in train section
    train_config = config.get("train", {})
    if isinstance(train_config, dict):
        seed = train_config.get("seed")
        if seed is not None:
            return int(seed) if isinstance(seed, (int, str)) else None
        
        seed = train_config.get("random_seed")
        if seed is not None:
            return int(seed) if isinstance(seed, (int, str)) else None
    
    # Try top-level keys
    seed = config.get("seed")
    if seed is not None:
        return int(seed) if isinstance(seed, (int, str)) else None
    
    seed = config.get("random_seed")
    if seed is not None:
        return int(seed) if isinstance(seed, (int, str)) else None
    
    return None


def extract_lora_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract LoRA configuration parameters from the experiment config.
    
    Full-finetuning is inferred when:
    - Backbone is NOT frozen (freeze_pretrained == False)
    - LoRA is disabled (use_lora == False)
    - Dual LoRA is disabled (dual_lora_enabled == False)
    
    Head-only is inferred when:
    - Backbone IS frozen (freeze_pretrained == True)
    - LoRA is disabled (use_lora == False)
    - Dual LoRA is disabled (dual_lora_enabled == False)
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary with keys: freeze_backbone, use_lora, blocks_to_lora, dual_lora_enabled, is_full_finetuning, is_head_only, enc_depth
    """
    freeze_backbone = config.get("model", {}).get("freeze_pretrained", True)
    use_lora = config.get("train", {}).get("use_lora", False)
    dual_lora_enabled = config.get("dual_lora", {}).get("enabled", False)
    
    # Infer full-finetuning: backbone not frozen + no LoRA + no dual LoRA
    is_full_finetuning = (not freeze_backbone) and (not use_lora) and (not dual_lora_enabled)
    
    # Infer head-only: backbone frozen + no LoRA + no dual LoRA
    is_head_only = freeze_backbone and (not use_lora) and (not dual_lora_enabled)
    
    # Extract encoder depth (number of transformer blocks) from model config
    # Maps model size to encoder depth: ViT-B=12, ViT-L=24, ViT-H=32
    enc_depth_by_size = {
        "tiny": 12,   # ViT-Ti (if used)
        "base": 12,   # ViT-B (default)
        "large": 24,  # ViT-L
        "huge": 32,   # ViT-H
    }
    model_size = config.get("model", {}).get("convvit", {}).get("size", "base")
    enc_depth = enc_depth_by_size.get(model_size, 12)  # Default to ViT-B (12) if size not recognized
    
    return {
        "freeze_backbone": freeze_backbone,
        "use_lora": use_lora,
        "blocks_to_lora": config.get("lora", {}).get("blocks_to_lora", None),
        "dual_lora_enabled": dual_lora_enabled,
        "is_full_finetuning": is_full_finetuning,
        "is_head_only": is_head_only,
        "enc_depth": enc_depth,  # Number of transformer blocks for LoRA estimation
    }


def load_aggregated_metrics(exp_path: Path) -> List[Dict[str, Any]]:
    """
    Load aggregated metrics history from a federated experiment run.
    
    Args:
        exp_path: Path to the experiment directory (root with federated_server/ subdirectory)
    
    Returns:
        List of per-round metrics dictionaries
    
    Raises:
        FileNotFoundError: If aggregated_metrics_history.json not found
        json.JSONDecodeError: If file is not valid JSON
    """
    metrics_file = exp_path / "federated_server" / "aggregated_metrics_history.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_file}\n"
            f"Expected federated experiment structure with federated_server/aggregated_metrics_history.json"
        )
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not isinstance(metrics, list):
        raise ValueError(f"Expected metrics file to contain a JSON array, got {type(metrics)}")
    
    return metrics


def load_test_evaluation(exp_path: Path) -> Dict[str, Any]:
    """
    Load test evaluation results from a completed experiment run.
    
    Args:
        exp_path: Path to the experiment directory
    
    Returns:
        Test metrics dictionary or empty dict if not found
    """
    test_metrics_file = exp_path / "test_evaluation" / "test_metrics.json"
    
    if not test_metrics_file.exists():
        return {}
    
    try:
        with open(test_metrics_file, 'r') as f:
            test_metrics = json.load(f)
        return test_metrics
    except (json.JSONDecodeError, IOError) as e:
        print(f"    Warning: Could not load test_metrics.json: {e}")
        return {}


def extract_bytes_per_round(exp_path: Path) -> Dict[int, Dict[str, int]]:
    """
    Extract bytes transmitted for model updates per communication round from server logs.
    
    Only counts bytes that are for model parameters (model_params in Keys), excluding
    status messages and metadata.
    
    Args:
        exp_path: Path to the experiment directory
    
    Returns:
        Dictionary mapping round number to {'bytes_sent': X, 'bytes_received': Y}
    """
    server_log = exp_path / "federated_server" / "server.log"
    
    if not server_log.exists():
        return {}
    
    bytes_per_round = {}
    current_round = None
    expecting_next_model_update = False
    
    try:
        with open(server_log, 'r') as f:
            for line in f:
                import re
                
                # Parse lines like: "[SERVER] Sending model to client acdc (round 0)"
                if "[SERVER] Sending model to client" in line and "(round " in line:
                    match = re.search(r'\(round (\d+)\)', line)
                    if match:
                        current_round = int(match.group(1))
                        if current_round not in bytes_per_round:
                            bytes_per_round[current_round] = {'bytes_sent': 0, 'bytes_received': 0}
                        expecting_next_model_update = False
                
                # Parse lines like: "[SERVER] Sending 139402 bytes to client. Keys: ['round', 'model_params', 'output_dir']"
                # Only count if 'model_params' is in the message (model update, not status)
                if current_round is not None and "[SERVER] Sending" in line and "bytes to client" in line:
                    if "model_params" in line:  # Only count model parameter updates
                        match = re.search(r'Sending (\d+) bytes to client', line)
                        if match:
                            bytes_sent = int(match.group(1))
                            bytes_per_round[current_round]['bytes_sent'] += bytes_sent
                            expecting_next_model_update = True  # Next large "Expecting" is the response
                
                # Parse lines like: "[SERVER] Expecting 139483 bytes from client."
                # Count the received bytes for model updates (typically large values corresponding to sent models)
                if current_round is not None and "[SERVER] Expecting" in line and "bytes from client" in line:
                    match = re.search(r'Expecting (\d+) bytes from client', line)
                    if match:
                        bytes_received = int(match.group(1))
                        # Only count if we're expecting a model update response (bytes > 1000 typically)
                        if bytes_received > 1000:
                            bytes_per_round[current_round]['bytes_received'] += bytes_received
    
    except IOError as e:
        print(f"    Warning: Could not read server log: {e}")
        return {}
    
    return bytes_per_round


def extract_metrics_per_round(
    metrics_history: List[Dict[str, Any]],
    exp_id: str,
    lora_config: Dict[str, Any],
    bytes_per_round: Dict[int, Dict[str, int]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract communication budget and validation metrics from metrics history.
    
    Args:
        metrics_history: List of per-round metric dictionaries from aggregated_metrics_history.json
        exp_id: Identifier for the experiment (for tracking)
        lora_config: LoRA configuration dictionary
        bytes_per_round: Optional dictionary mapping round number to bytes transmitted/received
    
    Returns:
        Tuple of:
        - List of round data dicts with keys: round, total_optimizer_steps, metric values, bytes_sent, bytes_received
        - Summary dict with experiment metadata
    """
    if bytes_per_round is None:
        bytes_per_round = {}
    
    # Estimate initialization cost if not available in logs
    estimated_init_bytes = estimate_first_round_bytes(lora_config)
    actual_init_bytes_sent = None  # Will be set from first round data
    actual_init_bytes_received = None
    
    rounds_data = []
    
    for round_idx, round_data in enumerate(metrics_history):
        round_num = round_data.get("round", "unknown")
        total_steps = round_data.get("total_optimizer_steps", 0)
        
        # Try to extract balanced accuracy; fallback to accuracy
        fedavg_metrics = round_data.get("fedavg_metrics", {})
        
        f1_score = fedavg_metrics.get("f1", None)
        specificty = fedavg_metrics.get("specificity", None)
        sensitivity = fedavg_metrics.get("sensitivity", None)
        roc_auc = fedavg_metrics.get("roc_auc", None)
        balanced_accuracy = (specificty + sensitivity)/2
        
        # Extract bytes transmitted for this round
        bytes_info = bytes_per_round.get(round_num, {})
        bytes_sent = bytes_info.get('bytes_sent', None)
        bytes_received = bytes_info.get('bytes_received', None)
        
        # For first round: add initialization cost if not already in logs
        if round_idx == 0 and (bytes_sent is None or bytes_sent == 0):
            bytes_sent = estimated_init_bytes
            bytes_received = estimated_init_bytes
        # For subsequent rounds: use logs if available, otherwise add estimated init to first round
        elif round_idx == 0 and bytes_sent is not None:
            # Bytes already in logs, no need to add estimate
            pass
        elif round_idx > 0 and (bytes_sent is None or bytes_sent == 0):
            # Non-first round with missing logs - use estimate from this round if available
            bytes_sent = bytes_info.get('bytes_sent', 0) or 0
            bytes_received = bytes_info.get('bytes_received', 0) or 0
        
        # Track actual initialization bytes from first round
        if round_idx == 0:
            actual_init_bytes_sent = bytes_sent
            actual_init_bytes_received = bytes_received
        
        rounds_data.append({
            "experiment": exp_id,
            "round": round_num,
            "total_optimizer_steps": total_steps,
            "balanced_accuracy": balanced_accuracy,
            "f1_score": f1_score,
            "specificity": specificty,
            "sensitivity": sensitivity,
            "roc_auc": roc_auc,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
        })
    
    # Generate summary
    # Compute balanced accuracy for final round
    final_specificity = metrics_history[-1].get("fedavg_metrics", {}).get("specificity", None) if metrics_history else None
    final_sensitivity = metrics_history[-1].get("fedavg_metrics", {}).get("sensitivity", None) if metrics_history else None
    final_balanced_accuracy = (final_specificity + final_sensitivity) / 2 if (final_specificity is not None and final_sensitivity is not None) else None
    
    # IMPORTANT: Only count bytes up to the last validation checkpoint, not all communication rounds
    # The server may have continued communicating after the last validation checkpoint, but we should
    # only report the communication cost up to where we have actual validation metrics
    max_validation_round = 0
    if metrics_history:
        for round_data in metrics_history:
            round_num = round_data.get("round", 0)
            if isinstance(round_num, int) and round_num > max_validation_round:
                max_validation_round = round_num
    
    # Sum only bytes for rounds 0 through max_validation_round
    total_bytes_sent = 0
    total_bytes_received = 0
    for round_num in range(max_validation_round + 1):
        if round_num in bytes_per_round:
            bytes_info = bytes_per_round[round_num]
            total_bytes_sent += bytes_info.get('bytes_sent', 0) or 0
            total_bytes_received += bytes_info.get('bytes_received', 0) or 0
    
    summary = {
        "experiment_id": exp_id,
        "n_rounds": len(metrics_history),
        "final_balanced_accuracy": final_balanced_accuracy,
        "final_accuracy": metrics_history[-1].get("fedavg_metrics", {}).get("accuracy", None) if metrics_history else None,
        "final_f1_score": metrics_history[-1].get("fedavg_metrics", {}).get("f1", None) if metrics_history else None,
        "final_specificity": final_specificity,
        "final_sensitivity": final_sensitivity,
        "final_roc_auc": metrics_history[-1].get("fedavg_metrics", {}).get("roc_auc", None) if metrics_history else None,
        "final_total_steps": metrics_history[-1].get("total_optimizer_steps", 0) if metrics_history else 0,
        "validation_rounds_count": sum(1 for m in metrics_history if m.get("validation_round", False)),
        "bytes_sent_initialization": actual_init_bytes_sent or estimated_init_bytes,
        "bytes_received_initialization": actual_init_bytes_received or estimated_init_bytes,
        "total_bytes_sent": total_bytes_sent,
        "total_bytes_received": total_bytes_received,
        # Training configuration
        "is_full_finetuning": lora_config.get("is_full_finetuning"),
        "is_head_only": lora_config.get("is_head_only"),
        "freeze_backbone": lora_config.get("freeze_backbone"),
        "use_lora": lora_config.get("use_lora"),
        "blocks_to_lora": lora_config.get("blocks_to_lora"),
        "dual_lora_enabled": lora_config.get("dual_lora_enabled"),
    }
    
    return rounds_data, summary


def estimate_first_round_bytes(lora_config: Dict[str, Any]) -> float:
    """
    Estimate first-round communication cost based on trainable configuration.
    
    Inference logic:
    - Head-only: ~0.066 MB (classification head parameters)
    - LoRA N blocks: ~0.33 MB per block (LoRA projections)
      - When blocks_to_lora=null: applies to all transformer blocks
      - When blocks_to_lora=[...]: applies only to specified blocks
      - Number of blocks depends on model size: ViT-B=12, ViT-L=24, ViT-H=32
    - Dual-LoRA N blocks: ~0.33 MB per block per encoder (image + tabular)
    
    Args:
        lora_config: Configuration with keys: freeze_backbone, use_lora, blocks_to_lora, dual_lora_enabled, enc_depth (optional)
    
    Returns:
        Estimated bytes for first round
    """
    # Based on empirical measurements from the data
    HEAD_ONLY_BYTES = 0.066e6  # bytes (0.066 MB)
    BYTES_PER_LORA_BLOCK = 0.33e6  # bytes (~0.33 MB per LoRA layer)
    
    freeze_backbone = lora_config.get("freeze_backbone", True)
    use_lora = lora_config.get("use_lora", False)
    blocks_to_lora = lora_config.get("blocks_to_lora")  # Don't default to [], keep None
    dual_lora_enabled = lora_config.get("dual_lora_enabled", False)
    
    # Get total number of transformer blocks from model configuration
    # This is read from the actual ViT encoder depth in the config
    total_vit_blocks = lora_config.get("enc_depth", 12)  # Default to ViT-B (12 blocks) if not specified
    
    # Head-only: frozen backbone + no LoRA
    if freeze_backbone and not use_lora and not dual_lora_enabled:
        return HEAD_ONLY_BYTES
    
    # LoRA or Dual-LoRA
    if use_lora:
        # Determine number of blocks
        if blocks_to_lora is None:
            # blocks_to_lora=null means apply to ALL transformer blocks
            n_blocks = total_vit_blocks
        elif isinstance(blocks_to_lora, list) and len(blocks_to_lora) > 0:
            # Specific blocks specified
            n_blocks = len(blocks_to_lora)
        else:
            # Empty list or unexpected format - treat as no LoRA blocks (shouldn't happen)
            # Fall through to default handling below
            n_blocks = None
        
        if n_blocks is not None:
            # Dual-LoRA: both image and tabular encoders
            if dual_lora_enabled:
                return n_blocks * BYTES_PER_LORA_BLOCK * 2
            else:
                # Standard LoRA
                return n_blocks * BYTES_PER_LORA_BLOCK
    
    # Default fallback (full finetuning or unknown)
    # Estimate based on whether backbone is trainable
    if not freeze_backbone:
        # Full model trainable - larger estimate
        return BYTES_PER_LORA_BLOCK * 10  # Conservative estimate for full model
    
    return HEAD_ONLY_BYTES


def analyze_first_round_communication(
    all_rounds: List[Dict[str, Any]],
    experiments_summary: List[Dict[str, Any]] = None,
    exp_name_map: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Analyze first-round (initialization) communication costs to understand model size differences.
    
    This breakdown shows:
    - Initial communication (first round): Model broadcast from server to clients
    - Per-round update cost: Average bytes after first round (model updates)
    - Total cost: All communication accumulated
    
    If bytes data is not in logs, it's inferred from model configuration.
    
    Args:
        all_rounds: List of all round data dictionaries
        experiments_summary: Optional list of experiment summaries with config info
        exp_name_map: Optional dictionary mapping experiment IDs to simplified names
    
    Returns:
        Dictionary with first-round communication analysis
    """
    if exp_name_map is None:
        exp_name_map = {}
    
    # Create a map of experiment configs for inference fallback
    exp_config_map = {}
    if experiments_summary:
        for exp_summary in experiments_summary:
            exp_id = exp_summary.get("experiment_id")
            if exp_id:
                exp_config_map[exp_id] = {
                    "freeze_backbone": exp_summary.get("freeze_backbone"),
                    "use_lora": exp_summary.get("use_lora"),
                    "blocks_to_lora": exp_summary.get("blocks_to_lora"),
                    "dual_lora_enabled": exp_summary.get("dual_lora_enabled"),
                }
    
    # Group rounds by experiment
    experiments = {}
    for round_data in all_rounds:
        exp_id = round_data.get("experiment")
        if exp_id not in experiments:
            experiments[exp_id] = []
        experiments[exp_id].append(round_data)
    
    # Analyze first round for each experiment
    first_round_analysis = {}
    
    for exp_id, rounds in experiments.items():
        if not rounds:
            continue
        
        # Sort by round number or optimizer steps
        rounds_sorted = sorted(rounds, key=lambda r: r.get("round", 0))
        
        # First round communication
        first_round = rounds_sorted[0]
        first_round_bytes_sent = first_round.get("bytes_sent", None)
        first_round_bytes_received = first_round.get("bytes_received", None)
        bytes_from_logs = first_round_bytes_sent is not None and first_round_bytes_received is not None
        
        # Fallback: infer from configuration if not in logs
        if not bytes_from_logs and exp_id in exp_config_map:
            estimated_bytes = estimate_first_round_bytes(exp_config_map[exp_id])
            first_round_bytes_sent = estimated_bytes
            first_round_bytes_received = estimated_bytes
        
        # Ensure we have numeric values
        first_round_bytes_sent = first_round_bytes_sent or 0
        first_round_bytes_received = first_round_bytes_received or 0
        
        # Calculate average update cost (excluding first round if we have multiple rounds)
        update_rounds = rounds_sorted[1:] if len(rounds_sorted) > 1 else []
        avg_update_sent = sum(r.get("bytes_sent", 0) or 0 for r in update_rounds) / len(update_rounds) if update_rounds else first_round_bytes_sent
        
        # Total accumulated
        total_sent = sum(r.get("bytes_sent", 0) or 0 for r in rounds_sorted)
        total_received = sum(r.get("bytes_received", 0) or 0 for r in rounds_sorted)
        
        exp_name = exp_name_map.get(exp_id, exp_id)
        
        first_round_analysis[exp_id] = {
            "exp_name": exp_name,
            "initial_bytes_sent_mb": first_round_bytes_sent / (1024 * 1024),
            "initial_bytes_received_mb": first_round_bytes_received / (1024 * 1024),
            "avg_update_bytes_sent_mb": avg_update_sent / (1024 * 1024) if avg_update_sent else 0,
            "bytes_source": "inferred_from_config" if not bytes_from_logs else "extracted_from_logs",
            "n_rounds": len(rounds_sorted),
            "total_bytes_sent_mb": total_sent / (1024 * 1024),
            "total_bytes_received_mb": total_received / (1024 * 1024),
            "bytes_from_updates_only_mb": (total_sent - first_round_bytes_sent) / (1024 * 1024),
        }
    
    return first_round_analysis


def _merge_test_metrics_into_summary(summary: Dict[str, Any], test_eval: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge test evaluation metrics into experiment summary.
    
    Adds test metrics with a 'test_' prefix to distinguish from validation metrics.
    
    Args:
        summary: Experiment summary dictionary from extract_metrics_per_round
        test_eval: Test evaluation metrics dictionary from load_test_evaluation
    
    Returns:
        Updated summary dictionary with test metrics included
    """
    # Add test metrics with 'test_' prefix
    if test_eval:
        for key, value in test_eval.items():
            summary[f"test_{key}"] = value
        
        # Also add a flag indicating that test evaluation was available
        summary["has_test_eval"] = True
    else:
        summary["has_test_eval"] = False
    
    return summary



def process_experiments(exp_paths: List[str], output_dir: Path) -> Dict[str, Any]:
    """
    Process multiple federated experiment runs.
    
    Args:
        exp_paths: List of string paths to experiment directories
        output_dir: Output directory for results
    
    Returns:
        Dictionary with processing results and statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_rounds = []
    experiments_summary = []
    test_results = []
    errors = []
    
    for exp_path_str in exp_paths:
        exp_path = Path(exp_path_str).resolve()
        
        # Create experiment ID from path (last few components)
        path_parts = exp_path.parts
        # Try to construct a meaningful ID from the path
        exp_id = "_".join(path_parts[-5:]) if len(path_parts) >= 5 else exp_path.name
        
        try:
            print(f"Processing: {exp_path_str}")
            print(f"  Experiment ID: {exp_id}")
            
            # Load configuration and extract LoRA settings
            config = load_config(exp_path)
            lora_config = extract_lora_config(config)
            
            # Load and extract metrics
            metrics_history = load_aggregated_metrics(exp_path)
            bytes_per_round = extract_bytes_per_round(exp_path)
            rounds_data, summary = extract_metrics_per_round(metrics_history, exp_id, lora_config, bytes_per_round)
            
            all_rounds.extend(rounds_data)
            
            # Load test evaluation results if available
            test_eval = load_test_evaluation(exp_path)
            if test_eval:
                # Extract seed from ID first, fallback to config file
                seed = extract_seed_from_id(exp_id)
                if seed is None:
                    seed = extract_seed_from_config(config)
                
                test_result_entry = {
                    "experiment_id": exp_id,
                    "seed": seed,
                    "path": str(exp_path),
                    "test_metrics": test_eval,
                }
                test_results.append(test_result_entry)
                print(f"  ✓ Loaded test evaluation results")
                
                # Merge test metrics into summary
                summary = _merge_test_metrics_into_summary(summary, test_eval)
            
            experiments_summary.append({
                **summary,
                "path": str(exp_path),
                "config_path": str(exp_path / "config.yaml"),
            })
            
            print(f"  ✓ Extracted {len(rounds_data)} rounds")
            print(f"    LoRA Config: use_lora={lora_config['use_lora']}, blocks_to_lora={lora_config['blocks_to_lora']}, dual_lora={lora_config['dual_lora_enabled']}")
            print(f"    Final communication budget: {summary['final_total_steps']} optimizer steps")
            
        except Exception as e:
            error_msg = f"Error processing {exp_path_str}: {str(e)}"
            print(f"  ✗ {error_msg}")
            errors.append(error_msg)
    
    if not all_rounds and errors:
        print("\n❌ No experiments processed successfully. Errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    
    # Analyze first-round communication costs
    first_round_analysis = analyze_first_round_communication(all_rounds, experiments_summary)
    
    # Save results
    results = {
        "all_rounds": all_rounds,
        "experiments_summary": experiments_summary,
        "test_results": test_results,
        "first_round_analysis": first_round_analysis,
        "total_experiments": len(experiments_summary),
        "total_rounds": len(all_rounds),
        "experiments_with_test_eval": len(test_results),
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }
    
    return results, output_dir


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Save extraction results to CSV and JSON files.
    
    Args:
        results: Dictionary with extraction results
        output_dir: Output directory
    """
    # Save detailed JSON
    detail_json_path = output_dir / "rounds_detail.json"
    with open(detail_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved detailed results: {detail_json_path}")
    
    # Save summary JSON
    summary_json_path = output_dir / "experiments_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(results["experiments_summary"], f, indent=2)
    print(f"✓ Saved experiments summary: {summary_json_path}")
    
    # Save first-round communication analysis
    if "first_round_analysis" in results and results["first_round_analysis"]:
        first_round_json_path = output_dir / "first_round_communication_analysis.json"
        with open(first_round_json_path, 'w') as f:
            json.dump(results["first_round_analysis"], f, indent=2)
        print(f"✓ Saved first-round analysis: {first_round_json_path}")
        
        # Also save as CSV for easy inspection
        first_round_csv_path = output_dir / "first_round_communication_analysis.csv"
        if results["first_round_analysis"]:
            analysis_list = list(results["first_round_analysis"].values())
            fieldnames = ["exp_name", "initial_bytes_sent_mb", "initial_bytes_received_mb", "avg_update_bytes_sent_mb", "n_rounds", "total_bytes_sent_mb", "total_bytes_received_mb", "bytes_from_updates_only_mb", "bytes_source"]
            with open(first_round_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(analysis_list)
            print(f"✓ Saved first-round analysis CSV: {first_round_csv_path}")
    
    # Save test evaluation results
    if results["test_results"]:
        test_results_path = output_dir / "test_evaluation_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(results["test_results"], f, indent=2)
        print(f"✓ Saved test evaluation results: {test_results_path}")
    
    # Save CSV with all rounds
    csv_path = output_dir / "rounds_communication_metrics.csv"
    if results["all_rounds"]:
        fieldnames = ["experiment", "round", "total_optimizer_steps", "balanced_accuracy", "f1_score", "specificity", "sensitivity", "roc_auc", "bytes_sent", "bytes_received"]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results["all_rounds"])
        print(f"✓ Saved CSV with all rounds: {csv_path}")
    
    # Save per-experiment CSV files
    experiments_by_id = {}
    for round_data in results["all_rounds"]:
        exp_id = round_data["experiment"]
        if exp_id not in experiments_by_id:
            experiments_by_id[exp_id] = []
        experiments_by_id[exp_id].append(round_data)
    
    for exp_id, rounds in experiments_by_id.items():
        exp_csv_path = output_dir / f"{exp_id}.csv"
        fieldnames = ["round", "total_optimizer_steps", "balanced_accuracy", "f1_score", "specificity", "sensitivity", "roc_auc", "bytes_sent", "bytes_received"]
        with open(exp_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for round_data in rounds:
                writer.writerow({k: round_data[k] for k in fieldnames})
        print(f"✓ Saved per-experiment CSV: {exp_csv_path}")
    
    # Create manifest
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write("Communication Metrics Extraction Manifest\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {results['timestamp']}\n")
        f.write(f"Total experiments processed: {results['total_experiments']}\n")
        f.write(f"Total rounds extracted: {results['total_rounds']}\n")
        f.write(f"Experiments with test evaluation: {results['experiments_with_test_eval']}\n")
        f.write(f"\nExperiments:\n")
        for exp in results["experiments_summary"]:
            f.write(f"  - {exp['experiment_id']}\n")
            f.write(f"    Rounds: {exp['n_rounds']}\n")
            f.write(f"    Final steps: {exp['final_total_steps']}\n")
            f.write(f"    LoRA Configuration:\n")
            f.write(f"      use_lora: {exp.get('use_lora')}\n")
            f.write(f"      blocks_to_lora: {exp.get('blocks_to_lora')}\n")
            f.write(f"      dual_lora_enabled: {exp.get('dual_lora_enabled')}\n")
            f.write(f"    Validation Metrics (Final Round):\n")
            f.write(f"      balanced_accuracy: {exp.get('final_balanced_accuracy')}\n")
            f.write(f"      f1_score: {exp.get('final_f1_score')}\n")
            f.write(f"      roc_auc: {exp.get('final_roc_auc')}\n")
            
            # Add test evaluation metrics if available
            if exp.get('has_test_eval', False):
                f.write(f"    Test Evaluation Metrics:\n")
                for key, value in exp.items():
                    if key.startswith('test_'):
                        metric_name = key[5:]  # Remove 'test_' prefix
                        f.write(f"      {metric_name}: {value}\n")
            
            f.write(f"    Path: {exp['path']}\n")
        if results["errors"]:
            f.write(f"\nErrors ({len(results['errors'])}):\n")
            for err in results["errors"]:
                f.write(f"  - {err}\n")
    print(f"✓ Saved manifest: {manifest_path}")


def main():
    # Predefined paths for convenience (can be overridden by command-line arguments)
    PREDEFINED_PATHS = [
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/02-38-52/",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/03-38-16/",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/04-41-20/",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/05-51-45/",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/07-09-03/",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/08-32-43/",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/09-33-44/",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/10-40-44/",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/11-57-48/",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/13-22-29/",

        # Head-only runs
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_False/seed0/2026-02-19/22-29-36",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_False/seed1/2026-02-20/17-02-16",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_False/seed2/2026-02-20/18-00-18",

        # LoRA 2 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-23/17-49-20",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-23/18-50-20",
        # LoRA 4 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-23/19-51-00",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-23/20-55-07",
        # LoRA 6 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-23/21-58-34",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-23/23-08-50",
        # LoRA 8 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-24/00-18-40",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-24/01-35-40",
        # LoRA 10 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-24/02-53-26",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-24/04-18-04",
        # LoRA all blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-19/23-28-31",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-20/18-58-30",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-20/20-28-02",

        # Dual-LoRA 2 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-24/05-42-25",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-24/06-42-49",
        # Dual-LoRA 4 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-24/07-43-52",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-24/08-51-16",
        # Dual-LoRA 6 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-24/09-58-25",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-24/11-14-56",
        # # Dual-LoRA 8 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-24/12-32-36",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-24/13-58-39",
        # # Dual-LoRA 10 blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-24/15-24-28",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-24/16-59-18",

        # Dual-LoRA all blocks
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/00-58-03",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed1/2026-02-20/21-57-49",
        "outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed2/2026-02-20/23-39-08",
    ]
    
    parser = argparse.ArgumentParser(
        description="Extract communication budget and validation metrics from federated experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined paths (no arguments needed)
  python scripts/extract_communication_metrics.py

  # Single experiment (CLI override)
  python scripts/extract_communication_metrics.py \\
    outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_False/seed0/2026-02-20/16-03-59/

  # Multiple experiments (CLI override)
  python scripts/extract_communication_metrics.py \\
    outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_False/seed0/2026-02-20/16-03-59/ \\
    outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/lora_True/seed0/2026-02-20/16-05-30/

  # With glob expansion (bash)
  python scripts/extract_communication_metrics.py \\
    outputs/cinema_classification/federated/acdc_mnms_combined/binary_label/*/seed0/*/*/
        """
    )
    
    parser.add_argument(
        "experiment_paths",
        nargs="*",
        help="Paths to federated experiment directories (optional: uses predefined paths if not provided)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ablation_study_lora_communication",
        help="Output directory for results (default: outputs/ablation_study_lora_communication)"
    )
    
    args = parser.parse_args()
    
    # Use CLI paths if provided, otherwise use predefined paths
    if args.experiment_paths:
        experiment_paths = args.experiment_paths
        print("Using experiment paths from command-line arguments")
    else:
        experiment_paths = PREDEFINED_PATHS
        print(f"No experiment paths provided. Using {len(PREDEFINED_PATHS)} predefined paths")
    
    print("=" * 60)
    print("Communication Metrics Extraction")
    print("=" * 60)
    print(f"Processing {len(experiment_paths)} experiment(s)")
    print(f"Output directory: {args.output_dir}\n")
    
    results, output_dir = process_experiments(experiment_paths, Path(args.output_dir))
    
    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"Experiments processed: {results['total_experiments']}")
    print(f"Total rounds extracted: {results['total_rounds']}")
    print(f"Experiments with test evaluation: {results['experiments_with_test_eval']}")
    if results['errors']:
        print(f"Errors encountered: {len(results['errors'])}")
    
    # Print first-round communication analysis
    if "first_round_analysis" in results and results["first_round_analysis"]:
        print(f"\n{'=' * 60}")
        print("FIRST-ROUND COMMUNICATION COST BREAKDOWN")
        print(f"{'=' * 60}")
        print("\nInitialization Cost (First Round) vs Update Cost (Subsequent Rounds):\n")
        print(f"{'Experiment':<25} | {'Init (MB)':<10} | {'Avg Update (MB)':<15} | {'Total (MB)':<10} | {'Updates Only (MB)':<15} | {'Source':<10}")
        print("-" * 110)
        
        sorted_analysis = sorted(results["first_round_analysis"].items(), 
                                key=lambda x: x[1]["total_bytes_sent_mb"])
        
        inferred_count = 0
        extracted_count = 0
        
        for exp_id, analysis in sorted_analysis:
            exp_name = analysis["exp_name"]
            init_sent = analysis["initial_bytes_sent_mb"]
            avg_update = analysis["avg_update_bytes_sent_mb"]
            total_sent = analysis["total_bytes_sent_mb"]
            updates_only = analysis["bytes_from_updates_only_mb"]
            bytes_source = analysis.get("bytes_source", "unknown")
            
            source_label = "Inferred" if bytes_source == "inferred_from_config" else "Logged"
            if bytes_source == "inferred_from_config":
                inferred_count += 1
            else:
                extracted_count += 1
            
            print(f"{exp_name:<25} | {init_sent:<10.3f} | {avg_update:<15.3f} | {total_sent:<10.3f} | {updates_only:<15.3f} | {source_label:<10}")
        
        print("\n✓ First-round analysis explanation:")
        print("  - Init (MB): Model weights sent in first round (initialization)")
        print("  - Avg Update (MB): Average bytes per round after first round (model updates)")
        print("  - Total (MB): Cumulative bytes across all communication rounds")
        print("  - Updates Only (MB): Total bytes excluding the first-round initialization")
        print(f"\n✓ Data sources: {extracted_count} extracted from logs, {inferred_count} inferred from config")
        print("  - Inferred: Calculated from blocks_to_lora, freeze_backbone, and dual_lora_enabled settings")
        print("  - Extracted: Parsed directly from federated_server/server.log")
    
    save_results(results, output_dir)
    
    print(f"\n✓ All results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
