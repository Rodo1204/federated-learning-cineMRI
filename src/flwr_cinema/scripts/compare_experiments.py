#!/usr/bin/env python3
"""
Script to compare experimental results from different runs.
Takes multiple experiment folders and generates comparison tables and plots.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import re


def find_hydra_config(date_time_str: str, experiment_path: Path) -> Path | None:
    """Find Hydra config.yaml file given a date-time string and experiment path.
    
    Searches in multiple locations to support both old and new naming conventions.
    """
    if not date_time_str or ' ' not in date_time_str:
        return None
    
    date_part, time_part = date_time_str.split(' ')
    import os
    
    # Possible paths to search
    search_paths = [
        # Standard new format: outputs/{date}/{time}/.hydra/config.yaml
        experiment_path.parent.parent.parent / "outputs" / date_part / time_part / ".hydra" / "config.yaml",
        Path("outputs") / date_part / time_part / ".hydra" / "config.yaml",
        Path(os.getcwd()) / "outputs" / date_part / time_part / ".hydra" / "config.yaml",
    ]
    
    # Check for old federated format: outputs/YYYY-MM-DD_HH-MM-SS/.hydra/config.yaml
    # This would be in the outputs root directory
    combined_datetime = date_time_str.replace(' ', '_')
    for base_dir in [Path("outputs"), Path(os.getcwd()) / "outputs"]:
        # Search for any folder matching the pattern
        if base_dir.exists():
            for folder in base_dir.glob(f"{date_part}*"):
                if combined_datetime in folder.name or time_part.replace('-', '-') in folder.name:
                    hydra_file = folder / ".hydra" / "config.yaml"
                    if hydra_file.exists():
                        return hydra_file
    
    # Return the first existing path from standard locations
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def check_logs_for_lora(experiment_path: Path) -> bool:
    """Check if LoRA is mentioned in the experiment logs."""
    log_dir = experiment_path / "logs"
    if not log_dir.exists():
        return False
    
    # Check all log files for LoRA mentions
    import glob
    for log_file in glob.glob(str(log_dir / "*.log")):
        try:
            with open(log_file, 'r') as f:
                content = f.read().lower()
                if 'applying lora' in content or 'lora adapters' in content:
                    return True
        except:
            pass
    
    return False


def find_seed_in_path(path: Path) -> int | None:
    """Extract seed number from path if it exists."""
    # Look for seedX pattern in the path
    for part in path.parts:
        match = re.search(r'seed(\d+)', part)
        if match:
            return int(match.group(1))
    return None


def find_experiment_seeds(parent_path: Path) -> Dict[int, Path]:
    """Check if parent_path contains subdirectories with different seeds.
    
    Returns a dict mapping seed number to seed directory path.
    Returns empty dict if no seeds found.
    """
    seeds = {}
    if not parent_path.exists() or not parent_path.is_dir():
        return seeds
    
    for item in parent_path.iterdir():
        if not item.is_dir():
            continue
        
        seed = find_seed_in_path(item)
        if seed is not None:
            seeds[seed] = item
    
    return seeds


def find_actual_run_directory(base_path: Path) -> Path | None:
    """Find the actual run directory which contains test_evaluation/test_metrics.json.
    
    Handles nested directory structures where runs might be in:
    - base_path/test_evaluation/test_metrics.json
    - base_path/YYYY-MM-DD/HH-MM-SS/test_evaluation/test_metrics.json
    """
    # Check if test_metrics.json exists directly in base_path
    if (base_path / "test_evaluation" / "test_metrics.json").exists():
        return base_path
    
    # Check for nested date/time directories (one level deep)
    if base_path.exists() and base_path.is_dir():
        for item in base_path.iterdir():
            if item.is_dir():
                # Check if this directory contains test_evaluation
                if (item / "test_evaluation" / "test_metrics.json").exists():
                    return item
                
                # Check for date directory (YYYY-MM-DD)
                if len(item.name) == 10 and item.name[4] == '-' and item.name[7] == '-':
                    # This looks like a date directory, check its subdirectories
                    for time_item in item.iterdir():
                        if time_item.is_dir() and (time_item / "test_evaluation" / "test_metrics.json").exists():
                            return time_item
    
    return None


def load_test_metrics(experiment_path: Path) -> Dict[str, Any]:
    """Load test metrics from an experiment folder."""
    # Find the actual run directory
    run_dir = find_actual_run_directory(experiment_path)
    
    if run_dir is None:
        raise FileNotFoundError(f"test_metrics.json not found in {experiment_path}")
    
    test_metrics_file = run_dir / "test_evaluation" / "test_metrics.json"
    
    with open(test_metrics_file, 'r') as f:
        return json.load(f)


def extract_experiment_label(experiment_path: Path) -> Dict[str, Any]:
    """Extract experiment parameters to create a descriptive label.
    
    Tries to load config from Hydra output folder first, falls back to path inference.
    
    Returns a dict with keys:
        - model_name: Name of the model (cinema, vita, etc.)
        - use_lora: Whether LoRA is used
        - mode: centralized or federated
        - regression_column: Regression target (lv_ef, etc.)
        - date_time: Date and time from the path
        - lora_rank: LoRA rank (if applicable)
        - lora_alpha: LoRA alpha (if applicable)
        - blocks_to_lora: Blocks with LoRA (if applicable)
        - dual_lora: Whether Dual LoRA is used (if applicable)
    """
    folder_name = experiment_path.name
    parent_structure = experiment_path.parent.parts
    
    # Initialize with defaults
    params = {
        'model_name': 'cinema',
        'use_lora': False,
        'mode': 'unknown',
        'regression_column': 'lv_ef',
        'date_time': None,
        'lora_rank': None,
        'lora_alpha': None,
        'blocks_to_lora': None,
        'dual_lora': False,
        'task': 'regression',
    }
    
    # Extract date and time from parent folder structure for later use
    # Format: .../2026-02-03/13-45-18_lv_ef_seed0 or ...2026-02-04_10-05-39_lv_ef_seed0
    date_time_str = None
    if len(parent_structure) >= 2:
        # Check for date folder (YYYY-MM-DD)
        for i, part in enumerate(parent_structure[-3:]):
            if part and len(part) == 10 and part[4] == '-' and part[7] == '-':
                # This is a date folder
                date_part = part
                time_part = folder_name.split('_')[0]  # HH-MM-SS
                date_time_str = f"{date_part} {time_part}"
                params['date_time'] = date_time_str
                break
        
        # If not found, check if time is in folder name (for federated format: YYYY-MM-DD_HH-MM-SS)
        if not date_time_str and '_' in folder_name:
            parts = folder_name.split('_')
            if len(parts[0]) == 10 and parts[0][4] == '-':
                date_time_str = f"{parts[0]} {parts[1]}"
                params['date_time'] = date_time_str
    
    # Try to find and load config from experiment folder (config.yaml in base dir or federated_server subdirectory)
    config_paths = [
        experiment_path / "config.yaml",
        experiment_path / "federated_server" / "config.yaml",
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    # Extract mode
                    if 'mode' in config:
                        params['mode'] = config['mode']
                    
                    # Extract task
                    if 'task' in config:
                        params['task'] = config['task']
                    
                    # Extract use_lora
                    if 'train' in config and 'use_lora' in config['train']:
                        params['use_lora'] = config['train']['use_lora']
                    
                    # Extract LoRA parameters
                    if 'lora' in config and params['use_lora']:
                        lora_cfg = config['lora']
                        params['lora_rank'] = lora_cfg.get('rank')
                        params['lora_alpha'] = lora_cfg.get('alpha')
                        blocks = lora_cfg.get('blocks_to_lora')
                        params['blocks_to_lora'] = blocks if blocks else 'all'
                    
                    # Extract Dual LoRA parameters
                    if 'dual_lora' in config and config['dual_lora'].get('enabled', False):
                        params['dual_lora'] = True
                    
                    # Extract regression column
                    if 'data' in config and 'regression_column' in config['data']:
                        params['regression_column'] = config['data']['regression_column']
                    
                    return params
            except Exception as e:
                print(f"Warning: Could not parse config at {config_path}: {e}")
    
    # Try to find and load Hydra config from outputs folder
    # Hydra saves configs in outputs/{YYYY-MM-DD}/{HH-MM-SS}/.hydra/config.yaml
    if date_time_str:
        hydra_config_path = find_hydra_config(date_time_str, experiment_path)
        
        if hydra_config_path and hydra_config_path.exists():
            try:
                with open(hydra_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    # Extract mode
                    if 'mode' in config:
                        params['mode'] = config['mode']
                    
                    # Extract task
                    if 'task' in config:
                        params['task'] = config['task']
                    
                    # Extract use_lora
                    if 'train' in config and 'use_lora' in config['train']:
                        params['use_lora'] = config['train']['use_lora']
                    
                    # Extract LoRA parameters
                    if 'lora' in config and params['use_lora']:
                        lora_cfg = config['lora']
                        params['lora_rank'] = lora_cfg.get('rank')
                        params['lora_alpha'] = lora_cfg.get('alpha')
                        blocks = lora_cfg.get('blocks_to_lora')
                        params['blocks_to_lora'] = blocks if blocks else 'all'
                    
                    # Extract Dual LoRA parameters
                    if 'dual_lora' in config and config['dual_lora'].get('enabled', False):
                        params['dual_lora'] = True
                    
                    # Extract regression column
                    if 'data' in config and 'regression_column' in config['data']:
                        params['regression_column'] = config['data']['regression_column']
                    
                    return params
            except Exception as e:
                print(f"Warning: Could not parse Hydra config at {hydra_config_path}: {e}")
    
    # Fallback: Check logs for LoRA usage
    if not params['use_lora']:
        params['use_lora'] = check_logs_for_lora(experiment_path)
    
    # Fallback: Determine mode from folder structure or name
    if 'centralized' in parent_structure or 'centralized' in folder_name.lower():
        params['mode'] = 'centralized'
    elif 'federated' in parent_structure or 'federated' in folder_name.lower():
        params['mode'] = 'federated'
    
    # Extract parameters from folder name and parent structure
    # Check all parts of the path for lora keyword
    all_path_parts = folder_name.split('_') + list(parent_structure)
    
    # Look for lora keyword in any part of the path
    for part in all_path_parts:
        if 'lora_true' in part.lower():
            params['use_lora'] = True
            break
        elif 'lora_false' in part.lower():
            params['use_lora'] = False
            break
    
    # Look for regression column (lv_ef, ejection_fraction, ef)
    for part in all_path_parts:
        if part in ['lv_ef', 'ejection_fraction', 'ef']:
            params['regression_column'] = part
    
    return params


def create_experiment_label(params: Dict[str, Any]) -> str:
    """Create a readable label from experiment parameters."""
    label_parts = [params['model_name']]
    
    if params['use_lora']:
        label_parts.append('LoRA')
    else:
        label_parts.append('full')
    
    label_parts.append(params['mode'])
    
    # Add timestamp to distinguish between runs with same settings
    if params['date_time']:
        label_parts.append(f"({params['date_time']}")
    
    return ' '.join(label_parts)


def print_architecture_details(params: Dict[str, Any]) -> None:
    """Print detailed LoRA architecture information."""
    if not params.get('use_lora', False):
        return
    
    print(f"  LoRA Architecture:")
    
    if params.get('dual_lora', False):
        print(f"    Type: Dual LoRA (personalized federated learning)")
    else:
        print(f"    Type: Standard LoRA")
    
    if params.get('lora_rank'):
        print(f"    Rank: {params['lora_rank']}")
    if params.get('lora_alpha'):
        print(f"    Alpha: {params['lora_alpha']}")
    
    blocks = params.get('blocks_to_lora')
    if blocks and blocks != 'all':
        if isinstance(blocks, list):
            print(f"    Blocks: {blocks} ({len(blocks)} blocks)")
        else:
            print(f"    Blocks: {blocks}")
    else:
        print(f"    Blocks: all (12 blocks for ViT-B)")


def load_training_metrics(experiment_path: Path) -> List[Dict[str, Any]]:
    """Load validation/training metrics from an experiment folder."""
    # Find the actual run directory first
    run_dir = find_actual_run_directory(experiment_path)
    if run_dir is None:
        # Fallback to original path if no nested structure found
        run_dir = experiment_path
    
    # For centralized runs
    validation_file = run_dir / "validation_metrics.jsonl"
    if validation_file.exists():
        metrics = []
        with open(validation_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics
    
    # For federated runs
    aggregated_metrics_file = run_dir / "federated_server" / "aggregated_metrics_history.json"
    if aggregated_metrics_file.exists():
        with open(aggregated_metrics_file, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"Training metrics not found in {experiment_path}")


def extract_config_from_experiment(experiment_path: Path) -> Dict[str, Any]:
    """Try to extract Hydra config from experiment folder.
    
    Falls back to parameter extraction from folder name and folder structure.
    """
    return extract_experiment_label(experiment_path)


def aggregate_metrics_across_seeds(seed_paths: Dict[int, Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and aggregate metrics across multiple seed runs.
    
    Returns (mean_metrics, std_metrics) dicts with the same structure as individual metrics.
    """
    all_metrics = []
    
    for seed, path in sorted(seed_paths.items()):
        try:
            metrics = load_test_metrics(path)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Warning: Could not load metrics for seed {seed} at {path}: {e}")
    
    if not all_metrics:
        raise ValueError(f"No valid metrics found across seeds")
    
    # Aggregate metrics
    mean_metrics = {}
    std_metrics = {}
    
    # Helper function to recursively aggregate dictionaries
    def aggregate_dict(metrics_list, key_path=[]):
        if not metrics_list:
            return {}, {}
        
        first = metrics_list[0]
        if isinstance(first, dict):
            mean_dict = {}
            std_dict = {}
            for key in first.keys():
                values = [m[key] for m in metrics_list if key in m]
                if values and isinstance(values[0], (int, float)):
                    mean_dict[key] = float(np.mean(values))
                    std_dict[key] = float(np.std(values))
                elif values and isinstance(values[0], dict):
                    m_d, s_d = aggregate_dict(values, key_path + [key])
                    mean_dict[key] = m_d
                    std_dict[key] = s_d
            return mean_dict, std_dict
        return {}, {}
    
    mean_metrics, std_metrics = aggregate_dict(all_metrics)
    return mean_metrics, std_metrics


def extract_restored_mae_table(experiments: Dict[str, Tuple[Path, Dict, Dict]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract restored_mae values from test_metrics for all experiments.
    
    Returns (mean_df, std_df) tuple with mean and standard deviation tables.
    """
    mean_results = {}
    std_results = {}
    
    for exp_name, (exp_path, params, seed_paths) in experiments.items():
        label = create_experiment_label(params)
        
        if seed_paths:
            # Multiple seeds - aggregate
            mean_metrics, std_metrics = aggregate_metrics_across_seeds(seed_paths)
        else:
            # Single run
            mean_metrics = load_test_metrics(exp_path)
            std_metrics = {}
        
        mean_results[label] = {
            'Overall': mean_metrics['overall']['restored_mae'],
            'Weighted': mean_metrics['weighted']['restored_mae'],
        }
        
        # Add per-client metrics
        if 'per_client' in mean_metrics:
            for client, metrics in mean_metrics['per_client'].items():
                mean_results[label][client] = metrics['restored_mae']
        
        # Add standard deviations if available
        if std_metrics:
            std_results[label] = {
                'Overall': std_metrics.get('overall', {}).get('restored_mae', 0),
                'Weighted': std_metrics.get('weighted', {}).get('restored_mae', 0),
            }
            if 'per_client' in std_metrics:
                for client, metrics in std_metrics['per_client'].items():
                    std_results[label][f"{client}"] = metrics.get('restored_mae', 0)
    
    mean_df = pd.DataFrame(mean_results).T.round(4)
    std_df = pd.DataFrame(std_results).T.round(4) if std_results else None
    return mean_df, std_df


def extract_mae_by_client_table(experiments: Dict[str, Tuple[Path, Dict, Dict]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract MAE values split by client/center.
    
    Returns (mean_df, std_df) tuple with mean and standard deviation tables.
    """
    mean_results = {}
    std_results = {}
    
    for exp_name, (exp_path, params, seed_paths) in experiments.items():
        label = create_experiment_label(params)
        
        if seed_paths:
            # Multiple seeds - aggregate
            mean_metrics, std_metrics = aggregate_metrics_across_seeds(seed_paths)
        else:
            # Single run
            mean_metrics = load_test_metrics(exp_path)
            std_metrics = {}
        
        mean_results[label] = {}
        
        if 'per_client' in mean_metrics:
            for client, metrics in mean_metrics['per_client'].items():
                mean_results[label][f"{client}_mae"] = metrics['mae']
                mean_results[label][f"{client}_restored_mae"] = metrics['restored_mae']
        
        # For cross-center metrics in federated runs
        if 'cross_center' in mean_metrics:
            for cross_center, metrics in mean_metrics['cross_center'].items():
                mean_results[label][f"{cross_center}_mae"] = metrics['mae']
                mean_results[label][f"{cross_center}_restored_mae"] = metrics['restored_mae']
        
        # Add standard deviations
        if std_metrics:
            std_results[label] = {}
            if 'per_client' in std_metrics:
                for client, metrics in std_metrics['per_client'].items():
                    std_results[label][f"{client}_mae"] = metrics.get('mae', 0)
                    std_results[label][f"{client}_restored_mae"] = metrics.get('restored_mae', 0)
            if 'cross_center' in std_metrics:
                for cross_center, metrics in std_metrics['cross_center'].items():
                    std_results[label][f"{cross_center}_mae"] = metrics.get('mae', 0)
                    std_results[label][f"{cross_center}_restored_mae"] = metrics.get('restored_mae', 0)
    
    mean_df = pd.DataFrame(mean_results).T.round(4)
    std_df = pd.DataFrame(std_results).T.round(4) if std_results else None
    return mean_df, std_df


def plot_training_progression(experiments: Dict[str, Tuple[Path, Dict, Dict]], output_dir: Path) -> None:
    """Plot the finetuning training progression for all runs using optimizer steps.
    
    Shows mean and std when multiple seeds are available.
    Skips experiments that don't have training metrics available.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Finetuning Training Progression Comparison (by Optimization Steps)', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    experiments_plotted = 0
    
    for idx, (exp_name, (exp_path, params, seed_paths)) in enumerate(experiments.items()):
        color = colors[idx % len(colors)]
        label = create_experiment_label(params)
        
        # Collect training metrics from all seeds
        all_training_metrics = []
        paths_to_check = list(seed_paths.values()) if seed_paths else [exp_path]
        
        for path in paths_to_check:
            try:
                training_metrics = load_training_metrics(path)
                all_training_metrics.append(training_metrics)
            except Exception as e:
                print(f"Info: Could not load training metrics from {path}: {e}")
        
        if not all_training_metrics:
            print(f"Skipping {label}: no training metrics available")
            continue
        
        # Aggregate metrics across seeds
        # First, align metrics by optimizer steps
        all_optimizer_steps = set()
        metrics_by_steps = {}
        
        for metrics_list in all_training_metrics:
            for metric in metrics_list:
                opt_steps = metric.get('optimizer_steps') or metric.get('total_optimizer_steps')
                if opt_steps is not None:
                    all_optimizer_steps.add(opt_steps)
                    if opt_steps not in metrics_by_steps:
                        metrics_by_steps[opt_steps] = []
                    metrics_by_steps[opt_steps].append(metric)
        
        if not all_optimizer_steps:
            print(f"Skipping {label}: no optimizer steps found in metrics")
            continue
        
        # Compute mean and std for each metric
        optimizer_steps = sorted(all_optimizer_steps)
        restored_mae_means, restored_mae_stds = [], []
        restored_rmse_means, restored_rmse_stds = [], []
        
        for opt_step in optimizer_steps:
            metrics_at_step = metrics_by_steps[opt_step]
            
            # Extract restored MAE from fedavg_metrics
            restored_mae_vals = [m['fedavg_metrics']['restored_mae'] for m in metrics_at_step if 'fedavg_metrics' in m and 'restored_mae' in m['fedavg_metrics']]
            
            # Extract restored RMSE from fedavg_metrics (as loss metric)
            restored_rmse_vals = [m['fedavg_metrics']['restored_rmse'] for m in metrics_at_step if 'fedavg_metrics' in m and 'restored_rmse' in m['fedavg_metrics']]
            
            if restored_mae_vals:
                restored_mae_means.append(np.mean(restored_mae_vals))
                restored_mae_stds.append(np.std(restored_mae_vals))
            if restored_rmse_vals:
                restored_rmse_means.append(np.mean(restored_rmse_vals))
                restored_rmse_stds.append(np.std(restored_rmse_vals))
        
        # Plot with error bands if we have multiple seeds
        has_multiple_seeds = len(all_training_metrics) > 1
        
        # Plot Restored MAE (only if we have data)
        if restored_mae_means:
            axes[0].plot(optimizer_steps, restored_mae_means, marker='o', label=label, color=color, linewidth=2)
            if has_multiple_seeds:
                axes[0].fill_between(optimizer_steps, 
                                       np.array(restored_mae_means) - np.array(restored_mae_stds),
                                       np.array(restored_mae_means) + np.array(restored_mae_stds),
                                       alpha=0.2, color=color)
        
        # Plot Restored RMSE as Loss (only if we have data)
        if restored_rmse_means:
            axes[1].plot(optimizer_steps, restored_rmse_means, marker='o', label=label, color=color, linewidth=2)
            if has_multiple_seeds:
                axes[1].fill_between(optimizer_steps, 
                                       np.array(restored_rmse_means) - np.array(restored_rmse_stds),
                                       np.array(restored_rmse_means) + np.array(restored_rmse_stds),
                                       alpha=0.2, color=color)
        
        experiments_plotted += 1
    
    if experiments_plotted == 0:
        print("Warning: No experiments had training metrics to plot")
        plt.close()
        return
    
    # Configure axes
    axes[0].set_xlabel('Optimizer Steps')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('MAE (Denormalized)')
    if axes[0].has_data():
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Optimizer Steps')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE (Denormalized)')
    if axes[1].has_data():
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_progression.png", dpi=300, bbox_inches='tight')
    print(f"Saved training progression plot to {output_dir / 'training_progression.png'}")
    plt.close()


def plot_training_progression_by_epoch(experiments: Dict[str, Tuple[Path, Dict, Dict]], output_dir: Path) -> None:
    """Plot the finetuning training progression by epoch/round (aligned across runs).
    
    For centralized training, uses epochs. For federated learning, uses rounds.
    Both are treated equivalently for comparison.
    Shows mean and std when multiple seeds are available.
    Skips experiments that don't have training metrics available.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Finetuning Training Progression Comparison (by Epoch/Round)', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    experiments_plotted = 0
    
    for idx, (exp_name, (exp_path, params, seed_paths)) in enumerate(experiments.items()):
        color = colors[idx % len(colors)]
        label = create_experiment_label(params)
        
        # Collect training metrics from all seeds
        all_training_metrics = []
        paths_to_check = list(seed_paths.values()) if seed_paths else [exp_path]
        
        for path in paths_to_check:
            try:
                all_training_metrics.append(load_training_metrics(path))
            except Exception as e:
                pass
        
        if not all_training_metrics:
            print(f"Skipping {label}: no training metrics available")
            continue
        
        # Aggregate metrics by epoch (or round for FL)
        max_epoch = 0
        metrics_by_epoch = {}
        is_federated = False
        
        for metrics_list in all_training_metrics:
            for metric in metrics_list:
                # Handle both centralized (epoch) and federated (round) formats
                epoch = metric.get('epoch') or metric.get('round')
                if epoch is not None:
                    max_epoch = max(max_epoch, epoch)
                    if epoch not in metrics_by_epoch:
                        metrics_by_epoch[epoch] = []
                    metrics_by_epoch[epoch].append(metric)
                    # Detect if this is federated by checking for 'round' field
                    if metric.get('round') is not None:
                        is_federated = True
        
        if not metrics_by_epoch:
            print(f"Skipping {label}: no epoch/round information found in metrics")
            continue
        
        # Compute mean and std for each epoch/round
        epochs = sorted(metrics_by_epoch.keys())
        restored_mae_means, restored_mae_stds = [], []
        restored_rmse_means, restored_rmse_stds = [], []
        
        for epoch in epochs:
            metrics_at_epoch = metrics_by_epoch[epoch]
            
            # Extract restored MAE from fedavg_metrics
            restored_mae_vals = [m['fedavg_metrics']['restored_mae'] for m in metrics_at_epoch if 'fedavg_metrics' in m and 'restored_mae' in m['fedavg_metrics']]
            
            # Extract restored RMSE from fedavg_metrics
            restored_rmse_vals = [m['fedavg_metrics']['restored_rmse'] for m in metrics_at_epoch if 'fedavg_metrics' in m and 'restored_rmse' in m['fedavg_metrics']]
            
            if restored_mae_vals:
                restored_mae_means.append(np.mean(restored_mae_vals))
                restored_mae_stds.append(np.std(restored_mae_vals))
            if restored_rmse_vals:
                restored_rmse_means.append(np.mean(restored_rmse_vals))
                restored_rmse_stds.append(np.std(restored_rmse_vals))
        
        # Plot with error bands if we have multiple seeds
        has_multiple_seeds = len(all_training_metrics) > 1
        
        # Plot Restored MAE (only if we have data)
        if restored_mae_means:
            axes[0].plot(epochs, restored_mae_means, marker='o', label=label, color=color, linewidth=2)
            if has_multiple_seeds:
                axes[0].fill_between(epochs, 
                                       np.array(restored_mae_means) - np.array(restored_mae_stds),
                                       np.array(restored_mae_means) + np.array(restored_mae_stds),
                                       alpha=0.2, color=color)
        
        # Plot Restored RMSE (only if we have data)
        if restored_rmse_means:
            axes[1].plot(epochs, restored_rmse_means, marker='o', label=label, color=color, linewidth=2)
            if has_multiple_seeds:
                axes[1].fill_between(epochs, 
                                       np.array(restored_rmse_means) - np.array(restored_rmse_stds),
                                       np.array(restored_rmse_means) + np.array(restored_rmse_stds),
                                       alpha=0.2, color=color)
        
        experiments_plotted += 1
    
    if experiments_plotted == 0:
        print("Warning: No experiments had training metrics to plot")
        plt.close()
        return
    
    # Configure axes with appropriate labels
    # Use "Round" for FL or "Epoch" for centralized
    epoch_label = "Round/Epoch" if any('federated' in str(params['mode']).lower() for _, (_, params, _) in experiments.items()) else "Epoch"
    
    axes[0].set_xlabel(f'{epoch_label}')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('MAE (Denormalized)')
    if axes[0].has_data():
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel(f'{epoch_label}')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE (Denormalized)')
    if axes[1].has_data():
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_progression_by_epoch.png", dpi=300, bbox_inches='tight')
    print(f"Saved training progression (by epoch) plot to {output_dir / 'training_progression_by_epoch.png'}")
    plt.close()


def plot_training_progression_normalized(experiments: Dict[str, Tuple[Path, Dict, Dict]], output_dir: Path) -> None:
    """Plot the finetuning training progression with normalized progress (0-100%).
    
    Normalizes each run's progress to 0-100% so they align regardless of total steps.
    Shows mean and std when multiple seeds are available.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Finetuning Training Progression Comparison (Normalized Progress)', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    experiments_plotted = 0
    
    for idx, (exp_name, (exp_path, params, seed_paths)) in enumerate(experiments.items()):
        color = colors[idx % len(colors)]
        label = create_experiment_label(params)
        
        # Collect training metrics from all seeds
        all_training_metrics = []
        paths_to_check = list(seed_paths.values()) if seed_paths else [exp_path]
        
        for path in paths_to_check:
            try:
                all_training_metrics.append(load_training_metrics(path))
            except Exception as e:
                pass
        
        if not all_training_metrics:
            print(f"Skipping {label}: no training metrics available")
            continue
        
        # For each seed, normalize its optimizer steps to 0-100%
        all_normalized_metrics = []
        
        for metrics_list in all_training_metrics:
            if not metrics_list:
                continue
                
            # Handle both centralized and federated formats
            max_steps = 0
            for m in metrics_list:
                steps = m.get('optimizer_steps') or m.get('total_optimizer_steps')
                if steps:
                    max_steps = max(max_steps, steps)
            
            if max_steps == 0:
                continue
            
            normalized_metrics = []
            for metric in metrics_list:
                opt_steps = metric.get('optimizer_steps') or metric.get('total_optimizer_steps')
                if opt_steps is not None and max_steps > 0:
                    normalized_progress = (opt_steps / max_steps) * 100
                    metric_copy = metric.copy()
                    metric_copy['normalized_progress'] = normalized_progress
                    normalized_metrics.append(metric_copy)
            
            if normalized_metrics:
                all_normalized_metrics.append(normalized_metrics)
        
        if not all_normalized_metrics:
            print(f"Skipping {label}: could not normalize metrics")
            continue
        
        # Aggregate by normalized progress points
        progress_points = set()
        metrics_by_progress = {}
        
        for metrics_list in all_normalized_metrics:
            for metric in metrics_list:
                prog = metric.get('normalized_progress')
                if prog is not None:
                    # Round to nearest 5% for grouping
                    prog_rounded = round(prog / 5) * 5
                    progress_points.add(prog_rounded)
                    if prog_rounded not in metrics_by_progress:
                        metrics_by_progress[prog_rounded] = []
                    metrics_by_progress[prog_rounded].append(metric)
        
        if not progress_points:
            print(f"Skipping {label}: no normalized progress found")
            continue
        
        # Compute mean and std for each progress point
        progress = sorted(progress_points)
        restored_mae_means, restored_mae_stds = [], []
        restored_rmse_means, restored_rmse_stds = [], []
        
        for prog in progress:
            metrics_at_prog = metrics_by_progress[prog]
            
            # Extract restored MAE from fedavg_metrics
            restored_mae_vals = [m['fedavg_metrics']['restored_mae'] for m in metrics_at_prog if 'fedavg_metrics' in m and 'restored_mae' in m['fedavg_metrics']]
            
            # Extract restored RMSE from fedavg_metrics
            restored_rmse_vals = [m['fedavg_metrics']['restored_rmse'] for m in metrics_at_prog if 'fedavg_metrics' in m and 'restored_rmse' in m['fedavg_metrics']]
            
            if restored_mae_vals:
                restored_mae_means.append(np.mean(restored_mae_vals))
                restored_mae_stds.append(np.std(restored_mae_vals))
            if restored_rmse_vals:
                restored_rmse_means.append(np.mean(restored_rmse_vals))
                restored_rmse_stds.append(np.std(restored_rmse_vals))
        
        # Plot with error bands if we have multiple seeds
        has_multiple_seeds = len(all_normalized_metrics) > 1
        
        # Plot Restored MAE (only if we have data)
        if restored_mae_means:
            axes[0].plot(progress, restored_mae_means, marker='o', label=label, color=color, linewidth=2)
            if has_multiple_seeds:
                axes[0].fill_between(progress, 
                                       np.array(restored_mae_means) - np.array(restored_mae_stds),
                                       np.array(restored_mae_means) + np.array(restored_mae_stds),
                                       alpha=0.2, color=color)
        
        # Plot Restored RMSE (only if we have data)
        if restored_rmse_means:
            axes[1].plot(progress, restored_rmse_means, marker='o', label=label, color=color, linewidth=2)
            if has_multiple_seeds:
                axes[1].fill_between(progress, 
                                       np.array(restored_rmse_means) - np.array(restored_rmse_stds),
                                       np.array(restored_rmse_means) + np.array(restored_rmse_stds),
                                       alpha=0.2, color=color)
        
        experiments_plotted += 1
    
    if experiments_plotted == 0:
        print("Warning: No experiments had training metrics to plot")
        plt.close()
        return
    
    # Configure axes
    axes[0].set_xlabel('Training Progress (%)')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('MAE (Denormalized)')
    axes[0].set_xlim([0, 100])
    if axes[0].has_data():
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Training Progress (%)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE (Denormalized)')
    axes[1].set_xlim([0, 100])
    if axes[1].has_data():
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_progression_normalized.png", dpi=300, bbox_inches='tight')
    print(f"Saved training progression (normalized) plot to {output_dir / 'training_progression_normalized.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare experimental results from different runs.')
    parser.add_argument('experiments', nargs='+', help='Paths to experiment folders')
    parser.add_argument('--output-dir', type=Path, 
                       default=None,
                       help='Output directory for results (default: auto-generated with timestamp)')
    parser.add_argument('--show-architecture', action='store_true',
                       help='Display LoRA architecture details (rank, blocks, parameters)')
    args = parser.parse_args()
    
    # Create timestamped output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = Path('outputs/cinema_regression/comparison_results') / timestamp
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load experiments with their parameters
    experiments = {}
    for exp_path_str in args.experiments:
        exp_path = Path(exp_path_str)
        if not exp_path.exists():
            print(f"Warning: Experiment path {exp_path} does not exist, skipping...")
            continue
        
        # Check if the provided path itself contains a seed (e.g., /path/seed0/)
        path_has_seed = find_seed_in_path(exp_path)
        
        if path_has_seed is not None:
            # This path contains a seed - look for sibling seeds in the parent directory
            parent_path = exp_path.parent
            seed_paths = find_experiment_seeds(parent_path)
            
            if seed_paths:
                # Parent directory has multiple seeds - treat them as the same run
                # Extract params from any seed (they should all be the same)
                params = extract_config_from_experiment(list(seed_paths.values())[0])
                # Create unique key by including mode to avoid collisions
                exp_key = f"{parent_path.name}_{params['mode']}"
                experiments[exp_key] = (parent_path, params, seed_paths)
                
                label = create_experiment_label(params)
                print(f"Loaded experiment: {parent_path.name} (with {len(seed_paths)} seeds)")
                print(f"  Seeds: {sorted(seed_paths.keys())}")
                print(f"  Label: {label}")
                print(f"  Mode: {params['mode']}")
                print(f"  LoRA: {params['use_lora']}")
                if args.show_architecture:
                    print_architecture_details(params)
            else:
                # No sibling seeds - treat as single run
                params = extract_config_from_experiment(exp_path)
                # Create unique key by including mode to avoid collisions
                exp_key = f"{exp_path.name}_{params['mode']}"
                experiments[exp_key] = (exp_path, params, {})
                
                label = create_experiment_label(params)
                print(f"Loaded experiment: {exp_path.name}")
                print(f"  Label: {label}")
                print(f"  Mode: {params['mode']}")
                print(f"  LoRA: {params['use_lora']}")
                if args.show_architecture:
                    print_architecture_details(params)
        else:
            # Check if this directory contains seed subdirectories
            seed_paths = find_experiment_seeds(exp_path)
            
            if seed_paths:
                # This is a parent directory with multiple seeds
                # Extract params from any seed (they should all be the same)
                params = extract_config_from_experiment(list(seed_paths.values())[0])
                # Create unique key by including mode to avoid collisions
                exp_key = f"{exp_path.name}_{params['mode']}"
                experiments[exp_key] = (exp_path, params, seed_paths)
                
                label = create_experiment_label(params)
                print(f"Loaded experiment: {exp_path.name} (with {len(seed_paths)} seeds)")
                print(f"  Seeds: {sorted(seed_paths.keys())}")
                print(f"  Label: {label}")
                print(f"  Mode: {params['mode']}")
                print(f"  LoRA: {params['use_lora']}")
                if args.show_architecture:
                    print_architecture_details(params)
            else:
                # Single run without seeds
                params = extract_config_from_experiment(exp_path)
                # Create unique key by including mode to avoid collisions
                exp_key = f"{exp_path.name}_{params['mode']}"
                experiments[exp_key] = (exp_path, params, {})
                
                label = create_experiment_label(params)
                print(f"Loaded experiment: {exp_path.name}")
                print(f"  Label: {label}")
                print(f"  Mode: {params['mode']}")
                print(f"  LoRA: {params['use_lora']}")
                if args.show_architecture:
                    print_architecture_details(params)
    
    if not experiments:
        print("No valid experiments found!")
        return
    
    print(f"\nProcessing {len(experiments)} experiments...")
    
    # Extract restored_mae table
    print("\nExtracting Restored MAE values...")
    restored_mae_mean_df, restored_mae_std_df = extract_restored_mae_table(experiments)
    restored_mae_mean_df.to_csv(output_dir / "restored_mae_comparison_mean.csv")
    print("Restored MAE comparison table (Mean):")
    print(restored_mae_mean_df)
    print(f"Saved to {output_dir / 'restored_mae_comparison_mean.csv'}")
    
    if restored_mae_std_df is not None:
        restored_mae_std_df.to_csv(output_dir / "restored_mae_comparison_std.csv")
        print("Restored MAE comparison table (Std):")
        print(restored_mae_std_df)
        print(f"Saved to {output_dir / 'restored_mae_comparison_std.csv'}")
    print()
    
    # Extract MAE by client table
    print("Extracting MAE by Client/Center...")
    mae_by_client_mean_df, mae_by_client_std_df = extract_mae_by_client_table(experiments)
    mae_by_client_mean_df.to_csv(output_dir / "mae_by_client_comparison_mean.csv")
    print("MAE by Client/Center table (Mean):")
    print(mae_by_client_mean_df)
    print(f"Saved to {output_dir / 'mae_by_client_comparison_mean.csv'}")
    
    if mae_by_client_std_df is not None:
        mae_by_client_std_df.to_csv(output_dir / "mae_by_client_comparison_std.csv")
        print("MAE by Client/Center table (Std):")
        print(mae_by_client_std_df)
        print(f"Saved to {output_dir / 'mae_by_client_comparison_std.csv'}")
    print()
    
    # Create training progression plots
    print("Creating training progression plots...")
    plot_training_progression(experiments, output_dir)
    plot_training_progression_by_epoch(experiments, output_dir)
    plot_training_progression_normalized(experiments, output_dir)
    
    # Save experiment metadata with labels
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'experiments': {
            name: {
                'path': str(path),
                'label': create_experiment_label(params),
                'params': params,
                'num_seeds': len(seed_paths),
                'seeds': sorted(seed_paths.keys()) if seed_paths else []
            }
            for name, (path, params, seed_paths) in experiments.items()
        },
        'num_experiments': len(experiments),
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Comparison complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
