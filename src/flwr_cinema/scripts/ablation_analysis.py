"""Analysis tools for LoRA ablation studies in federated learning.

This script provides utilities to analyze ablation experiments by:
1. Extracting configuration parameters (blocks_to_lora, rank, alpha, etc.)
2. Calculating parameter counts and communication overhead
3. Aggregating validation metrics across runs
4. Generating comparison tables and plots
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class AblationAnalyzer:
    """Analyze and compare LoRA ablation studies."""
    
    def __init__(self, base_output_dir: Path | str = "outputs"):
        """Initialize analyzer.
        
        Args:
            base_output_dir: Root directory containing experiment outputs.
        """
        self.base_output_dir = Path(base_output_dir)
        self.experiments = []
        
    def load_experiment(self, exp_dir: Path | str) -> dict:
        """Load and parse a single experiment.
        
        Args:
            exp_dir: Path to experiment directory (e.g., outputs/cinema_classification/federated/...)
            
        Returns:
            Dict with experiment metadata and metrics.
        """
        exp_dir = Path(exp_dir)
        
        if not exp_dir.exists():
            logger.warning(f"Experiment directory not found: {exp_dir}")
            return {}
        
        exp_data = {"dir": str(exp_dir), "config": {}, "metrics": {}}
        
        # Load config.yaml
        config_file = exp_dir / "config.yaml"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                exp_data["config"] = self._extract_lora_config(config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
        
        # Load metrics from federated_server directory
        server_dir = exp_dir / "federated_server"
        if server_dir.exists():
            # Try to find test_metrics.json or best_metrics.json
            for metrics_file in ["test_metrics.json", "best_metrics.json", "val_metrics.json"]:
                metrics_path = server_dir / metrics_file
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r") as f:
                            exp_data["metrics"][metrics_file.replace(".json", "")] = json.load(f)
                    except Exception as e:
                        logger.debug(f"Failed to load {metrics_file}: {e}")
        
        # Load per-client metrics if available
        for client_dir in server_dir.glob("client_*"):
            client_id = client_dir.name
            metrics_path = client_dir / f"metrics_{client_id}_*.json"
            matching_files = list(metrics_path.parent.glob(metrics_path.name))
            if matching_files:
                try:
                    with open(matching_files[0], "r") as f:
                        client_metrics = json.load(f)
                        # Extract best epoch metrics if available
                        if isinstance(client_metrics, dict) and "best" in client_metrics:
                            exp_data["metrics"][f"{client_id}_best"] = client_metrics["best"]
                except Exception as e:
                    logger.debug(f"Failed to load client metrics from {matching_files[0]}: {e}")
        
        return exp_data
    
    def _extract_lora_config(self, config: dict) -> dict:
        """Extract LoRA-relevant config parameters.
        
        Args:
            config: Full configuration dict.
            
        Returns:
            Dict with LoRA configuration parameters.
        """
        lora_config = {}
        
        # LoRA parameters
        if "lora" in config:
            lora = config["lora"]
            lora_config["rank"] = lora.get("rank", None)
            lora_config["alpha"] = lora.get("alpha", None)
            lora_config["dropout"] = lora.get("dropout", None)
            lora_config["target_modules"] = lora.get("target_modules", None)
            blocks = lora.get("blocks_to_lora", None)
            lora_config["blocks_to_lora"] = blocks if blocks else "all"
            lora_config["n_blocks"] = len(blocks) if blocks and isinstance(blocks, list) else 12
        
        # Dual LoRA parameters
        if "dual_lora" in config and config["dual_lora"].get("enabled", False):
            dual = config["dual_lora"]
            lora_config["dual_lora"] = True
            lora_config["global_rank"] = dual.get("global_rank", None)
            lora_config["local_rank"] = dual.get("local_rank", None)
            blocks = dual.get("blocks_to_lora", None)
            lora_config["dual_blocks_to_lora"] = blocks if blocks else "all"
        else:
            lora_config["dual_lora"] = False
        
        # Task and dataset
        lora_config["task"] = config.get("task", "unknown")
        lora_config["data_name"] = config.get("data", {}).get("name", "unknown")
        
        return lora_config
    
    def calculate_parameter_estimate(self, n_blocks: int | str, rank: int, alpha: float) -> dict:
        """Estimate LoRA parameter counts based on configuration.
        
        Args:
            n_blocks: Number of blocks with LoRA adapters ("all" for 12, or int count).
            rank: LoRA rank.
            alpha: LoRA alpha scaling.
            
        Returns:
            Dict with parameter estimates.
        """
        # ConvViT-B attention dimensions
        hidden_dim = 768  # ViT-B hidden dimension
        n_heads = 12  # ViT-B number of heads
        
        if isinstance(n_blocks, str):
            n_blocks_int = 12  # Assume full model for "all"
        else:
            n_blocks_int = n_blocks
        
        # Per attention module (Q, KV): 2 LoRA adapters per block
        # Each LoRA: (A: rank × hidden_dim) + (B: hidden_dim × rank)
        params_per_lora = rank * hidden_dim + hidden_dim * rank  # 2 * rank * hidden_dim
        
        # 2 modules per attention block (Q and KV)
        params_per_block = 2 * params_per_lora
        
        # Total LoRA parameters
        lora_params = n_blocks_int * params_per_block
        
        # Add prediction head (approximate)
        pred_head_params = hidden_dim * 100  # Rough estimate
        
        total_trainable = lora_params + pred_head_params
        
        # Full model size (approximate)
        full_model_params = 86_000_000  # ConvViT-B approximate
        
        return {
            "lora_params": lora_params,
            "pred_head_params": pred_head_params,
            "total_trainable": total_trainable,
            "full_model_params": full_model_params,
            "trainable_pct": 100 * total_trainable / full_model_params,
        }
    
    def load_experiments_from_directory_tree(self, base_dir: Path | str) -> list[dict]:
        """Recursively load all experiments from a directory tree.
        
        Args:
            base_dir: Base directory to search (e.g., outputs/).
            
        Returns:
            List of experiment dicts.
        """
        base_dir = Path(base_dir)
        experiments = []
        
        # Look for config.yaml files to identify experiment directories
        for config_file in base_dir.rglob("config.yaml"):
            exp_dir = config_file.parent
            
            # Only process if it's a federated learning directory
            if (exp_dir / "federated_server").exists():
                logger.info(f"Loading experiment from {exp_dir}")
                exp_data = self.load_experiment(exp_dir)
                
                if exp_data["config"]:
                    # Add parameter estimates
                    n_blocks = exp_data["config"].get("n_blocks", 12)
                    rank = exp_data["config"].get("rank", 4)
                    alpha = exp_data["config"].get("alpha", 8)
                    exp_data["params"] = self.calculate_parameter_estimate(n_blocks, rank, alpha)
                    
                    experiments.append(exp_data)
        
        self.experiments = experiments
        return experiments
    
    def generate_ablation_table(self) -> pd.DataFrame:
        """Generate comparison table of ablation experiments.
        
        Returns:
            DataFrame with experiment parameters and best metrics.
        """
        rows = []
        
        for exp in self.experiments:
            row = {
                "experiment": Path(exp["dir"]).name,
                "blocks_to_lora": exp["config"].get("blocks_to_lora", "N/A"),
                "n_blocks": exp["config"].get("n_blocks", 12),
                "rank": exp["config"].get("rank", "N/A"),
                "alpha": exp["config"].get("alpha", "N/A"),
                "dual_lora": exp["config"].get("dual_lora", False),
                "lora_params": exp["params"].get("lora_params", "N/A") if "params" in exp else "N/A",
                "trainable_pct": f"{exp['params'].get('trainable_pct', 0):.2f}%" if "params" in exp else "N/A",
            }
            
            # Add best metrics if available
            metrics = exp.get("metrics", {})
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        # Extract best metric from dict
                        for metric_name, metric_val in value.items():
                            if isinstance(metric_val, (int, float)):
                                row[f"{key}_{metric_name}"] = f"{metric_val:.4f}"
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def print_summary(self, output_file: Path | str | None = None) -> str:
        """Print ablation study summary.
        
        Args:
            output_file: Optional file to save summary.
            
        Returns:
            Formatted summary string.
        """
        df = self.generate_ablation_table()
        
        summary_lines = [
            "=" * 100,
            "LoRA Ablation Study Summary",
            "=" * 100,
            "",
            df.to_string(),
            "",
            "=" * 100,
            f"Analyzed {len(self.experiments)} experiments",
            "",
        ]
        
        summary = "\n".join(summary_lines)
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(summary)
            logger.info(f"Summary saved to {output_file}")
        
        return summary
    
    def save_csv_report(self, output_file: Path | str) -> None:
        """Save ablation results as CSV.
        
        Args:
            output_file: Path to save CSV.
        """
        df = self.generate_ablation_table()
        df.to_csv(output_file, index=False)
        logger.info(f"CSV report saved to {output_file}")


def main():
    """Main entry point for ablation analysis."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Get output directory from command line or use default
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
    
    # Create analyzer and load experiments
    analyzer = AblationAnalyzer()
    logger.info(f"Loading experiments from {output_dir}...")
    experiments = analyzer.load_experiments_from_directory_tree(output_dir)
    
    if not experiments:
        logger.warning(f"No experiments found in {output_dir}")
        return
    
    # Print summary to console
    summary = analyzer.print_summary()
    print(summary)
    
    # Save reports
    report_dir = output_dir / "ablation_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = report_dir / "ablation_summary.txt"
    analyzer.print_summary(summary_file)
    
    csv_file = report_dir / "ablation_comparison.csv"
    analyzer.save_csv_report(csv_file)


if __name__ == "__main__":
    main()
