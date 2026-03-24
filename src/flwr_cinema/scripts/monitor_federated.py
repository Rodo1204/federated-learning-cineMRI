"""Real-time monitoring dashboard for federated training."""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_header(text: str, char: str = "=") -> None:
    """Print formatted header."""
    print(f"\n{char * 80}")
    print(f"{text.center(80)}")
    print(f"{char * 80}")


def format_metric(value, name: str = "") -> str:
    """Format metric value for display."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        if isinstance(value, (int, float)):
            return f"{value:.4e}" if abs(value) < 0.01 and value != 0 else f"{value:.4f}"
        return str(value)
    except:
        return str(value)


def print_metrics(metrics: dict, title: str) -> None:
    """Print metrics in formatted table."""
    if not metrics:
        return
    
    print(f"\n{title}:")
    for key, value in metrics.items():
        if not key.startswith("_"):
            formatted = format_metric(value)
            print(f"  {key:.<30} {formatted:>20}")


def monitor_federated_run(output_dir: str | Path, update_interval: int = 5) -> None:
    """Monitor federated training in real-time.
    
    Args:
        output_dir: Directory containing aggregated_metrics_history.json
        update_interval: Update interval in seconds
    """
    output_path = Path(output_dir)
    metrics_file = output_path / "aggregated_metrics_history.json"
    
    print_header("FEDERATED TRAINING MONITOR", "=")
    print(f"Monitoring: {metrics_file}")
    print(f"Update interval: {update_interval}s")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    last_round = -1
    metrics_history = []
    
    try:
        while True:
            try:
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        lines = f.readlines()
                    
                    if lines:
                        # Parse all metrics to check for new rounds
                        current_metrics = []
                        for line in lines:
                            try:
                                current_metrics.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                        
                        if current_metrics:
                            latest = current_metrics[-1]
                            current_round = latest.get("round", -1)
                            
                            # New round detected
                            if current_round > last_round:
                                last_round = current_round
                                metrics_history = current_metrics
                                
                                print_header(f"Round {current_round}", "-")
                                
                                # FedAvg metrics (weighted by validation samples)
                                fedavg = latest.get("fedavg_metrics", {})
                                print_metrics(fedavg, "FedAvg (weighted by validation samples)")
                                
                                # Macro metrics (unweighted)
                                macro = latest.get("macro_metrics", {})
                                print_metrics(macro, "Macro (unweighted average)")
                                
                                # Step information
                                print(f"\nOptimization Steps:")
                                print(f"  This round:       {latest.get('optimizer_steps_this_round', 'N/A'):>20}")
                                print(f"  Total:            {latest.get('total_optimizer_steps', 'N/A'):>20}")
                                print(f"  Gradient accum:   {latest.get('n_accum_steps', 'N/A'):>20}")
                                print(f"  Local epochs:     {latest.get('local_epochs', 'N/A'):>20}")
                                
                                # Per-client breakdown
                                if "client_steps" in latest and latest["client_steps"]:
                                    print(f"\nPer-Client Step Breakdown:")
                                    print(f"  {'Client':<25} {'Samples':>15} {'Steps/Round':>15}")
                                    print(f"  {'-' * 25} {'-' * 15} {'-' * 15}")
                                    for client_id, steps_info in latest["client_steps"].items():
                                        n_samples = steps_info.get("n_train_samples", "N/A")
                                        n_steps = steps_info.get("optimizer_steps_this_round", "N/A")
                                        print(f"  {client_id:<25} {n_samples:>15} {n_steps:>15}")
                                
                                # Progress summary
                                print(f"\nProgress:")
                                print(f"  Round completed at: {datetime.now().strftime('%H:%M:%S')}")
                                if "round" in latest:
                                    print(f"  Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # Show periodic updates (every 10s)
                            else:
                                elapsed = datetime.now().strftime('%H:%M:%S')
                                print(f"[{elapsed}] Still processing Round {current_round}...", end='\r')
            
            except json.JSONDecodeError as e:
                logger.warning(f"Error decoding metrics JSON: {e}")
            except Exception as e:
                logger.warning(f"Error reading metrics: {e}")
            
            time.sleep(update_interval)
    
    except KeyboardInterrupt:
        print_header("Monitoring stopped by user", "=")
        
        if metrics_history:
            print(f"\nFinal Statistics (from {len(metrics_history)} rounds):")
            print(f"  Total rounds completed: {last_round + 1}")
            print(f"  Total optimization steps: {metrics_history[-1].get('total_optimizer_steps', 'N/A')}")
            
            # Compute best validation metrics
            best_fedavg_mae = min(m.get("fedavg_metrics", {}).get("mae", float('inf')) for m in metrics_history if "fedavg_metrics" in m)
            best_macro_mae = min(m.get("macro_metrics", {}).get("mae", float('inf')) for m in metrics_history if "macro_metrics" in m)
            
            print(f"\nBest Metrics Achieved:")
            print(f"  FedAvg MAE (weighted):  {format_metric(best_fedavg_mae)}")
            print(f"  Macro MAE (unweighted): {format_metric(best_macro_mae)}")
        
        print(f"\nMetrics file location: {metrics_file}")
        sys.exit(0)


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-time monitoring for federated learning runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor specific output directory
  python scripts/monitor_federated.py --output_dir outputs/federated_20240203_120000
  
  # Monitor with custom update interval
  python scripts/monitor_federated.py --output_dir outputs/federated_20240203_120000 --update_interval 2
        """
    )
    
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to federated output directory containing aggregated_metrics_history.json"
    )
    parser.add_argument(
        "--update_interval",
        type=int,
        default=5,
        help="Update interval in seconds (default: 5)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    if not output_path.exists():
        print(f"Error: Output directory not found: {output_path}")
        sys.exit(1)
    
    monitor_federated_run(output_path, args.update_interval)


if __name__ == "__main__":
    main()
