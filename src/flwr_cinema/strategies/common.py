import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
import torch.nn as nn
from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid

class MetricTargetReachedException(Exception):
    """Custom exception to signal that the target metric has been reached for early stopping."""
    pass

def get_primary_metric_name(config) -> str:
    """Get the primary metric name for early stopping from config.
    
    Args:
        config: Configuration with config.train.early_stopping.metric field.
        
    Returns:
        Primary metric name (e.g., 'mae' for regression, 'accuracy' for classification).
    """
    # Read metric from config to align with centralized training
    if hasattr(config, "train") and hasattr(config.train, "early_stopping") and hasattr(config.train.early_stopping, "metric"):
        return config.train.early_stopping.metric
    
    # Fallback: derive from task if metric not specified in config
    task = getattr(config, "task", "regression")
    if task == "classification":
        return "accuracy"
    else:
        return "mae"
    
def weighted_average_metrics(
    metrics_records: list[RecordDict], weight_key: str = "num_samples"
) -> MetricRecord:
    """Aggregate validation metrics from clients using FedAvg (weighted average).
    
    This replaces aggregate_metrics() from the old server_app.py.
    """
    if not metrics_records:
        return MetricRecord()
    total_samples = 0.0
    weighted_metrics = {}
    for record in metrics_records:
        metrics = record.get("metrics", MetricRecord())
        
        # Get weight (number of validation samples from this client)
        weight = float(metrics.get(weight_key, 0.0))
        if weight == 0.0:
            continue
            
        total_samples += weight
        # Accumulate weighted metrics
        for key, value in metrics.items():
            # Skip non-numeric keys or the weight key itself
            if key in [weight_key, "num_samples", "n_val_samples"] or not isinstance(value, (int, float)):
                continue
                
            if key not in weighted_metrics:
                weighted_metrics[key] = 0.0
            weighted_metrics[key] += float(value) * weight
    # Calculate final averages
    if total_samples == 0:
        return MetricRecord()
    averaged_metrics = {
        key: value / total_samples for key, value in weighted_metrics.items()
    }
    
    # Add total samples to the final record
    averaged_metrics[f"total_{weight_key}"] = total_samples
    return MetricRecord(averaged_metrics)

class CinemaStrategyMixin:
    """Mixin class that adds step-based validation, checkpointing, and early stopping to any Flower strategy.
    
    This should be used via multiple inheritance with a Flower built-in strategy.
    Example: class FedStrategy(CinemaStrategyMixin, FedAvg): pass
    
    The mixin receives the global model by reference and a callable to load aggregated
    arrays into it. This keeps the mixin semi-agnostic: it doesn't know about LoRA/head
    specifics — the caller provides the loading logic.
    """
    
    def __init__(
        self,
        eval_steps: int,
        primary_metric_name: str,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
        global_model: nn.Module,
        load_arrays_fn: Callable[[nn.Module, ArrayRecord], None],
        save_dir: Path,
        max_n_ckpts: int = 5,
        aggregate_head: bool = True,
        **kwargs
    ):
        # Pass remaining kwargs to the Flower base strategy (FedAvg, etc.)
        super().__init__(**kwargs)
        
        self.eval_steps = eval_steps
        self.primary_metric_name = primary_metric_name
        self.total_optimizer_steps = 0
        self.last_eval_step = 0
        
        # Model reference for checkpointing (not owned by the mixin)
        self.global_model = global_model
        self.load_arrays_fn = load_arrays_fn
        self.aggregate_head = aggregate_head
        
        # Checkpoint management
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_n_ckpts = max_n_ckpts
        self.saved_ckpt_paths: list[Path] = []
        self._latest_arrays: ArrayRecord | None = None
        self._latest_local_loras: dict[str, ArrayRecord] = {}  # client_id → local LoRA ArrayRecord
        self._latest_client_heads: dict[str, ArrayRecord] = {}  # client_id → head ArrayRecord (when aggregate_head=False)
        self._eval_count = 0  # Track number of evaluations for first-eval checkpoint
        
        # Early Stopping Configuration
        metrics_to_minimize = {"mae", "mse", "rmse", "loss", "cross_entropy"}
        metrics_to_maximize = {"accuracy", "f1", "precision", "recall", "auc", "mcc", "roc_auc"}
        
        if primary_metric_name.lower() in metrics_to_minimize:
            self.metric_direction = "minimize"
            self.best_metric = float('inf')
        elif primary_metric_name.lower() in metrics_to_maximize:
            self.metric_direction = "maximize"
            self.best_metric = -float('inf')
        else:
            # logger.warning(f"Unknown metric direction for '{primary_metric_name}', defaulting to minimize")
            self.metric_direction = "minimize"
            self.best_metric = float('inf')
            
        # Initialize Early Stopping
        try:
            from models.cinema.cinema.optim import EarlyStopping
            self.early_stopping = EarlyStopping(
                min_delta=early_stopping_min_delta,
                patience=early_stopping_patience,
            )
            self.early_stopping_triggered = False
            # logger.info(
            #     f"CinemaStrategyMixin initialized: eval_steps={eval_steps}, "
            #     f"metric={primary_metric_name} ({self.metric_direction}), "
            #     f"patience={early_stopping_patience}, max_ckpts={max_n_ckpts}"
            # )
        except ImportError:
            # logger.warning("Could not import EarlyStopping from models.cinema.cinema.optim. Early stopping will be disabled.")
            self.early_stopping = None
            self.early_stopping_triggered = False

    # ---- Checkpoint Management ----

    def _save_checkpoint(self, server_round: int, metrics: MetricRecord | None) -> None:
        """Save a full model checkpoint. Called when the primary metric has improved.
        
        Loads the latest aggregated arrays into the global model (via load_arrays_fn),
        then saves the full model.state_dict() along with metadata.
        """

        if self._latest_arrays is None:
            # logger.warning("Cannot save checkpoint: no arrays available yet.")
            return
        
        # Load aggregated parameters into the global model
        self.load_arrays_fn(self.global_model, self._latest_arrays)
        
        # Build checkpoint
        checkpoint = {
            "round": server_round,
            "total_optimizer_steps": self.total_optimizer_steps,
            "model": self.global_model.state_dict(),
        }
        
        if metrics is not None:
            # Convert MetricRecord to plain dict for serialization
            checkpoint["metrics"] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        
        # Include per-client local LoRA if available (Dual LoRA, not aggregated)
        if self._latest_local_loras:
            checkpoint["client_local_loras"] = {
                cid: rec.to_torch_state_dict() for cid, rec in self._latest_local_loras.items()
            }
        
        # Include per-client heads if available (when aggregate_head=False)
        if self._latest_client_heads:
            checkpoint["client_heads"] = {
                cid: rec.to_torch_state_dict() for cid, rec in self._latest_client_heads.items()
            }
        
        # Save
        ckpt_path = self.save_dir / f"ckpt_round_{server_round:03d}_steps_{self.total_optimizer_steps}.pt"
        torch.save(checkpoint, ckpt_path)
        self.saved_ckpt_paths.append(ckpt_path)
        # logger.info(
        #     f"Saved checkpoint: {ckpt_path.name} "
        #     f"({len(self.saved_ckpt_paths)}/{self.max_n_ckpts if self.max_n_ckpts > 0 else '∞'})"
        # )
        
        # Delete oldest if over limit (matches centralized train.py L339-342)
        if self.max_n_ckpts > 0 and len(self.saved_ckpt_paths) > self.max_n_ckpts:
            oldest = self.saved_ckpt_paths.pop(0)
            oldest.unlink(missing_ok=True)
            # logger.info(f"Deleted outdated checkpoint: {oldest.name}")

    # Flower Strategy Overrides

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid
    ) -> Iterable[Message]:
        """Configure the training round. Stops providing clients if early stopping was triggered."""
        if getattr(self, "early_stopping_triggered", False):
            # logger.warning(f"[{self.primary_metric_name.upper()} Early Stopping Triggered] Skipping logic for round {server_round}")
            return []
            # raise MetricTargetReachedException("Early stopping target reached.")  # Gracefully skip training rounds if we already stopped
            
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate training results and extract average optimizer steps."""
        
        # Convert to list to avoid consuming the iterator before super() can use it
        replies_list = list(replies)
        
        # 1. Extract per-client heads BEFORE aggregation (when aggregate_head=False)
        #    This prevents FedAvg from aggregating head params
        self._latest_local_loras.clear()
        self._latest_client_heads.clear()
        
        round_steps = 0
        n_clients_with_steps = 0
        for msg in replies_list:
            client_id = str(msg.metadata.src_node_id)
            
            # Strip head params from model_params if not aggregating heads
            if not self.aggregate_head:
                ar = msg.content.array_records.get("model_params")
                if ar is not None:
                    state_dict = ar.to_torch_state_dict()
                    head = {k: v for k, v in state_dict.items() if k.startswith("pred_head_dict.")}
                    if head:
                        self._latest_client_heads[client_id] = ArrayRecord.from_torch_state_dict(head)
                        no_head = {k: v for k, v in state_dict.items() if not k.startswith("pred_head_dict.")}
                        msg.content.array_records["model_params"] = ArrayRecord.from_torch_state_dict(no_head)
            
            # Extract optimizer steps (average, to match centralized)
            if "optimizer_steps_this_round" in msg.content.config_records.get("config", {}):
                steps = msg.content.config_records["config"]["optimizer_steps_this_round"]
                if steps > 0:
                    round_steps += steps
                    n_clients_with_steps += 1
            
            # Extract local LoRA (Dual LoRA, not aggregated, for archival)
            local_lora = msg.content.array_records.get("local_lora_params")
            if local_lora is not None:
                self._latest_local_loras[client_id] = local_lora
        
        # 2. Call the base strategy (FedAvg, FedAdam, etc.) — heads are already stripped
        agg_arrays, agg_metrics = super().aggregate_train(server_round, replies_list)
        
        # 3. Accumulate average optimizer steps
        if n_clients_with_steps > 0:
            avg_steps = int(round_steps / n_clients_with_steps)
            self.total_optimizer_steps += avg_steps
            
        return agg_arrays, agg_metrics
    
    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid
    ) -> Iterable[Message]:
        """Configure evaluation, skipping it if we haven't reached eval_steps."""
        if getattr(self, "early_stopping_triggered", False):
            return []
            #raise MetricTargetReachedException("Early stopping target reached.")  # Gracefully skip
            
        # Trigger evaluation if we've accumulated at least eval_steps since last time
        should_validate = (
            self.total_optimizer_steps > 0 and 
            (self.total_optimizer_steps - self.last_eval_step) >= self.eval_steps
        )
        
        if not should_validate:
            return []  # Tells Flower to skip evaluation this round

        # Update tracker
        self.last_eval_step = (self.total_optimizer_steps // self.eval_steps) * self.eval_steps 

        # Store arrays reference for checkpoint saving
        self._latest_arrays = arrays

        # Insert total steps into config so clients know it
        config["total_optimizer_steps"] = self.total_optimizer_steps
        
        # Call base strategy to select clients for evaluation
        return super().configure_evaluate(server_round, arrays, config, grid)
        
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate evaluation metrics, step EarlyStopping, and save checkpoint if improved."""
        metrics = super().aggregate_evaluate(server_round, replies)
        
        if metrics is None or self.primary_metric_name not in metrics:
            return metrics
        
        current_metric = float(metrics[self.primary_metric_name])
        self._eval_count += 1
        
        # Early Stopping Evaluation 
        if getattr(self, "early_stopping", None) is not None:
            # EarlyStopping always minimizes internally
            metric_for_early_stopping = current_metric
            if self.metric_direction == "maximize":
                metric_for_early_stopping = -current_metric
            
            self.early_stopping.update(metric_for_early_stopping)
            
            # Format best metric for display
            # display_best = self.early_stopping.best_metric
            # if self.metric_direction == "maximize" and display_best != float('inf'):
            #     display_best = -display_best
            
            # logger.info(
            #     f"[EVAL Round {server_round}] {self.primary_metric_name}: {current_metric:.4f} "
            #     f"| Best: {display_best:.4f} "
            #     f"| Patience: {self.early_stopping.patience_count}/{self.early_stopping.patience}"
            # )
            
            # Save checkpoint if improved
            # Also save on first evaluation to always have at least one checkpoint
            if self.early_stopping.has_improved or self._eval_count == 1:
                self._save_checkpoint(server_round, metrics)
            
            if self.early_stopping.should_stop:
                    # logger.warning(
                    #     f" EARLY STOPPING TRIGGERED AT ROUND {server_round} "
                    #     f"(No improvement for {self.early_stopping.patience} validation intervals)"
                    # )
                self.early_stopping_triggered = True
                    
        return metrics
