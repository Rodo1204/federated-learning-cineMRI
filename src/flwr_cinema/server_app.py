"""Flower ServerApp entry point for federated training.

Bootstraps the global model, resolves the aggregation strategy from config,
runs the federated loop, and saves the final model with metrics.
"""
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, Result, Strategy 
import json
from omegaconf import OmegaConf

from flwr_cinema.scripts.lora import (
    get_lora_policy,
    get_lora_state_dict,  # still needed for save_final_model checkpoint
)
from flwr_cinema.utils.hydra_utils import load_hydra_config
from flwr_cinema.strategies.common import get_primary_metric_name
from flwr_cinema.tasks.adapters import get_federated_task_adapter
from models.cinema.cinema.device import get_amp_dtype_and_device

# Create ServerApp
app = ServerApp()

def get_model(config: DictConfig) -> torch.nn.Module:
    """Build model using the task adapter model factory."""
    adapter = get_federated_task_adapter(config)
    return adapter.build_model(config)

def get_model_params(config: DictConfig, model: torch.nn.Module) -> dict:
    """Get parameters to send to clients.

    Uses get_lora_policy to resolve the correct extractor based on model
    architecture, LoRA flavour, and federated.aggregate_head. Full fine-tuning
    returns the full state_dict regardless of aggregate_head.
    """
    if not config.train.use_lora and not config.model.get("freeze_pretrained", False):
        return model.state_dict()
    
    agg_head = getattr(config.federated, "aggregate_head", True)
    policy = get_lora_policy(
        model,
        use_lora=config.train.use_lora,
        freeze_pretrained=config.model.get("freeze_pretrained", False),
        aggregate_head=agg_head,
    )
    return policy.get(model)

def load_arrays_into_model(
    model: torch.nn.Module,
    arrays: 'ArrayRecord',
    config: 'DictConfig',
) -> None:
    """Load aggregated ArrayRecord parameters back into the global model.

    Handles LoRA (single/dual), frozen backbone, and full model modes.
    Used by both the strategy mixin (for checkpoints) and save_final_model.
    """
    state_dict = arrays.to_torch_state_dict()
    if not config.train.use_lora and not config.model.get("freeze_pretrained", False):
        model.load_state_dict(state_dict)
        return
    agg_head = getattr(config.federated, "aggregate_head", True)
    policy = get_lora_policy(
        model,
        use_lora=config.train.use_lora,
        freeze_pretrained=config.model.get("freeze_pretrained", False),
        aggregate_head=agg_head,
    )
    policy.load(model, state_dict)


def get_strategy(
    config: 'DictConfig',
    global_model: torch.nn.Module,
    save_dir: Path,
) -> Strategy:
    """Resolve and instantiate the aggregation strategy from config.

    Wires all CinemaStrategyMixin parameters from Hydra config, including
    the global model reference for checkpointing.
    """

    agg = getattr(config.federated, "aggregation_method", None)
    agg_name = getattr(agg, "name", "fedavg").lower()
    agg_config = getattr(agg, "config", None)
    
    kwargs = {
        "arrayrecord_key": "model_params",
        "weighted_by_key": "num_samples",  # default, overridable per strategy
    }
    if agg_name == "fedavg":
        from flwr_cinema.strategies.fedAvg import FedAvgStrategy as FedStrategy
    elif agg_name == "fedce":
        from flwr_cinema.strategies.fedCE import FedCEStrategy as FedStrategy
        kwargs.update({
            "combination_method": getattr(agg_config, "combination_method", "multiplication"),
        })
    elif agg_name == "fedratemylora":
        from flwr_cinema.strategies.fedRateMyLoRA import RateMyLoraStrategy as FedStrategy
        import math
        kwargs.update({
            "lambda_init": getattr(agg_config, "lambda_init", 0.2),
            "lambda_decay": getattr(agg_config, "lambda_decay", 0.95),
        })
        n_warmup_epochs = getattr(config.train, "n_warmup_epochs", 0)
        local_epochs = getattr(config.federated, "local_epochs", 1)
        kwargs["warmup_rounds"] = math.ceil(n_warmup_epochs / max(local_epochs, 1))
    elif agg_name == "fedprox":
        from flwr_cinema.strategies.fedProx import FedProxStrategy as FedStrategy
        kwargs.update({
            "proximal_mu": getattr(agg_config, "proximal_mu", 0.1),
        })
    else:
        raise ValueError(f"Unknown aggregation method: {agg_name}")

    # Build the load function with the server-side config
    def _load_arrays_fn(model, arrays):
        load_arrays_into_model(model, arrays, config)

    return FedStrategy(
        # CinemaStrategyMixin params
        eval_steps=config.train.eval_steps,
        primary_metric_name=get_primary_metric_name(config),
        primary_metric_direction=config.train.early_stopping.mode,
        early_stopping_patience=config.train.early_stopping.patience,
        early_stopping_min_delta=config.train.early_stopping.min_delta,
        global_model=global_model,
        load_arrays_fn=_load_arrays_fn,
        save_dir=save_dir,
        max_n_ckpts=getattr(config.train, "max_n_ckpts", 5),
        aggregate_head=getattr(config.federated, "aggregate_head", True),
        aggregate_decoder=getattr(config.federated, "aggregate_decoder", True),
        # FedAvg params (e.g. fraction_train) can be added here as needed.
        **kwargs
    )

def save_final_model(
    save_dir: Path,
    model: torch.nn.Module,
    result: Result,
    config: DictConfig,
    n_rounds: int,
    strategy: Strategy = None,
) -> None:
    """Save the final global model checkpoint with metrics and config.

    Saves a comprehensive checkpoint: model state_dict, separate LoRA weights
    (if applicable), per-client states, metrics history as JSON, and the
    Hydra config.

    Args:
        save_dir: Directory to save checkpoint files.
        model: Global model (updated with final aggregated params).
        result: Flower Result with final arrays and metrics.
        config: Hydra DictConfig for this run.
        n_rounds: Number of rounds executed.
        strategy: Strategy instance for client-specific states.
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    # Load final aggregated params back into the global model
    load_arrays_into_model(model, result.arrays, config)

    # Build checkpoint dict
    checkpoint = {
        "n_rounds": n_rounds,
        "model": model.state_dict(),
    }

    # Save LoRA weights separately for easy extraction
    if config.train.use_lora:
        checkpoint["lora_state_dict"] = get_lora_state_dict(model)

    # Incorporate client-specific states from the strategy mixin if available
    if strategy is not None:
        if hasattr(strategy, "_latest_local_loras") and strategy._latest_local_loras:
            checkpoint["client_local_loras"] = {
                cid: rec.to_torch_state_dict() for cid, rec in strategy._latest_local_loras.items()
            }
        
        if hasattr(strategy, "_latest_client_heads") and strategy._latest_client_heads:
            checkpoint["client_heads"] = {
                cid: rec.to_torch_state_dict() for cid, rec in strategy._latest_client_heads.items()
            }

        if hasattr(strategy, "_latest_client_decoders") and strategy._latest_client_decoders:
            checkpoint["client_decoders"] = {
                cid: rec.to_torch_state_dict() for cid, rec in strategy._latest_client_decoders.items()
            }

    # Save checkpoint
    ckpt_path = save_dir / f"global_model_final_round_{n_rounds}.pt"
    torch.save(checkpoint, ckpt_path)

    train_metrics = {
        str(k): dict(v) for k, v in result.train_metrics_clientapp.items()
    }
    evaluate_metrics = {
        str(k): dict(v) for k, v in result.evaluate_metrics_clientapp.items()
    }
    evaluate_metrics_serverapp = {
        str(k): dict(v) for k, v in result.evaluate_metrics_serverapp.items()
    }
    round_metadata = getattr(strategy, "round_metadata", {}) if strategy is not None else {}

    all_rounds = sorted(
        {
            *train_metrics.keys(),
            *evaluate_metrics.keys(),
            *evaluate_metrics_serverapp.keys(),
            *round_metadata.keys(),
        },
        key=lambda round_key: int(round_key),
    )

    rounds = []
    for round_key in all_rounds:
        metadata = round_metadata.get(round_key, {})
        rounds.append({
            "round": int(round_key),
            "validation_round": round_key in evaluate_metrics or round_key in evaluate_metrics_serverapp,
            "participants": metadata.get("participants", []),
            "client_steps": metadata.get("client_steps", {}),
            "total_optimizer_steps": metadata.get("total_optimizer_steps"),
            "train_metrics": train_metrics.get(round_key, {}),
            "evaluate_metrics": evaluate_metrics.get(round_key, {}),
            "evaluate_metrics_serverapp": evaluate_metrics_serverapp.get(round_key, {}),
        })

    # Save metrics history as JSON for analysis 
    metrics_history = {
        # "train_metrics": train_metrics, # Optional separated data but can be redundant with the rounds structure 
        # "evaluate_metrics": evaluate_metrics,
        # "evaluate_metrics_serverapp": evaluate_metrics_serverapp,
        # "round_metadata": round_metadata, 
        "rounds": rounds,
    }
    metrics_path = save_dir / "metrics_history.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2, default=float)
    # TODO: consider overwriting the config file in case it has change during training



@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    config = load_hydra_config(context)
    torch.manual_seed(config.seed)

    timestamp_date = datetime.now().strftime("%Y-%m-%d")
    timestamp_time = datetime.now().strftime("%H-%M-%S")
    timestamp_str = f"{timestamp_date}/{timestamp_time}"
    
    if getattr(config, "task", "classification") == "classification":
        target_column = getattr(config.data, "class_column", "classification")
    elif getattr(config, "task", "regression") == "regression":
        target_column = getattr(config.data, "regression_column", "regression")
    else:
        target_column = getattr(config, "task", "unknown")
    base_output_dir = Path("outputs") / (
        f"{config.model_name}_{config.task}/"
        f"{config.mode}/"
        f"{config.data.name}/{target_column}/"
        f"lora_{config.train.use_lora}/seed{config.seed}/"
        f"{timestamp_str}"
    )  # TODO: probar enviar mejor el path completo del hydra que lee el server
    base_output_dir.mkdir(parents=True, exist_ok=True)

    n_rounds = getattr(config, "n_rounds", None) or config.federated.n_rounds
    save_dir = base_output_dir / "federated_server"
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save config separately as YAML for easy reference
    config_path = save_dir / "config.yaml"
    OmegaConf.save(config, config_path)

    # Initialize global model
    # logger.info("Initializing global model...")
    _, device = get_amp_dtype_and_device()
    global_model = get_model(config)
    global_model.to(device)
    arrays = ArrayRecord.from_torch_state_dict(get_model_params(config, global_model))

    strategy = get_strategy(config, global_model, save_dir)
    
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({
            "federated_timestamp": timestamp_str,
            "train.lr": config.train.lr,
            "train.weight_decay": config.train.weight_decay,
            "train.clip_grad": config.train.clip_grad,
            "federated.local_epochs": config.federated.local_epochs,
            "federated.aggregate_head": getattr(config.federated, "aggregate_head", True),
            "federated.aggregate_decoder": getattr(config.federated, "aggregate_decoder", True),
            "strategy": getattr(config.federated.aggregation_method, "name", "fedavg").lower(),
        }),
        evaluate_config=ConfigRecord({
            "federated_timestamp": timestamp_str,
        }),
        num_rounds=n_rounds
        # evaluate_fn=global_evaluate  # TODO: implement global evaluation function that evaluates on central test set after each round
    )

    # Rate-My-LoRA: swap preliminary model with re-aggregated final model
    if hasattr(strategy, 'get_final_arrays'):
        final = strategy.get_final_arrays()
        if final is not None:
            result.arrays = final

    save_final_model(save_dir, global_model, result, config, n_rounds, strategy)


# def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
#     """Evaluate model on central data."""

#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(arrays.to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load entire test set
#     test_dataloader = load_centralized_dataset()

#     # Evaluate the global model on the test set
#     test_loss, test_acc = test(model, test_dataloader, device)

#     # Return the evaluation metrics
#     return MetricRecord({"accuracy": test_acc, "loss": test_loss})
