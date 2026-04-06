from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, Result, Strategy 
import json
from omegaconf import OmegaConf

from models.cinema.cinema.convvit import ConvViT

from flwr_cinema.scripts.lora import (
    apply_lora_to_convvit,
    apply_dual_lora_to_convvit,
    get_global_lora_and_head_state_dict,
    get_global_lora_state_dict,
    get_lora_and_head_state_dict,
    get_lora_state_dict,
    get_trainable_state_dict,
    has_dual_lora

)
from flwr_cinema.utils.hydra_utils import load_hydra_config
from flwr_cinema.strategies.common import get_primary_metric_name
from models.cinema.cinema.device import get_amp_dtype_and_device

# Create ServerApp
app = ServerApp()

def get_model(config: DictConfig) -> torch.nn.Module:
    """Initialize model with optional LoRA (single or dual) using CineMA pattern."""

    # Load pretrained model: freeze backbone unless doing full finetuning
    freeze_backbone = config.model.get('freeze_pretrained', True) or config.train.use_lora
    
    # logger.info(f"Loading pretrained model (freeze_backbone={freeze_backbone})...")
    model = ConvViT.from_pretrained(config=config, freeze=freeze_backbone)

    # Apply LoRA if enabled (using lora.blocks_to_lora for both standard and dual LoRA)
    blocks_to_lora = getattr(config.lora, 'blocks_to_lora', None)
    
    if config.train.use_lora:
        # Check if Dual LoRA is enabled
        if getattr(config, 'dual_lora', None) and getattr(config.dual_lora, 'enabled', False):
            # logger.info("Applying Dual LoRA adapters (personalized federated learning)...")
            apply_dual_lora_to_convvit(
                model=model,
                global_rank=config.dual_lora.global_rank,
                local_rank=config.dual_lora.local_rank,
                alpha=config.dual_lora.alpha,
                dropout=config.dual_lora.dropout,
                target_modules=config.dual_lora.target_modules,
                blocks_to_lora=blocks_to_lora,
            )
        else:
            # logger.info("Applying standard LoRA adapters...")
            apply_lora_to_convvit(
                model=model,
                rank=config.lora.rank,
                alpha=config.lora.alpha,
                dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
                blocks_to_lora=blocks_to_lora,
            )
    elif config.model.get('freeze_pretrained', False):
        # Head-only training: unfreeze classification head while keeping backbone frozen
        # logger.info("Head-only training mode: unfreezing classification head...")
        if hasattr(model, 'pred_head_dict'):
            for head_param in model.pred_head_dict.parameters():
                head_param.requires_grad = True

    return model

def get_model_params(config: DictConfig, model: torch.nn.Module) -> dict:
        """Get parameters to send to clients.
        
        Supports combining three dimensions:
        - LoRA modes: Dual LoRA / Standard LoRA / No LoRA
        - Head aggregation: aggregate_head=True (sync head) or False (keep locally trained)
        - Backbone: freeze_pretrained (frozen) or not (trainable)
        """

        # logger.info("Preparing model parameters to send to clients")
        if config.train.use_lora:
            if has_dual_lora(model):
                if config.federated.aggregate_head:
                        # logger.info(f"  Using Dual LoRA + aggregated head - syncing global LoRA + head (local LoRA stays on client)")
                    return get_global_lora_and_head_state_dict(model)
                else:
                    # logger.info(f"  Using Dual LoRA + local head - syncing only global LoRA (head stays locally trained)")
                    return get_global_lora_state_dict(model)
            else:
                if config.federated.aggregate_head:
                    # logger.info(f"  Using standard LoRA + aggregated head - syncing LoRA + head")
                    return get_lora_and_head_state_dict(model)
                else:
                    # logger.info(f"  Using standard LoRA + local head - syncing only LoRA (head stays locally trained)")
                    return get_lora_state_dict(model)
        elif config.model.get('freeze_pretrained', False):
            # logger.info(f"  Using frozen backbone mode - syncing only trainable parameters (head)")
            return get_trainable_state_dict(model)
        else:
            # logger.info(f"  Using full model mode - syncing all parameters")
            return model.state_dict()

def load_arrays_into_model(
    model: torch.nn.Module,
    arrays: 'ArrayRecord',
    config: 'DictConfig',
) -> None:
    """Load aggregated ArrayRecord parameters back into the global model.
    
    Handles LoRA (single/dual), frozen backbone, and full model modes.
    Used by both the strategy mixin (for checkpoints) and save_final_model().
    
    Args:
        model: The global model to update.
        arrays: Aggregated ArrayRecord from Flower.
        config: Hydra DictConfig.
    """
    state_dict = arrays.to_torch_state_dict()
    if config.train.use_lora:
        if has_dual_lora(model):
            from flwr_cinema.scripts.lora import load_global_lora_and_head_state_dict
            load_global_lora_and_head_state_dict(model, state_dict)
        else:
            from flwr_cinema.scripts.lora import load_lora_and_head_state_dict
            load_lora_and_head_state_dict(model, state_dict)
    elif config.model.get("freeze_pretrained", False):
        from flwr_cinema.scripts.lora import load_trainable_state_dict
        load_trainable_state_dict(model, state_dict)
    else:
        model.load_state_dict(state_dict)


def get_strategy(config: 'DictConfig', global_model: torch.nn.Module, save_dir: Path) -> Strategy:
    """Get federated learning strategy based on config.
    
    Creates the strategy with all CinemaStrategyMixin parameters wired from
    the Hydra config, including the global model reference for checkpointing.
    """

    fed_strategy = getattr(config.federated, "strategy", "fedavg").lower()
    if fed_strategy == "fedavg":
        from flwr_cinema.strategies.fedAvg import FedStrategy
    else:
        raise ValueError(f"Unknown strategy: {fed_strategy}")

    # Build the load function with the server-side config
    def _load_arrays_fn(model, arrays):
        load_arrays_into_model(model, arrays, config)

    return FedStrategy(
        # CinemaStrategyMixin params
        eval_steps=config.train.eval_steps,
        primary_metric_name=get_primary_metric_name(config),
        early_stopping_patience=config.train.early_stopping.patience,
        early_stopping_min_delta=config.train.early_stopping.min_delta,
        global_model=global_model,
        load_arrays_fn=_load_arrays_fn,
        save_dir=save_dir,
        max_n_ckpts=getattr(config.train, "max_n_ckpts", 5),
        aggregate_head=getattr(config.federated, "aggregate_head", True),
        # FedAvg params can be added here as needed, e.g.:
        # fraction_train=config.federated.fraction_train,
        arrayrecord_key="model_params",
        weighted_by_key="num_samples", 
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
    
    Loads the final aggregated parameters back into the global model,
    then saves a comprehensive checkpoint with:
    - Full model state_dict (backbone + aggregated LoRA/head)
    - Separate LoRA weights (if applicable)
    - Per-client local LoRA or unaggregated heads (if available in strategy)
    - All training and evaluation metrics from the Flower Result
    - The Hydra config used for this run
    
    Args:
        save_dir: Directory to save checkpoint files.
        model: The global model (will be updated with final aggregated params).
        result: Flower Result object containing final arrays and metrics history.
        config: Hydra DictConfig used for this run.
        n_rounds: Number of rounds executed.
        strategy: The strategy instance used, to extract client-specific states.
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

    # Save checkpoint
    ckpt_path = save_dir / f"global_model_final_round_{n_rounds}.pt"
    torch.save(checkpoint, ckpt_path)

    # Save metrics history as JSON for analysis 
    metrics_history = {
        "train_metrics": {
            str(k): dict(v) for k, v in result.train_metrics_clientapp.items()
        },
        "evaluate_metrics": {
            str(k): dict(v) for k, v in result.evaluate_metrics_clientapp.items()
        },
        "evaluate_metrics_serverapp": {
            str(k): dict(v) for k, v in result.evaluate_metrics_serverapp.items()
        },
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
    
    target_column = getattr(config.data, "class_column", None) or getattr(config.data, "regression_column", "unknown")
    base_output_dir = Path("outputs") / (
        f"{config.model_name}_{config.task}/"
        f"{config.mode}/"
        f"{config.data.name}/{target_column}/"
        f"lora_{config.train.use_lora}/seed{config.seed}/"
        f"{timestamp_str}"
    )
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
        }),
        num_rounds=n_rounds
        # evaluate_fn=global_evaluate  # TODO: implement global evaluation function that evaluates on central test set after each round
    )
    
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
