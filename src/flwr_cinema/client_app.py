import torch
from flwr.app import Context, Message, ConfigRecord
from flwr.clientapp import ClientApp
import json
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig
from models.cinema.cinema.optim import get_n_accum_steps
from torch.utils.data import DataLoader

from flwr_cinema.scripts.federated_dataset_loader import load_client_dataloaders
from flwr_cinema.tasks.adapters import get_federated_task_adapter

from flwr_cinema.utils.hydra_utils import load_hydra_config, update_config_from_config_record
from flwr.common.logger import log
import logging

# Flower ClientApp 
app = ClientApp()

# Model cache per client_id to preserve local decoder/head state across rounds
# when aggregate_decoder=False or aggregate_head=False. Each client gets its
# own model object so per-client artifacts don't leak across clients sharing
# the same Ray actor in simulation mode.
_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_OPTIM_STATE_CACHE: dict[str, dict] = {}
_TRAIN_DATALOADER_CACHE = None
_VAL_DATALOADER_CACHE = None
_STATS_CACHE = None
# def setup_logging(log_dir: Path, ):

def get_model(config: DictConfig, client_id: str | None = None) -> torch.nn.Module:
    """Build or retrieve the model for a specific federated client.
    
    Caches one model per client_id so local decoder/head parameters
    survive across rounds. Inactive client models are flushed to CPU to
    avoid exhausting GPU VRAM when many clients share the same actor.
    The active model will be moved to GPU by FederatedClient.
    """
    global _MODEL_CACHE
    
    cid = client_id or "__default__"
    if cid in _MODEL_CACHE:
        #_flush_other_clients_to_cpu(cid)
        log(logging.INFO, f"Reusing cached model for client_id: {cid}")
        return _MODEL_CACHE[cid]

    adapter = get_federated_task_adapter(config)
    model = adapter.build_model(config)
    _MODEL_CACHE[cid] = model
    # _OPTIM_STATE_CACHE.pop(cid, None)  # new model → stale optimizer state is invalid
    #_flush_other_clients_to_cpu(cid)
    log(logging.INFO, f"Created new model for client_id: {cid}")
    return model


def _flush_other_clients_to_cpu(active_client_id: str) -> None:
    """Move inactive client models to CPU to free GPU memory."""
    for cid, model in _MODEL_CACHE.items():
        if cid != active_client_id:
            model.to("cpu")

def get_client_class(config: DictConfig):
    """Resolve federated client class from task adapter."""
    return get_federated_task_adapter(config).client_class_factory()

def get_task_specific_functions(config: DictConfig):
    """Resolve task-specific loss and evaluation functions from adapter."""
    adapter = get_federated_task_adapter(config)
    task_spec = adapter.task_spec
    return task_spec.loss_fn_factory(), task_spec.eval_fn_factory()


def load_client_data(
    config: DictConfig,
    client_id: str,
    vendor: str | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    """Load training data for a specific federated learning client.
    
    This is a thin wrapper around the common load_client_dataloaders() from
    data/federated_dataset_loader.py that handles task-agnostic data loading.
    
    Terminology:
        - client_id: Federated learning participant (acdc, GE, Canon, Siemens, Philips)
        - vendor: Scanner manufacturer in metadata (only for filtered MnMs data)
    
    Args:
        config: Configuration dictionary.
        client_id: Federated client identifier (e.g., "acdc", "GE", "Canon").
                  Matches the 'client' column in combined dataset metadata.
        vendor: [DEPRECATED] Vendor to filter by. Now uses client_id directly.
        
    Returns:
        Tuple of (train_loader, val_loader, stats_dict) for the client.
    """

    global _TRAIN_DATALOADER_CACHE
    global _VAL_DATALOADER_CACHE
    global _STATS_CACHE

    # if _TRAIN_DATALOADER_CACHE is not None and _VAL_DATALOADER_CACHE is not None and _STATS_CACHE is not None:
    #     return _TRAIN_DATALOADER_CACHE, _VAL_DATALOADER_CACHE, _STATS_CACHE
    # TODO: Not sure caching works here; test without caching first.

    # logger.info(f"Loading data for client {client_id} using common task-agnostic loader...")
    
    # Use the common loader from data module
    train_loader, val_loader, stats = load_client_dataloaders(
        config=config,
        client_id=client_id,
        vendor=vendor,
        batch_size=config.train.batch_size_per_device,
        num_workers=config.train.n_workers,
    )
    
    # Log statistics for IID/non-IID check
    # logger.info(f"Client statistics: {stats}")
    
    _TRAIN_DATALOADER_CACHE = train_loader
    _VAL_DATALOADER_CACHE = val_loader
    _STATS_CACHE = stats
    return train_loader, val_loader, stats

def maybe_apply_class_weighted_loss(config: DictConfig, loss_fn, client_stats: dict):
    """Wrap the loss function with class weights if enabled in config and class distribution is available."""
    use_class_weighted_loss = getattr(config.train, "use_class_weighted_loss", False)
    if getattr(config, "task", "classification") == "classification" and use_class_weighted_loss:
        from flwr_cinema.scripts.weighted_losses import wrap_loss_with_class_weights
        class_distribution = client_stats.get("train", {}).get("class_distribution", None)
        if class_distribution:
            class_col = config.data.class_column
            classes = config.data[class_col]
            loss_fn = wrap_loss_with_class_weights(loss_fn, class_distribution, classes)
            # logger.info(f"Applied class-weighted CE with distribution: {class_distribution}")
        # else:
            # logger.info("Classification task: class_weighted_loss enabled but no class_distribution available")
    return loss_fn

    
# TODO: Refactor train and evaluate functions abstracting common code 
@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load server and hydra config
    in_config = msg.content.config_records.get("config", ConfigRecord())
    config = load_hydra_config(context)
    config = update_config_from_config_record(config, in_config)  # Update Hydra config with any overrides from server-sent config record
    
    round_idx = int(in_config.get("server-round", 0))
    torch.manual_seed(config.seed + round_idx)

    # Construct output directory using same config template structure as server
    # Use coordinated timestamp from server if available
    # Otherwise generate new timestamp
    coordinated_timestamp = in_config.get("federated_timestamp", None)
    if coordinated_timestamp:
        timestamp_str = str(coordinated_timestamp)
    else:
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
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Flower assigns numeric 'partition-id' (0, 1, 2...) automatically in simulation.
    # Map it to string dataset names (e.g. acdc, Siemens...) configured in Hydra.
    if "partition-id" in context.node_config:
        partition_id = context.node_config["partition-id"]
        # config.federated.client_ids can be a list: ["acdc", "Siemens", "GE", "Philips", "Canon"]
        client_names = getattr(config.federated, "client_ids", [])
        if partition_id < len(client_names):
            client_id = str(client_names[partition_id])
        else:
            client_id = str(partition_id)
    else:
        # Fallback to custom injected client_id if provided by custom NodeConfig
        client_id = str(context.node_config.get("client_id", "unknown"))

    vendor = getattr(config, "vendor", None)

    # Initialize model
    model = get_model(config, client_id)
    
    # Load client data
    train_loader, val_loader, client_stats = load_client_data(
        config=config,
        client_id=client_id,
        vendor=vendor,
    )

    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views

    # Calculate gradient accumulation steps for federated client
    n_accum_steps = get_n_accum_steps(
        batch_size=config.train.batch_size,
        batch_size_per_device=config.train.batch_size_per_device,
        world_size=1,  # Single GPU per client
    )

    # Get task-specific functions
    loss_fn, eval_dataloader_fn = get_task_specific_functions(config)
    # logger.info(f"Loaded task-specific functions for task: {getattr(config, 'task', 'regression')}")
    
    # Wrap loss function with class weighting if available and enabled (for classification)
    loss_fn = maybe_apply_class_weighted_loss(config, loss_fn, client_stats)

    # Dynamically load the appropriate client class based on task
    federated_client = get_client_class(config)

    # Create client-specific output directory
    client_run_dir = base_output_dir / f"client_{client_id}"
    client_run_dir.mkdir(parents=True, exist_ok=True)
    stats_save_path = str(client_run_dir / f"client_stats_{client_id}_round_{round_idx:03d}.json")
    metrics_save_path = str(client_run_dir / f"train_metrics_{client_id}_round_{round_idx:03d}.json")
    if round_idx == 0 or round_idx == 1:  # Only save client stats on first round to avoid redundancy across rounds
        with open(stats_save_path, "w") as f:
            json.dump(client_stats, f, indent=2)

    client = federated_client(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        views=views,
        local_epochs=config.federated.local_epochs,
        lr=float(in_config.get("lr", config.train.lr)),
        weight_decay=0.0 if config.train.use_lora else float(in_config.get("weight_decay", config.train.weight_decay)),
        clip_grad=float(in_config.get("clip_grad", config.train.clip_grad)) if in_config.get("clip_grad", config.train.clip_grad) is not None and float(in_config.get("clip_grad", config.train.clip_grad)) > 0 else None,
        use_lora=config.train.use_lora,
        freeze_pretrained=config.model.freeze_pretrained,
        aggregate_head=getattr(config.federated, "aggregate_head", True),
        aggregate_decoder=getattr(config.federated, "aggregate_decoder", True),
        metrics_save_path=metrics_save_path,
        stats_save_path=stats_save_path,
        eval_dataloader_fn=eval_dataloader_fn,
        config=config, # TODO: maybe client doesn't need full config, could just pass necessary fields or a subset of config related to training/evaluation
        run_dir=str(client_run_dir),
        n_accum_steps=n_accum_steps,
    )

    # Persist optimizer state across rounds (dirty optimizer).
    #for testing purposes
    # if client_id in _OPTIM_STATE_CACHE:
    #     client.optimizer.load_state_dict(_OPTIM_STATE_CACHE[client_id])

    # Call the client's train method and send the reply message
    reply = client.train(msg, context)

    # _OPTIM_STATE_CACHE[client_id] = client.optimizer.state_dict()

    return reply


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load server and hydra config
    in_config = msg.content.config_records.get("config", ConfigRecord())
    config = load_hydra_config(context)
    
    round_idx = int(in_config.get("server-round", 0))
    torch.manual_seed(config.seed + round_idx)
    
    # Construct output directory using same config template structure as server
    # Use coordinated timestamp from server if available
    # Otherwise generate new timestamp
    coordinated_timestamp = in_config.get("federated_timestamp", None)
    if coordinated_timestamp:
        timestamp_str = str(coordinated_timestamp)
    else:
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
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map Flower 'partition-id' to actual string dataset center name for validation
    if "partition-id" in context.node_config:
        partition_id = context.node_config["partition-id"]
        client_names = getattr(config.federated, "client_ids", [])
        if partition_id < len(client_names):
            client_id = str(client_names[partition_id])
        else:
            client_id = str(partition_id)
    else:
        client_id = str(context.node_config.get("client_id", "unknown"))

    vendor = getattr(config, "vendor", None)

    # Initialize model
    model = get_model(config, client_id)
    
    # Load client data
    train_loader, val_loader, client_stats = load_client_data(
        config=config,
        client_id=client_id,
        vendor=vendor,
    )
    
    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    
    # Get task-specific functions
    loss_fn, eval_dataloader_fn = get_task_specific_functions(config)
    
    # Wrap loss function with class weighting if available and enabled (for classification)
    loss_fn = maybe_apply_class_weighted_loss(config, loss_fn, client_stats)

    # Dynamically load the appropriate client class based on task
    federated_client = get_client_class(config)

    # Create client-specific output directory only if we persist eval artifacts
    client_run_dir = base_output_dir / f"client_{client_id}" / "eval_metrics"
    client_run_dir.mkdir(parents=True, exist_ok=True)
    metrics_save_path = str(client_run_dir / f"eval_metrics_{client_id}_round_{round_idx:03d}.json")
    stats_save_path = None

    client = federated_client(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        views=views,
        local_epochs=0,
        lr=0.0,
        weight_decay=0.0,
        clip_grad=None, 
        use_lora=config.train.use_lora,
        freeze_pretrained=config.model.freeze_pretrained,
        aggregate_head=getattr(config.federated, "aggregate_head", True),
        aggregate_decoder=getattr(config.federated, "aggregate_decoder", True),
        metrics_save_path=metrics_save_path, # TODO: maybe delete path parameters because evaluation metrics are only saved in eval function for now
        stats_save_path=stats_save_path,
        eval_dataloader_fn=eval_dataloader_fn,
        config=config,
        run_dir=str(client_run_dir),
        n_accum_steps=1,
    )

    # Call the client's evaluate method and send the reply message
    reply = client.evaluate(msg, context)

    if metrics_save_path and reply.content is not None:
        metrics_record = reply.content.metric_records.get("metrics")
        if metrics_record is not None:
            client_run_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "client_id": client_id,
                "round": round_idx,
                "metrics": dict(metrics_record),
            }
            with open(metrics_save_path, "w") as f:
                json.dump(payload, f, indent=2, default=float)

    return reply
