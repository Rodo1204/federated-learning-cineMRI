import argparse
from collections.abc import Callable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import torch
from torch import nn

from flwr_cinema.scripts.eval_utils import load_metadata
from flwr_cinema.scripts.lora import apply_lora_to_convvit, get_lora_state_dict
from models.cinema.cinema.train import maybe_reduce_batch_size

from models.cinema.cinema.classification.dataset import get_image_transforms
from models.cinema.cinema.convvit import ConvViT, load_pretrain_weights, param_groups_lr_decay
from models.cinema.cinema.device import get_amp_dtype_and_device, print_model_info
from models.cinema.cinema.log import get_logger, init_wandb
from models.cinema.cinema.optim import EarlyStopping, GradScaler
from models.cinema.cinema.classification.dataset import EndDiastoleEndSystoleDataset
from models.cinema.cinema.train import maybe_subset_dataset, train_one_epoch
from models.cinema.cinema.optim import get_n_accum_steps

from pathlib import Path
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from hydra.core.global_hydra import GlobalHydra
import json

logger = get_logger(__name__)


def get_task_specific_functions(config: DictConfig):
    """Load task-specific loss and eval functions based on config.task.

    Args:
        config: Configuration dictionary with config.task (classification or regression).

    Returns:
        Tuple of (loss_fn, eval_dataloader_fn) appropriate for the task.
    """
    task = getattr(config, "task", "regression")  # Default to regression for backward compatibility

    if task == "classification":
        from models.cinema.cinema.classification.train import (
            classification_loss,
            classification_eval_dataloader,
        )
        return classification_loss, classification_eval_dataloader
    else:
        from models.cinema.cinema.regression.train import (
            regression_loss,
            regression_eval_dataloader,
        )
        return regression_loss, regression_eval_dataloader


def load_combined_datasets(config: DictConfig) -> tuple[Dataset, Dataset, dict]:
    """Load and split the combined ACDC and MnMs merged dataset.

    Args:
        config: configuration file.

    Returns:
        train_dataset: dataset for training.
        val_dataset: dataset for validation (combined, for reference).
        train_class_distribution: dictionary mapping class labels to counts.
    """
    data_dir = Path(config.data.combined.dir).expanduser()

    class_col = config.data.class_column  # For classification, this is the column with class labels or binary labels
    classes = config.data[class_col]

    train_meta_df = load_metadata(data_dir / "train_metadata.csv", metadata_type="train")
    val_meta_df = load_metadata(data_dir / "val_metadata.csv", metadata_type="val")

    # Filter by specific client if specified (for client-specific centralized training)
    client_name = getattr(config.data, "client_name", None)
    if client_name and client_name.lower() != "all":
        logger.info(f"Training on client-specific data: {client_name}")
        train_meta_df = train_meta_df[train_meta_df["client"].str.lower() == client_name.lower()].reset_index(drop=True)
        val_meta_df = val_meta_df[val_meta_df["client"].str.lower() == client_name.lower()].reset_index(drop=True)
        if len(train_meta_df) == 0:
            logger.warning(f"No training samples found for client: {client_name}")
        if len(val_meta_df) == 0:
            logger.warning(f"No validation samples found for client: {client_name}")
    # Otherwise, filter by client_ids for multi-client centralized training (if specified) 
    elif hasattr(config.federated, "client_ids") and config.federated.client_ids is not None: # usar client name como lista
        logger.info(f"Filtering datasets for clients: {config.federated.client_ids}")
        train_meta_df = train_meta_df[train_meta_df["client"].isin(config.federated.client_ids)].reset_index(drop=True)
        val_meta_df = val_meta_df[val_meta_df["client"].isin(config.federated.client_ids)].reset_index(drop=True)

    train_meta_df, val_meta_df = maybe_subset_dataset(config, train_meta_df, val_meta_df)

    train_transform, val_transform = get_image_transforms(config)
    train_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "train",
        meta_df=train_meta_df,
        class_col=class_col,
        classes=classes,
        views=config.model.views,
        transform=train_transform,
    )
    val_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "val",
        meta_df=val_meta_df,
        class_col=class_col,
        classes=classes,
        views=config.model.views,
        transform=val_transform,
    )

    # Print the number of samples in each dataset per class / binary label for verification
    logger.info("Training dataset class distribution:")
    train_class_distribution = train_meta_df[class_col].value_counts().sort_index().to_dict()
    logger.info(train_class_distribution)
    logger.info("Validation dataset class distribution:")
    logger.info(val_meta_df[class_col].value_counts().to_dict())

    return train_dataset, val_dataset, train_class_distribution


def load_per_client_val_dataloaders(config: DictConfig) -> dict[str, tuple[int, torch.utils.data.DataLoader]]:
    """Load validation dataloaders split by client/dataset for federated-style aggregation.

    When training on a specific client (data.client_name != "all"), only loads that client's validation data.
    When training on all clients, loads validation data for each client separately for federated averaging.

    Args:
        config: configuration file.

    Returns:
        Dictionary mapping client_id to (n_samples, dataloader).
    """
    data_dir = Path(config.data.combined.dir).expanduser()

    class_col = config.data.class_column  # For classification, this is the column with class labels or binary labels
    classes = config.data[class_col]

    val_meta_df = load_metadata(data_dir / "val_metadata.csv", metadata_type="val")

    # Filter by specific client if specified (for client-specific centralized training)
    client_name = getattr(config.data, "client_name", None)
    if client_name and client_name.lower() != "all":
        logger.info(f"Loading validation data for specific client: {client_name}")
        val_meta_df = val_meta_df[val_meta_df["client"].str.lower() == client_name.lower()].reset_index(drop=True)
    # Otherwise, filter by client_ids for multi-client centralized training (if specified)
    elif hasattr(config.federated, "client_ids") and config.federated.client_ids is not None:
        logger.info(f"Filtering validation dataloaders for clients: {config.federated.client_ids}")
        val_meta_df = val_meta_df[val_meta_df["client"].isin(config.federated.client_ids)].reset_index(drop=True)

    val_transform = get_image_transforms(config)[1]  # Get val transform

    client_dataloaders = {}
    for client_id in val_meta_df["client"].unique():
        client_meta = val_meta_df[val_meta_df["client"] == client_id]
        # Single client case
        val_dataset = EndDiastoleEndSystoleDataset(
            data_dir=data_dir / "val",
            meta_df=client_meta.reset_index(drop=True),
            class_col=class_col,
            classes=classes,
            views=config.model.views,
            transform=val_transform,
        )
        val_sampler = SequentialSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=1,
            num_workers=config.train.n_workers,
            pin_memory=True,
            drop_last=False,
        )

        client_dataloaders[str(client_id)] = (len(val_dataset), val_loader)

    logger.info(f"Loaded validation dataloaders for clients: {list(client_dataloaders.keys())}")
    return client_dataloaders


def get_model(config: DictConfig) -> torch.nn.Module:
    """Initialize model with optional LoRA (single or dual) using CineMA pattern."""

    # Load pretrained model: freeze backbone unless doing full finetuning
    freeze_backbone = config.model.get('freeze_pretrained', True) or config.train.use_lora

    logger.info(f"Loading pretrained model (freeze_backbone={freeze_backbone})...")
    model = ConvViT.from_pretrained(config=config, freeze=freeze_backbone)

    # Apply LoRA if enabled (using lora.blocks_to_lora for both standard and dual LoRA)
    blocks_to_lora = getattr(config.lora, 'blocks_to_lora', None)

    if config.train.use_lora:
        # Check if Dual LoRA is enabled
        logger.info("Applying standard LoRA adapters...")
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
        logger.info("Head-only training mode: unfreezing classification head...")
        if hasattr(model, 'pred_head_dict'):
            for head_param in model.pred_head_dict.parameters():
                head_param.requires_grad = True

    return model


def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: GradScaler,
    n_samples: int,
    use_lora: bool = False,
) -> Path:
    """Save checkpoint with separate LoRA weights if enabled.

    Args:
        ckpt_dir: checkpoint directory.
        epoch: current epoch.
        model: model to save.
        optimizer: optimizer.
        loss_scaler: loss scaler.
        n_samples: number of samples processed.
        use_lora: whether LoRA adapters are being used.

    Returns:
        ckpt_path: checkpoint path.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ckpt_epoch_{epoch}.pt"

    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "n_samples": n_samples,
    }

    # Save LoRA weights separately if using LoRA
    if use_lora:
        to_save["lora_state_dict"] = get_lora_state_dict(model)

    torch.save(to_save, ckpt_path)
    return ckpt_path


def run_train(  # noqa: C901
    config: DictConfig,
    load_dataset: Callable[[DictConfig], tuple[Dataset, Dataset]],
    get_model_fn: Callable[[DictConfig], nn.Module],
    loss_fn: Callable[
        [nn.Module, dict[str, torch.Tensor], list[str], torch.device], tuple[torch.Tensor, dict[str, float]]
    ],
    eval_dataloader_fn: Callable[
        [
            nn.Module,
            DataLoader,
            dict[str, tuple[int, ...]],
            dict[str, tuple[float, ...]],
            torch.dtype,
            torch.device,
        ],
        dict[str, float],
    ],
    client_val_dataloaders: dict[str, tuple[int, DataLoader]] | None = None,
) -> None:
    """Main function to fine-tune.

    Args:
        config: configuration file.
        load_dataset: function to load the train and val datasets.
        get_model_fn: function to initialize the model.
        loss_fn: loss function for segmentation tasks.
        eval_dataloader_fn: evaluation function for segmentation tasks.
        client_val_dataloaders: optional dict of {client_id: (n_samples, dataloader)}
            for per-client evaluation with FedAvg aggregation. When provided and
            eval_mode=="client", evaluation is done per-client and aggregated.
            When None or eval_mode=="centralized", evaluation uses combined val_dataloader.
    """
    eval_mode = getattr(config, "eval_mode", "centralized")
    logger.info(f"Evaluation mode: {eval_mode}")

    amp_dtype, device = get_amp_dtype_and_device()
    torch.manual_seed(config.seed)

    # load dataset
    train_dataset, val_dataset, train_class_distribution = load_dataset(config=config)
    # Apply class-weighted loss for classification with imbalance
    use_class_weighted_loss = getattr(config.train, "use_class_weighted_loss", False)
    if getattr(config, "task", "regression") == "classification" and use_class_weighted_loss:
        from flwr_cinema.scripts.weighted_losses import wrap_loss_with_class_weights
        loss_fn = wrap_loss_with_class_weights(loss_fn, train_class_distribution)
        logger.info(f"Applied class-weighted BCE with distribution: {train_class_distribution}")
    elif getattr(config, "task", "regression") == "classification":
        logger.info("Classification task without class-weighted loss (disabled in config)")
    config = maybe_reduce_batch_size(config, len(train_dataset))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config.train.batch_size_per_device,
        drop_last=True,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=1,  # do not batch as each sample has different number of slices
        drop_last=False,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )

    # determine gradient accumulation
    n_accum_steps = get_n_accum_steps(
        batch_size=config.train.batch_size,
        batch_size_per_device=config.train.batch_size_per_device,
        world_size=1,
    )

    # load model
    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    model = get_model_fn(config)
    print_model_info(model)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    if config.model.ckpt_path is not None:
        model = load_pretrain_weights(
            model=model, views=views, ckpt_path=Path(config.model.ckpt_path), freeze=config.model.freeze_pretrained
        )
    model.to(device)

    # init wandb
    tags = [
        config.data.name,
        config.model.name,
        *views,
        config.task,
        f"seed{config.seed}",
        f"{int(config.data.proportion * 100)}%",
    ]
    if config.model.ckpt_path is not None:
        tags.append("finetuned")
    if hasattr(config.data, "class_column"):
        tags.append(config.data.class_column)
    if hasattr(config.data, "regression_column"):
        tags.append(config.data.regression_column)
    wandb_run, ckpt_dir = init_wandb(config=config, tags=sorted(set(tags)))

    # init optimizer
    logger.info("Initializing optimizer.")
    if not config.train.use_lora:
        param_groups = param_groups_lr_decay(
            model,
            no_weight_decay_list=[],
            weight_decay=config.train.weight_decay,
            layer_decay=config.train.layer_decay,
        )
    else:
        lora_params = []
        head_and_other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora" in name.lower():
                lora_params.append(param)
            else:
                head_and_other_params.append(param)

        param_groups = [
            {"params": head_and_other_params, "weight_decay": config.train.weight_decay},
            {"params": lora_params, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.betas)
    loss_scaler = GradScaler()

    # train
    logger.info("Start training.")
    patch_size_dict = {v: config.data.sax.patch_size if v == "sax" else config.data.lax.patch_size for v in views}
    spacing_dict = {v: config.data.sax.spacing if v == "sax" else config.data.lax.spacing for v in views}
    early_stop = EarlyStopping(
        min_delta=config.train.early_stopping.min_delta,
        patience=config.train.early_stopping.patience,
    )
    n_samples = 0
    saved_ckpt_paths = []

    forward_passes_per_epoch = len(train_dataloader)
    optimizer_steps_per_epoch = forward_passes_per_epoch // n_accum_steps

    # Get evaluation interval directly from config (in optimizer steps)
    eval_optimizer_steps = getattr(config.train, "eval_steps", 80)

    logger.info(f"Training step metrics:")
    logger.info(f" Forward passes per epoch: {forward_passes_per_epoch}")
    logger.info(f" Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    logger.info(f" Eval steps: {eval_optimizer_steps} optimizer steps")
    logger.info(f" Gradient accumulation steps: {n_accum_steps}")
    logger.info(f" → Validation triggers at optimizer steps: {eval_optimizer_steps}, {eval_optimizer_steps * 2}, {eval_optimizer_steps * 3}, ...")

    total_optimizer_steps = 0
    last_eval_step = 0

    for epoch in range(config.train.n_epochs):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{config.train.n_epochs}")
        logger.info(f"{'='*70}")
        optimizer.zero_grad(set_to_none=True)
        n_samples = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            amp_dtype=amp_dtype,
            device=device,
            epoch=epoch,
            n_accum_steps=n_accum_steps,
            n_samples=n_samples,
            config=config,
            wandb_run=wandb_run,
            loss_fn=loss_fn,
        )
        total_optimizer_steps += optimizer_steps_per_epoch

        # Align with federated scheduling: validate once we've advanced at least
        # eval_steps since the previous validation checkpoint.
        if eval_optimizer_steps <= 0:
            continue
        should_validate = (
            total_optimizer_steps > 0
            and (total_optimizer_steps - last_eval_step) >= eval_optimizer_steps
        )
        if not should_validate:
            continue
        last_eval_step = (total_optimizer_steps // eval_optimizer_steps) * eval_optimizer_steps

        # ── evaluate model ──────────────────────────────────────────────────────
        logger.info(f"Start evaluating model at step {total_optimizer_steps}.")
        model.eval()

        # Decide evaluation strategy based on eval_mode
        use_client_eval = eval_mode == "client" and client_val_dataloaders

        if use_client_eval:
            # ── per-client evaluation with FedAvg aggregation ───────────────────
            logger.info("Eval mode 'client': evaluating each client separately then aggregating with FedAvg...")
            per_client_metrics: dict[str, tuple[int, dict[str, float]]] = {}
            total_samples = 0

            with torch.no_grad():
                for client_id, (n_samples_client, val_loader) in client_val_dataloaders.items():
                    logger.info(f"  Evaluating client: {client_id} ({n_samples_client} samples)")
                    metrics = eval_dataloader_fn(
                        model,
                        val_loader,
                        patch_size_dict,
                        spacing_dict,
                        amp_dtype,
                        device,
                    )
                    per_client_metrics[client_id] = (n_samples_client, metrics)
                    total_samples += n_samples_client

            # Federated averaging: weighted sum of per-client metrics
            first_client_metrics = list(per_client_metrics.values())[0][1]
            val_metrics: dict[str, float] = {}
            macro_metrics: dict[str, float] = {}
            for metric_key in first_client_metrics.keys():
                weighted_sum = sum(
                    n_samples_client * per_client_metrics[client_id][1][metric_key]
                    for client_id, (n_samples_client, _) in per_client_metrics.items()
                )
                val_metrics[metric_key] = weighted_sum / total_samples
                # Also compute unweighted (macro) average for post-hoc analysis
                macro_sum = sum(
                    per_client_metrics[client_id][1][metric_key]
                    for client_id, _ in per_client_metrics.items()
                )
                macro_metrics[metric_key] = macro_sum / len(per_client_metrics)

            # Log per-client and aggregated metrics
            logger.info("Per-client validation metrics:")
            for client_id, (n_samples_client, metrics) in per_client_metrics.items():
                metrics_str = {k: v if isinstance(v, int) else f"{v:.2e}" for k, v in metrics.items()}
                logger.info(f"  {client_id} ({n_samples_client} samples): {metrics_str}")

            val_metrics_str = {k: v if isinstance(v, int) else f"{v:.2e}" for k, v in val_metrics.items()}
            logger.info(f"Aggregated (FedAvg - weighted) validation metrics: {val_metrics_str}.")
            macro_metrics_str = {k: v if isinstance(v, int) else f"{v:.2e}" for k, v in macro_metrics.items()}
            logger.info(f"Aggregated (Macro - unweighted) validation metrics: {macro_metrics_str}.")

        else:
            # ── centralized evaluation (pool completo) ──────────────────────────
            logger.info("Eval mode 'centralized': evaluating on combined validation pool...")
            val_metrics = eval_dataloader_fn(
                model,
                val_dataloader,
                patch_size_dict,
                spacing_dict,
                amp_dtype,
                device,
            )

            val_metrics_str = {k: v if isinstance(v, int) else f"{v:.2e}" for k, v in val_metrics.items()}
            logger.info(f"Validation metrics: {val_metrics_str}.")

        # Log to wandb
        # For client eval mode, use total_samples (validation samples); for centralized, use dataset length
        if eval_mode == "client" and client_val_dataloaders:
            val_metrics["n_samples"] = total_samples
        else:
            val_metrics["n_samples"] = len(val_dataset)
        if wandb_run is not None:
            wandb_run.log({f"val_{k}": v for k, v in val_metrics.items()})

        # early stopping update
        early_stop_metric = val_metrics[config.train.early_stopping.metric]
        if config.train.early_stopping.mode == "max":
            early_stop_metric = -early_stop_metric
        early_stop.update(early_stop_metric)
        logger.info(
            f"Early stop updated {epoch}: "
            f"should_stop = {early_stop.should_stop}, "
            f"patience_count = {early_stop.patience_count}, "
            f"patience = {early_stop.patience}."
        )

        # save model checkpoint
        if early_stop.has_improved or epoch == 0:
            ckpt_path = save_checkpoint(
                ckpt_dir, epoch, model, optimizer, loss_scaler, n_samples, use_lora=config.train.use_lora
            )
            saved_ckpt_paths.append(ckpt_path)
            logger.info(f"Saved checkpoint of epoch {epoch} at {ckpt_path} after {n_samples} samples.")
            if len(saved_ckpt_paths) > config.train.max_n_ckpts > 0:
                to_delete = saved_ckpt_paths.pop(0)
                to_delete.unlink(missing_ok=True)
                logger.info(f"Deleted an outdated checkpoint {to_delete}.")

        # early stopping
        if early_stop.should_stop:
            logger.info(
                f"Met early stopping criteria with {config.train.early_stopping.metric} = "
                f"{early_stop.best_metric} and patience {early_stop.patience_count}, breaking."
            )
            break

        # ── JSONL metrics logging for post-hoc analysis ───────────────────────
        def convert_to_serializable(obj):
            """Convert numpy/torch types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy/torch scalar
                return obj.item()
            elif isinstance(obj, (float, int, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        # Prepare metrics record
        metrics_record = {
            "epoch": epoch,
            "optimizer_steps": total_optimizer_steps,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "forward_passes_per_epoch": forward_passes_per_epoch,
            "eval_mode": eval_mode,
            "n_accum_steps": n_accum_steps,
        }

        # Canonical validation contract:
        # - primary_validation_metrics always stores the metrics used for early stopping/checkpointing
        # - validation_aggregation is informational metadata about how primary_validation_metrics was produced
        metrics_record["primary_validation_metrics"] = val_metrics
        if eval_mode == "client" and client_val_dataloaders:
            metrics_record["validation_aggregation"] = "weighted - fedavg"
            metrics_record["macro_metrics"] = macro_metrics
            metrics_record["per_client_metrics"] = {
                client_id: {"n_samples": n_client, **metrics}
                for client_id, (n_client, metrics) in per_client_metrics.items()
            }
        else:
            metrics_record["validation_aggregation"] = "pooled"

        # Save to JSONL file
        metrics_file = Path(config.logging.dir).expanduser() / "validation_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(convert_to_serializable(metrics_record)) + "\n")
        logger.info(f"Saved metrics to {metrics_file}")

    logger.info(f"Last checkpoint is {saved_ckpt_paths[-1]}")


def main() -> None:
    """Entrypoint for centralized regression training with federated-style validation.

    Args:
        config: config loaded from yaml.
    """
    global logger
    # Use the root logger after Hydra's logging configuration is set up
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(
        description="Centralized training script for CineMRI classification/regression with federated-style or centralized-style validation. "
    )

    parser.add_argument(
        "--config",
        type=str,
        default="cinema_classification_centralized",
        help="Name of the configuration file to use"
    )

    args = parser.parse_args()

    try:
        GlobalHydra.instance().clear()  # Clear any existing Hydra instance

        config_dir = Path(__file__).resolve().parents[4] / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            config = compose(config_name=args.config)

        # Save configuration to output directory for reproducibility
        from omegaconf import OmegaConf
        config_output_dir = Path(config.logging.dir).expanduser()
        config_output_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_output_dir / "config.yaml"
        with open(config_file, "w") as f:
            OmegaConf.save(config, f)
        logger.info(f"Saved configuration to {config_file}")

        task = getattr(config, "task", "classification")
        eval_mode = getattr(config, "eval_mode", "centralized")
        logger.info(f"Starting centralized training for task: {task}")
        logger.info(f"Eval mode: {eval_mode}")

        loss_fn, eval_dataloader_fn = get_task_specific_functions(config)

        # Load per-client val dataloaders for client-based evaluation
        client_val_dataloaders = None
        if eval_mode == "client":
            logger.info("Loading per-client validation dataloaders for FedAvg aggregation...")
            client_val_dataloaders = load_per_client_val_dataloaders(config)

        run_train(
            config=config,
            load_dataset=load_combined_datasets,
            get_model_fn=get_model,
            loss_fn=loss_fn,
            eval_dataloader_fn=eval_dataloader_fn,
            client_val_dataloaders=client_val_dataloaders,
        )
    finally:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Finished centralized training script, cleaned up resources.")


if __name__ == "__main__":
    main()
