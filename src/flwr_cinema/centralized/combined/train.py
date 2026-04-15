from collections.abc import Callable

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import torch
from torch import nn

from flwr_cinema.scripts.eval_utils import load_metadata
from flwr_cinema.scripts.lora import apply_dual_lora_to_convvit, apply_lora_to_convvit
from models.cinema.cinema.train import maybe_reduce_batch_size

from models.cinema.cinema.classification.dataset import get_image_transforms
from models.cinema.cinema.convvit import ConvViT, load_pretrain_weights, param_groups_lr_decay
from models.cinema.cinema.device import get_amp_dtype_and_device, print_model_info
from models.cinema.cinema.log import get_logger, init_wandb
from models.cinema.cinema.optim import EarlyStopping, GradScaler, save_checkpoint
from models.cinema.cinema.classification.dataset import EndDiastoleEndSystoleDataset
from models.cinema.cinema.train import maybe_subset_dataset, train_one_epoch
from models.cinema.cinema.optim import get_n_accum_steps

import logging
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from hydra.core.global_hydra import GlobalHydra

logger = None


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
    elif hasattr(config.federated, "client_ids") and config.federated.client_ids is not None:
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
    from torch.utils.data import DataLoader, SequentialSampler
    
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
) -> None:
    """Main function to fine-tune.

    Args:
        config: configuration file.
        load_dataset: function to load the train and val datasets.
        get_model_fn: function to initialize the model.
        loss_fn: loss function for segmentation tasks.
        eval_dataloader_fn: evaluation function for segmentation tasks.
    """
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
        param_groups = [p for p in model.parameters() if p.requires_grad]  # Only optimize trainable parameters (e.g., LoRA adapters or unfrozen head)
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, weight_decay=config.train.weight_decay if not config.train.use_lora else 0.0) #, betas=config.train.betas
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
    logger.info(f"  Forward passes per epoch: {forward_passes_per_epoch}")
    logger.info(f"  Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    logger.info(f"  Eval steps: {eval_optimizer_steps} optimizer steps")
    logger.info(f"  Gradient accumulation steps: {n_accum_steps}")
    logger.info(f"  → Validation triggers at optimizer steps: {eval_optimizer_steps}, {eval_optimizer_steps * 2}, {eval_optimizer_steps * 3}, ...")

    total_optimizer_steps = 0

    for epoch in range(config.train.n_epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{config.train.n_epochs}")
        logger.info(f"{'='*80}")
        optimizer.zero_grad()
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
        if eval_optimizer_steps <= 0 or (total_optimizer_steps % eval_optimizer_steps != 0):
            continue
        # evaluate model
        logger.info(f"Start evaluating model at step {total_optimizer_steps}.")
        model.eval()

        val_metrics = eval_dataloader_fn(
            model,
            val_dataloader,
            patch_size_dict,
            spacing_dict,
            amp_dtype,
            device,
        )
        val_metrics = {f"{k}": v for k, v in val_metrics.items()}
        val_metrics["n_samples"] = n_samples
        if wandb_run is not None:
            wandb_run.log(val_metrics)
        val_metrics_str = {k: v if isinstance(v, int) else f"{v:.2e}" for k, v in val_metrics.items()}
        logger.info(f"Validation metrics: {val_metrics_str}.")

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
            ckpt_path = save_checkpoint(ckpt_dir, epoch, model, optimizer, loss_scaler, n_samples)
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
    logger.info(f"Last checkpoint is {saved_ckpt_paths[-1]}")

def main() -> None:
    """Entrypoint for centralized regression training with federated-style validation.

    Args:
        config: config loaded from yaml.
    """
    global logger
    # Use the root logger after Hydra's logging configuration is set up
    logger = get_logger(__name__)

    GlobalHydra.instance().clear()  # Clear any existing Hydra instance

    config_dir = Path(__file__).resolve().parents[5] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="cinema_classification_centralized")
    
    task = getattr(config, "task", "classification")
    logger.info(f"Starting centralized training for task: {task}")

    loss_fn, eval_dataloader_fn = get_task_specific_functions(config)
    

    run_train(
        config=config,
        load_dataset=load_combined_datasets,
        get_model_fn=get_model,
        loss_fn=loss_fn,
        eval_dataloader_fn=eval_dataloader_fn
    )