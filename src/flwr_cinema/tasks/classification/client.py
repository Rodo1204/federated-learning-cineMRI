import logging
from flwr_cinema.tasks.baseClient import BaseClient
from models.cinema.cinema.train import train_one_epoch

# logger = logging.getLogger(__name__)

class ClassificationClient(BaseClient):
    """Specific Client implementation for Classification tasks."""

    # El __init__ se hereda completamente de BaseClient para evitar código duplicado.

    def train_local(self, round_idx: int) -> tuple[int, dict]:
        """Implements the concrete local training for Cinema classification."""
        
        self.model.train()
        n_samples_trained = 0

        for local_epoch in range(self.local_epochs):
            global_epoch = round_idx * self.local_epochs + local_epoch
            self.optimizer.zero_grad(set_to_none=True)

            n_samples_trained = train_one_epoch(
                model=self.model,
                train_dataloader=self.train_loader,
                optimizer=self.optimizer,
                loss_scaler=self.loss_scaler,
                amp_dtype=self.amp_dtype,
                device=self.device,
                epoch=global_epoch,
                n_accum_steps=self.n_accum_steps,
                n_samples=n_samples_trained,
                config=self.config,
                wandb_run=None,
                loss_fn=self.loss_fn,
            )
            
        return self.n_samples

    def validate_local(self) -> tuple[int, dict]:
        """Implements local validation for Cinema classification."""
        if self.val_loader is None or self.eval_dataloader_fn is None:
            # logger.warning("No validation loader or eval func provided, returning empty.")
            return 0, {}

        patch_size_dict, spacing_dict = self.get_patch_and_spacing_dicts()

        self.model.eval()
        metrics = self.eval_dataloader_fn(
            self.model,
            self.val_loader,
            patch_size_dict,
            spacing_dict,
            self.amp_dtype,
            self.device,
        )
        
        return self.n_val_samples, metrics
