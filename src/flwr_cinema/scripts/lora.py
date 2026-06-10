"""LoRA (Low-Rank Adaptation) implementation for CineMA models.

This module provides LoRA adapters for parameter-efficient fine-tuning of the CineMA
ConvViT model in federated learning settings. LoRA adds trainable low-rank matrices to
the attention Q/K/V projections while freezing pretrained weights.

Reference: LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from models.cinema.cinema.log import get_logger

if TYPE_CHECKING:
    from models.cinema.cinema.convvit import ConvViT
    from models.cinema.cinema.segmentation.convunetr import ConvUNetR

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    """LoRA adapter for nn.Linear layers.
    
    Adds low-rank decomposition A @ B to the original weight matrix W, where:
    - W: (out_features, in_features) - frozen pretrained weights
    - A: (rank, in_features) - trainable down-projection
    - B: (out_features, rank) - trainable up-projection
    - scaling = alpha / rank
    
    Forward: y = Wx + (scaling * B @ A)x = Wx + LoRA(x)
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LoRA adapter.
        
        Args:
            original_layer: The original nn.Linear layer to adapt.
            rank: Rank of the low-rank decomposition (r in paper).
            alpha: Scaling factor for LoRA weights (typically 1x-4x of rank).
            dropout: Dropout probability for LoRA path.
        """
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # LoRA trainable parameters
        # Initialize A with Gaussian, B with zeros (so initial LoRA contribution is zero)
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)  # Same as nn.Linear
        nn.init.zeros_(self.lora_b)
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor of shape (*, in_features).
            
        Returns:
            Output tensor of shape (*, out_features).
        """
        # Original forward pass (frozen)
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA forward pass (trainable)
        # x @ A^T @ B^T = (x @ A^T) @ B^T
        lora_out = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()
        result = result + self.scaling * lora_out
        
        return result
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into the original layer for inference.
        
        This operation is irreversible and modifies the original weight tensor.
        After merging, LoRA parameters can be discarded.
        """
        if self.rank > 0:
            # W_new = W + scaling * B @ A
            with torch.no_grad():
                self.weight.data += self.scaling * (self.lora_b @ self.lora_a)
                # Zero out LoRA parameters to avoid double counting
                self.lora_a.data.zero_()
                self.lora_b.data.zero_()


class DualLoRALinear(nn.Module):
    """Dual LoRA adapter with global (aggregated) and local (personalized) components.
    
    For personalized federated learning, this module maintains:
    - Global adapters (A_g, B_g): Shared across clients, aggregated via FedAvg
    - Local adapters (A_l, B_l): Client-specific, never aggregated
    
    Forward pass:
    output = Wx + (alpha_g/rank_g) * B_g @ A_g @ x + (alpha_l/rank_l) * B_l @ A_l @ x
    
    This enables:
    1. Personalization: Each client adapts the model to its local distribution
    2. Shared knowledge: Global adapters capture common patterns across domains
    3. Privacy: Local adapters never leave the client device
    4. Scalability: Only global params need to be aggregated
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        global_rank: int = 4,
        local_rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize Dual LoRA adapter.
        
        Args:
            original_layer: The original nn.Linear layer to adapt.
            global_rank: Rank of global (aggregated) adapters (typically 4-8).
            local_rank: Rank of local (personalized) adapters (typically 4-8).
            alpha: Scaling factor for LoRA weights (typically 8-16).
            dropout: Dropout probability for LoRA path.
        """
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.alpha = alpha
        
        # Separate scaling for global and local
        self.scaling_g = alpha / global_rank if global_rank > 0 else 0.0
        self.scaling_l = alpha / local_rank if local_rank > 0 else 0.0
        
        # Freeze original weights
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # Global LoRA trainable parameters (aggregated across clients via FedAvg)
        self.lora_a_g = nn.Parameter(torch.zeros(global_rank, self.in_features))
        self.lora_b_g = nn.Parameter(torch.zeros(self.out_features, global_rank))
        nn.init.kaiming_uniform_(self.lora_a_g, a=5**0.5)
        nn.init.zeros_(self.lora_b_g)
        
        # Local LoRA trainable parameters (client-specific, never aggregated)
        self.lora_a_l = nn.Parameter(torch.zeros(local_rank, self.in_features))
        self.lora_b_l = nn.Parameter(torch.zeros(self.out_features, local_rank))
        nn.init.kaiming_uniform_(self.lora_a_l, a=5**0.5)
        nn.init.zeros_(self.lora_b_l)
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dual LoRA adaptation (global + local).
        
        Args:
            x: Input tensor of shape (*, in_features).
            
        Returns:
            Output tensor of shape (*, out_features).
        """
        # Base forward pass (frozen)
        result = F.linear(x, self.weight, self.bias)
        
        # Global LoRA forward pass (aggregated component)
        if self.global_rank > 0:
            x_dropout = self.dropout(x)
            lora_out_g = x_dropout @ self.lora_a_g.t() @ self.lora_b_g.t()
            result = result + self.scaling_g * lora_out_g
        
        # Local LoRA forward pass (personalization component)
        if self.local_rank > 0:
            x_dropout = self.dropout(x)
            lora_out_l = x_dropout @ self.lora_a_l.t() @ self.lora_b_l.t()
            result = result + self.scaling_l * lora_out_l
        
        return result
    
    def merge_weights(self) -> None:
        """Merge both global and local LoRA weights into the original layer for inference.
        
        This operation is irreversible and modifies the original weight tensor.
        After merging, LoRA parameters can be discarded.
        """
        with torch.no_grad():
            if self.global_rank > 0:
                self.weight.data += self.scaling_g * (self.lora_b_g @ self.lora_a_g)
                self.lora_a_g.data.zero_()
                self.lora_b_g.data.zero_()
            if self.local_rank > 0:
                self.weight.data += self.scaling_l * (self.lora_b_l @ self.lora_a_l)
                self.lora_a_l.data.zero_()
                self.lora_b_l.data.zero_()


def count_lora_parameters_by_block(
    model: ConvViT | ConvUNetR,
    blocks_to_lora: list[int] | None = None,
) -> dict:
    """Count LoRA parameters by block index.
    
    Args:
        model: ConvViT or ConvUNetR model with LoRA adapters applied.
        blocks_to_lora: List of block indices with LoRA (if None, checks all blocks).
        
    Returns:
        Dict with per-block parameter counts and totals.
    """
    param_counts = {}
    encoder = model.encoder
    
    if hasattr(encoder, "blocks") and blocks_to_lora is not None:
        for block_idx in blocks_to_lora:
            if 0 <= block_idx < len(encoder.blocks):
                block = encoder.blocks[block_idx]
                if hasattr(block, "attn"):
                    block_params = sum(
                        p.numel() for p in block.attn.parameters() if p.requires_grad
                    )
                    param_counts[f"block_{block_idx}"] = block_params
    
    # Count prediction head parameters
    if hasattr(model, 'pred_head_dict'):
        head_params = sum(
            p.numel() for p in model.pred_head_dict.parameters() if p.requires_grad
        )
        param_counts["pred_head"] = head_params
    
    param_counts["total_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_counts["total_model"] = sum(p.numel() for p in model.parameters())
    
    return param_counts


def apply_lora_to_attention(
    attention_module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> None:
    """Apply LoRA adapters to attention Q/K/V projections.
    
    Args:
        attention_module: The attention module (cinema.vit.Attention).
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: LoRA dropout rate.
        target_modules: List of module names to apply LoRA to. 
            Options: ["q", "k", "v", "kv", "proj"]. 
            Default is ["q", "kv"] for query and key/value projections.
    """
    if target_modules is None:
        target_modules = ["q", "kv"]  # Default: Q and KV projections
    
    for name in target_modules:
        if hasattr(attention_module, name):
            original_layer = getattr(attention_module, name)
            if isinstance(original_layer, nn.Linear):
                lora_layer = LoRALinear(
                    original_layer=original_layer,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(attention_module, name, lora_layer)
                logger.debug(f"Applied LoRA to {name} projection: {original_layer.in_features} -> {original_layer.out_features}, rank={rank}")
            else:
                logger.warning(f"Module {name} is not nn.Linear, skipping LoRA application")
        else:
            logger.warning(f"Attention module does not have attribute {name}")


def apply_dual_lora_to_attention(
    attention_module: nn.Module,
    global_rank: int = 4,
    local_rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> None:
    """Apply Dual LoRA adapters to attention Q/K/V projections.
    
    Args:
        attention_module: The attention module (cinema.vit.Attention).
        global_rank: Rank of global (aggregated) adapters (typically 4-8).
        local_rank: Rank of local (personalized) adapters (typically 4-8).
        alpha: LoRA alpha scaling factor.
        dropout: LoRA dropout rate.
        target_modules: List of module names to apply LoRA to.
    """
    if target_modules is None:
        target_modules = ["q", "kv"]
    
    for name in target_modules:
        if hasattr(attention_module, name):
            original_layer = getattr(attention_module, name)
            if isinstance(original_layer, nn.Linear):
                dual_lora_layer = DualLoRALinear(
                    original_layer=original_layer,
                    global_rank=global_rank,
                    local_rank=local_rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(attention_module, name, dual_lora_layer)
                logger.debug(
                    f"Applied Dual LoRA to {name}: global_rank={global_rank}, local_rank={local_rank}"
                )
            else:
                logger.warning(f"Module {name} is not nn.Linear, skipping Dual LoRA")
        else:
            logger.warning(f"Attention module does not have attribute {name}")


def apply_lora_to_convvit(
    model: ConvViT,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    blocks_to_lora: list[int] | None = None,
) -> None:
    """Apply LoRA adapters to ConvViT model's transformer attention layers.
    
    This function:
    1. Applies LoRA to specified (or all) attention Q/K/V projections in ViT encoder blocks
    2. Freezes all pretrained parameters (encoders, fusion, prediction heads)
    3. Only LoRA parameters remain trainable
    
    Args:
        model: ConvViT model instance.
        rank: LoRA rank (typically 4-16).
        alpha: LoRA alpha scaling (typically 16-32, or 1x-4x of rank).
        dropout: LoRA dropout rate (typically 0.0-0.1).
        target_modules: Which attention modules to adapt. Default ["q", "kv"].
        blocks_to_lora: List of block indices to apply LoRA to. If None, applies to all blocks (0-11 for ViT-B).
                       Example: [8, 9, 10, 11] for last 4 blocks, [0, 1, 2, 3] for first 4 blocks.
    """
    # First, freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Default: apply to all blocks if not specified
    encoder = model.encoder
    if blocks_to_lora is None and hasattr(encoder, "blocks"):
        blocks_to_lora = list(range(len(encoder.blocks)))
    
    # Validate block indices
    if blocks_to_lora and hasattr(encoder, "blocks"):
        invalid_blocks = [b for b in blocks_to_lora if b < 0 or b >= len(encoder.blocks)]
        if invalid_blocks:
            logger.warning(
                f"Invalid block indices {invalid_blocks}. Valid range: [0, {len(encoder.blocks)-1}]. "
                f"Using valid blocks from {blocks_to_lora}."
            )
            blocks_to_lora = [b for b in blocks_to_lora if 0 <= b < len(encoder.blocks)]
    
    logger.info(
        f"Applying LoRA (rank={rank}, alpha={alpha}, dropout={dropout}) to ConvViT attention layers. "
        f"Blocks: {blocks_to_lora}"
    )
    
    # Apply LoRA to selected encoder blocks
    lora_count = 0
    if hasattr(encoder, "blocks"):
        for block_idx, block in enumerate(encoder.blocks):
            # Only apply LoRA to blocks in blocks_to_lora list
            if block_idx in blocks_to_lora and hasattr(block, "attn"):
                apply_lora_to_attention(
                    attention_module=block.attn,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    target_modules=target_modules,
                )
                lora_count += 1
                logger.debug(f"Applied LoRA to encoder block {block_idx}")

    # Unfreeze prediction heads for task-specific fine-tuning
    if hasattr(model, 'pred_head_dict'):
        for head in model.pred_head_dict.parameters():
            head.requires_grad = True
        logger.info("Unfroze prediction heads for task-specific training")
    
    # Count trainable parameters
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count per-block parameters
    param_counts = count_lora_parameters_by_block(model, blocks_to_lora)
    
    logger.info(
        f"LoRA applied to {lora_count} attention blocks. "
        f"Trainable params: {lora_params:,} / {total_params:,} "
        f"({100 * lora_params / total_params:.2f}%)"
    )
    logger.debug(f"Parameter breakdown: {param_counts}")

def apply_lora_to_convunetr(
    model: ConvUNetR,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    blocks_to_lora: list[int] | None = None,
) -> None:
    """Apply LoRA adapters to ConvUNetR model's transformer attention layers (only to convViT encoder, decoder remains unchanged).
    
    This function:
    1. Applies LoRA to specified (or all) attention Q/K/V projections in ViT encoder blocks
    2. Freezes all pretrained parameters (encoders, fusion, prediction heads)
    3. Only LoRA and decoder parameters remain trainable
    
    Args:
        model: ConvUNetR model instance.
        rank: LoRA rank (typically 4-16).
        alpha: LoRA alpha scaling (typically 16-32, or 1x-4x of rank).
        dropout: LoRA dropout rate (typically 0.0-0.1).
        target_modules: Which attention modules to adapt. Default ["q", "kv"].
        blocks_to_lora: List of block indices to apply LoRA to. If None, applies to all blocks (0-11 for ViT-B).
                       Example: [8, 9, 10, 11] for last 4 blocks, [0, 1, 2, 3] for first 4 blocks.
    """

    # First, freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Default: apply to all blocks if not specified
    encoder = model.encoder
    if blocks_to_lora is None and hasattr(encoder, "blocks"):
        blocks_to_lora = list(range(len(encoder.blocks)))
    
    # Validate block indices
    if blocks_to_lora and hasattr(encoder, "blocks"):
        invalid_blocks = [b for b in blocks_to_lora if b < 0 or b >= len(encoder.blocks)]
        if invalid_blocks:
            logger.warning(
                f"Invalid block indices {invalid_blocks}. Valid range: [0, {len(encoder.blocks)-1}]. "
                f"Using valid blocks from {blocks_to_lora}."
            )
            blocks_to_lora = [b for b in blocks_to_lora if 0 <= b < len(encoder.blocks)]
    
    logger.info(
        f"Applying LoRA (rank={rank}, alpha={alpha}, dropout={dropout}) to ViT encoder attention layers. "
        f"Blocks: {blocks_to_lora}"
    )
    
    # Apply LoRA to selected encoder blocks
    lora_count = 0
    if hasattr(encoder, "blocks"):
        for block_idx, block in enumerate(encoder.blocks):
            # Only apply LoRA to blocks in blocks_to_lora list
            if block_idx in blocks_to_lora and hasattr(block, "attn"):
                apply_lora_to_attention(
                    attention_module=block.attn,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    target_modules=target_modules,
                )
                lora_count += 1
                logger.debug(f"Applied LoRA to encoder block {block_idx}")

    # Unfreeze decoder parameters and prediction heads for task-specific fine-tuning
    for view in model.views:
        if hasattr(model, 'dec_image_conv_block_dict') and view in model.dec_image_conv_block_dict:
            for param in model.dec_image_conv_block_dict[view].parameters():
                param.requires_grad = True
        if hasattr(model, 'dec_down_blocks_dict') and view in model.dec_down_blocks_dict:
            for param in model.dec_down_blocks_dict[view].parameters():
                param.requires_grad = True
        if hasattr(model, 'dec_conv_blocks_dict') and view in model.dec_conv_blocks_dict:
            for param in model.dec_conv_blocks_dict[view].parameters():
                param.requires_grad = True
        if hasattr(model, 'decoder_dict') and view in model.decoder_dict:
            for param in model.decoder_dict[view].parameters():
                param.requires_grad = True
        if hasattr(model, 'pred_head_dict'):
            for param in model.pred_head_dict[view].parameters():
                param.requires_grad = True

        logger.info(f"Unfroze decoder parameters and prediction head for task-specific training on view {view}")
    
    # Count trainable parameters
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count per-block parameters
    # param_counts = count_lora_parameters_by_block(model, blocks_to_lora)
    
    logger.info(
        f"LoRA applied to {lora_count} attention blocks. "
        f"Trainable params: {lora_params:,} / {total_params:,} "
        f"({100 * lora_params / total_params:.2f}%)"
    )
    # logger.debug(f"Parameter breakdown: {param_counts}")




def apply_dual_lora_to_convvit(
    model: ConvViT,
    global_rank: int = 4,
    local_rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    blocks_to_lora: list[int] | None = None,
) -> None:
    """Apply Dual LoRA adapters to ConvViT model's transformer attention layers.
    
    This function:
    1. Applies Dual LoRA to specified (or all) attention Q/K/V projections in ViT encoder blocks
    2. Global adapters (shared) are aggregated via FedAvg during federation
    3. Local adapters (personalized) stay on each client and are never aggregated
    4. Freezes all pretrained parameters except LoRA and prediction heads
    
    Args:
        model: ConvViT model instance.
        global_rank: Rank of global (aggregated) adapters (typically 4-8).
        local_rank: Rank of local (personalized) adapters (typically 4-8).
        alpha: LoRA alpha scaling (typically 8-16).
        dropout: LoRA dropout rate (typically 0.0-0.1).
        target_modules: Which attention modules to adapt. Default ["q", "kv"].
        blocks_to_lora: List of block indices to apply LoRA to. If None, applies to all blocks (0-11 for ViT-B).
                       Both global and local adapters use the same blocks.
    """
    # First, freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Default: apply to all blocks if not specified
    encoder = model.encoder
    if blocks_to_lora is None and hasattr(encoder, "blocks"):
        blocks_to_lora = list(range(len(encoder.blocks)))
    
    # Validate block indices
    if blocks_to_lora and hasattr(encoder, "blocks"):
        invalid_blocks = [b for b in blocks_to_lora if b < 0 or b >= len(encoder.blocks)]
        if invalid_blocks:
            logger.warning(
                f"Invalid block indices {invalid_blocks}. Valid range: [0, {len(encoder.blocks)-1}]. "
                f"Using valid blocks from {blocks_to_lora}."
            )
            blocks_to_lora = [b for b in blocks_to_lora if 0 <= b < len(encoder.blocks)]
    
    logger.info(
        f"Applying Dual LoRA (global_rank={global_rank}, local_rank={local_rank}, "
        f"alpha={alpha}, dropout={dropout}) to ConvViT attention layers. "
        f"Blocks: {blocks_to_lora}"
    )
    
    # Apply Dual LoRA to selected encoder blocks
    lora_count = 0
    if hasattr(encoder, "blocks"):
        for block_idx, block in enumerate(encoder.blocks):
            # Only apply Dual LoRA to blocks in blocks_to_lora list
            if block_idx in blocks_to_lora and hasattr(block, "attn"):
                apply_dual_lora_to_attention(
                    attention_module=block.attn,
                    global_rank=global_rank,
                    local_rank=local_rank,
                    alpha=alpha,
                    dropout=dropout,
                    target_modules=target_modules,
                )
                lora_count += 1
                logger.debug(f"Applied Dual LoRA to encoder block {block_idx}")
    
    # Unfreeze prediction heads for task-specific fine-tuning
    if hasattr(model, 'pred_head_dict'):
        for head in model.pred_head_dict.parameters():
            head.requires_grad = True
        logger.info("Unfroze prediction heads for task-specific training")
    
    # Count trainable parameters
    global_params = sum(
        p.numel() for m in model.modules() if isinstance(m, DualLoRALinear)
        for p in [m.lora_a_g, m.lora_b_g]
    )
    local_params = sum(
        p.numel() for m in model.modules() if isinstance(m, DualLoRALinear)
        for p in [m.lora_a_l, m.lora_b_l]
    )
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count per-block parameters
    param_counts = count_lora_parameters_by_block(model, blocks_to_lora)
    
    logger.info(
        f"Dual LoRA applied to {lora_count} attention blocks. "
        f"Global params: {global_params:,} (aggregated via FedAvg) | "
        f"Local params: {local_params:,} (personalized, never aggregated) | "
        f"Total trainable: {global_params + local_params:,} / {total_params:,} "
        f"({100 * (global_params + local_params) / total_params:.2f}%)"
    )
    logger.debug(f"Parameter breakdown: {param_counts}")


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Extract only LoRA parameters from a model.
    
    Args:
        model: Model with LoRA adapters applied.
        
    Returns:
        List of LoRA parameters (lora_a and lora_b).
    """
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_a, module.lora_b])
    return lora_params


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA parameters as a state dict.
    
    Args:
        model: Model with LoRA adapters.
        
    Returns:
        State dict containing LoRA parameters.
        - Standard LoRA: {module.lora_a, module.lora_b}
        - Dual LoRA: {module.lora_a_g, module.lora_b_g, module.lora_a_l, module.lora_b_l}
    """
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            # Dual LoRA: extract global and local adapters
            lora_state_dict[f"{name}.lora_a_g"] = module.lora_a_g.data.clone()
            lora_state_dict[f"{name}.lora_b_g"] = module.lora_b_g.data.clone()
            lora_state_dict[f"{name}.lora_a_l"] = module.lora_a_l.data.clone()
            lora_state_dict[f"{name}.lora_b_l"] = module.lora_b_l.data.clone()
        elif isinstance(module, LoRALinear):
            # Standard LoRA: extract adapters
            lora_state_dict[f"{name}.lora_a"] = module.lora_a.data.clone()
            lora_state_dict[f"{name}.lora_b"] = module.lora_b.data.clone()
    
    return lora_state_dict


def load_lora_state_dict(model: nn.Module, lora_state_dict: dict[str, torch.Tensor]) -> None:
    """Load LoRA parameters from a state dict.
    
    Args:
        model: Model with LoRA adapters.
        lora_state_dict: State dict containing LoRA parameters.
            - Standard LoRA: {module.lora_a, module.lora_b}
            - Dual LoRA: {module.lora_a_g, module.lora_b_g, module.lora_a_l, module.lora_b_l}
    """
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            # Dual LoRA loading
            lora_a_g_key = f"{name}.lora_a_g"
            lora_b_g_key = f"{name}.lora_b_g"
            lora_a_l_key = f"{name}.lora_a_l"
            lora_b_l_key = f"{name}.lora_b_l"
            
            if lora_a_g_key in lora_state_dict:
                module.lora_a_g.data.copy_(lora_state_dict[lora_a_g_key])
            if lora_b_g_key in lora_state_dict:
                module.lora_b_g.data.copy_(lora_state_dict[lora_b_g_key])
            if lora_a_l_key in lora_state_dict:
                module.lora_a_l.data.copy_(lora_state_dict[lora_a_l_key])
            if lora_b_l_key in lora_state_dict:
                module.lora_b_l.data.copy_(lora_state_dict[lora_b_l_key])
        elif isinstance(module, LoRALinear):
            # Standard LoRA loading
            lora_a_key = f"{name}.lora_a"
            lora_b_key = f"{name}.lora_b"
            if lora_a_key in lora_state_dict:
                module.lora_a.data.copy_(lora_state_dict[lora_a_key])
            if lora_b_key in lora_state_dict:
                module.lora_b.data.copy_(lora_state_dict[lora_b_key])


def get_lora_and_head_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA parameters AND classification/regression head parameters.
    
    During LoRA training, the backbone is frozen but the classification head is unfrozen.
    This function extracts both the LoRA adapters and the task-specific head for synchronization.
    
    Args:
        model: Model with LoRA adapters applied.
        
    Returns:
        State dict containing:
        - LoRA parameters (lora_a, lora_b for standard LoRA or lora_a_g/lora_b_g/lora_a_l/lora_b_l for dual)
        - Classification/regression head parameters (pred_head_dict)
    """
    combined_state_dict = {}
    
    # Extract LoRA parameters
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            # Dual LoRA: extract global and local adapters
            combined_state_dict[f"{name}.lora_a_g"] = module.lora_a_g.data.clone()
            combined_state_dict[f"{name}.lora_b_g"] = module.lora_b_g.data.clone()
            combined_state_dict[f"{name}.lora_a_l"] = module.lora_a_l.data.clone()
            combined_state_dict[f"{name}.lora_b_l"] = module.lora_b_l.data.clone()
        elif isinstance(module, LoRALinear):
            # Standard LoRA: extract adapters
            combined_state_dict[f"{name}.lora_a"] = module.lora_a.data.clone()
            combined_state_dict[f"{name}.lora_b"] = module.lora_b.data.clone()
    
    # Extract classification/regression head parameters
    if hasattr(model, 'pred_head_dict'):
        for view_name, head_module in model.pred_head_dict.items():
            for param_name, param in head_module.named_parameters():
                key = f"pred_head_dict.{view_name}.{param_name}"
                combined_state_dict[key] = param.data.clone()
    
    return combined_state_dict


def load_lora_and_head_state_dict(model: nn.Module, combined_state_dict: dict[str, torch.Tensor]) -> None:
    """Load LoRA parameters AND classification/regression head parameters.
    
    Args:
        model: Model with LoRA adapters applied.
        combined_state_dict: State dict containing both LoRA and head parameters.
    """
    # Load LoRA parameters
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            # Dual LoRA loading
            lora_a_g_key = f"{name}.lora_a_g"
            lora_b_g_key = f"{name}.lora_b_g"
            lora_a_l_key = f"{name}.lora_a_l"
            lora_b_l_key = f"{name}.lora_b_l"
            
            if lora_a_g_key in combined_state_dict:
                module.lora_a_g.data.copy_(combined_state_dict[lora_a_g_key])
            if lora_b_g_key in combined_state_dict:
                module.lora_b_g.data.copy_(combined_state_dict[lora_b_g_key])
            if lora_a_l_key in combined_state_dict:
                module.lora_a_l.data.copy_(combined_state_dict[lora_a_l_key])
            if lora_b_l_key in combined_state_dict:
                module.lora_b_l.data.copy_(combined_state_dict[lora_b_l_key])
        elif isinstance(module, LoRALinear):
            # Standard LoRA loading
            lora_a_key = f"{name}.lora_a"
            lora_b_key = f"{name}.lora_b"
            if lora_a_key in combined_state_dict:
                module.lora_a.data.copy_(combined_state_dict[lora_a_key])
            if lora_b_key in combined_state_dict:
                module.lora_b.data.copy_(combined_state_dict[lora_b_key])
    
    # Load classification/regression head parameters
    if hasattr(model, 'pred_head_dict'):
        for view_name, head_module in model.pred_head_dict.items():
            for param_name, param in head_module.named_parameters():
                key = f"pred_head_dict.{view_name}.{param_name}"
                if key in combined_state_dict:
                    param.data.copy_(combined_state_dict[key])
    
def merge_all_lora_weights(model: nn.Module) -> None:
    """Merge all LoRA weights into original layers for inference.
    
    Works with both LoRALinear and DualLoRALinear modules.
    
    Args:
        model: Model with LoRA adapters.
    """
    for module in model.modules():
        if isinstance(module, (LoRALinear, DualLoRALinear)):
            module.merge_weights()
    logger.info("Merged all LoRA weights into base model")


# Standard LoRA + decoder + head state dict functions (for ConvUNetR FedAvg)

def _extract_lora_params(state_dict: dict[str, torch.Tensor], model: nn.Module) -> None:
    """Extract standard LoRA parameters from model into state_dict."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state_dict[f"{name}.lora_a"] = module.lora_a.data.clone()
            state_dict[f"{name}.lora_b"] = module.lora_b.data.clone()


def _load_lora_params(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load standard LoRA parameters from state_dict into model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_a_key = f"{name}.lora_a"
            lora_b_key = f"{name}.lora_b"
            if lora_a_key in state_dict:
                module.lora_a.data.copy_(state_dict[lora_a_key])
            if lora_b_key in state_dict:
                module.lora_b.data.copy_(state_dict[lora_b_key])


def _extract_decoder_params(state_dict: dict[str, torch.Tensor], model: nn.Module) -> None:
    """Extract decoder parameters from ConvUNetR model into state_dict."""
    for view in model.views:
        if hasattr(model, 'dec_image_conv_block_dict') and view in model.dec_image_conv_block_dict:
            prefix = f"dec_image_conv_block_dict.{view}"
            for param_name, param in model.dec_image_conv_block_dict[view].named_parameters():
                state_dict[f"{prefix}.{param_name}"] = param.data.clone()

        if hasattr(model, 'dec_down_blocks_dict') and view in model.dec_down_blocks_dict:
            prefix = f"dec_down_blocks_dict.{view}"
            for param_name, param in model.dec_down_blocks_dict[view].named_parameters():
                state_dict[f"{prefix}.{param_name}"] = param.data.clone()

        if hasattr(model, 'dec_conv_blocks_dict') and view in model.dec_conv_blocks_dict:
            prefix = f"dec_conv_blocks_dict.{view}"
            for param_name, param in model.dec_conv_blocks_dict[view].named_parameters():
                state_dict[f"{prefix}.{param_name}"] = param.data.clone()

        if hasattr(model, 'decoder_dict') and view in model.decoder_dict:
            prefix = f"decoder_dict.{view}"
            for param_name, param in model.decoder_dict[view].named_parameters():
                state_dict[f"{prefix}.{param_name}"] = param.data.clone()


def _load_decoder_params(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load decoder parameters from state_dict into ConvUNetR model."""
    for view in model.views:
        if hasattr(model, 'dec_image_conv_block_dict') and view in model.dec_image_conv_block_dict:
            prefix = f"dec_image_conv_block_dict.{view}"
            for param_name, param in model.dec_image_conv_block_dict[view].named_parameters():
                key = f"{prefix}.{param_name}"
                if key in state_dict:
                    param.data.copy_(state_dict[key])

        if hasattr(model, 'dec_down_blocks_dict') and view in model.dec_down_blocks_dict:
            prefix = f"dec_down_blocks_dict.{view}"
            for param_name, param in model.dec_down_blocks_dict[view].named_parameters():
                key = f"{prefix}.{param_name}"
                if key in state_dict:
                    param.data.copy_(state_dict[key])

        if hasattr(model, 'dec_conv_blocks_dict') and view in model.dec_conv_blocks_dict:
            prefix = f"dec_conv_blocks_dict.{view}"
            for param_name, param in model.dec_conv_blocks_dict[view].named_parameters():
                key = f"{prefix}.{param_name}"
                if key in state_dict:
                    param.data.copy_(state_dict[key])

        if hasattr(model, 'decoder_dict') and view in model.decoder_dict:
            prefix = f"decoder_dict.{view}"
            for param_name, param in model.decoder_dict[view].named_parameters():
                key = f"{prefix}.{param_name}"
                if key in state_dict:
                    param.data.copy_(state_dict[key])


def _extract_head_params(state_dict: dict[str, torch.Tensor], model: nn.Module) -> None:
    """Extract prediction head parameters from model into state_dict."""
    if hasattr(model, 'pred_head_dict'):
        for view_name, head_module in model.pred_head_dict.items():
            prefix = f"pred_head_dict.{view_name}"
            for param_name, param in head_module.named_parameters():
                state_dict[f"{prefix}.{param_name}"] = param.data.clone()


def _load_head_params(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load prediction head parameters from state_dict into model."""
    if hasattr(model, 'pred_head_dict'):
        for view_name, head_module in model.pred_head_dict.items():
            prefix = f"pred_head_dict.{view_name}"
            for param_name, param in head_module.named_parameters():
                key = f"{prefix}.{param_name}"
                if key in state_dict:
                    param.data.copy_(state_dict[key])


# Standard LoRA + decoder + head state dict functions (for ConvUNetR)

def get_lora_decoder_head_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA, decoder, and head parameters for ConvUNetR federated communication.

    For standard LoRA models, this extracts all parameters that should be
    communicated to the server for FedAvg aggregation:
    - LoRA adapters (lora_a, lora_b) in encoder attention layers
    - Decoder parameters (dec_image_conv, dec_down_blocks, dec_conv_blocks, decoder per view)
    - Prediction head parameters (pred_head_dict per view)

    Frozen backbone parameters (enc_down_dict, encoder non-LoRA) are excluded
    because they are never trained.

    Args:
        model: ConvUNetR (or ConvViT) model with standard LoRA adapters applied.

    Returns:
        State dict containing LoRA, decoder, and head parameters.
    """
    state_dict: dict[str, torch.Tensor] = {}
    _extract_lora_params(state_dict, model)
    _extract_decoder_params(state_dict, model)
    _extract_head_params(state_dict, model)
    return state_dict


def load_lora_decoder_head_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load LoRA, decoder, and head parameters from aggregated state dict.

    Use this on clients after receiving FedAvg-aggregated parameters from the
    server. Only parameters present in the state dict are loaded — any extra
    keys are safely ignored.

    Args:
        model: ConvUNetR (or ConvViT) model with standard LoRA adapters applied.
        state_dict: State dict from server aggregation containing LoRA + decoder + head params.
    """
    _load_lora_params(model, state_dict)
    _load_decoder_params(model, state_dict)
    _load_head_params(model, state_dict)


# Standard LoRA + decoder (NO head) for ConvUNetR when aggregated_head=False

def get_lora_decoder_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA and decoder parameters WITHOUT prediction heads.

    Use this when ``aggregated_head=False`` — the prediction head is trained
    locally and never sent to the server. Only LoRA adapters and decoder
    parameters are communicated for FedAvg aggregation:

    - LoRA adapters (lora_a, lora_b) in encoder attention layers
    - Decoder parameters per view (dec_image_conv, dec_down_blocks,
      dec_conv_blocks, decoder)
    - Excludes: pred_head_dict, frozen backbone (enc_down_dict, encoder non-LoRA)

    Args:
        model: ConvUNetR model with standard LoRA adapters applied.

    Returns:
        State dict containing LoRA and decoder parameters only.
    """
    state_dict: dict[str, torch.Tensor] = {}
    _extract_lora_params(state_dict, model)
    _extract_decoder_params(state_dict, model)
    return state_dict


def load_lora_decoder_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load LoRA and decoder parameters while preserving local prediction heads.

    Use this on clients after receiving FedAvg-aggregated parameters from the
    server when ``aggregated_head=False``. Since pred_head_dict keys are not
    present in the state dict, they remain untouched — each client keeps its
    own personalized prediction head.

    Args:
        model: ConvUNetR model with standard LoRA adapters applied.
        state_dict: State dict from server aggregation containing LoRA + decoder params.
    """
    _load_lora_params(model, state_dict)
    _load_decoder_params(model, state_dict)


# Dual LoRA parameter extraction and loading functions

def get_global_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only GLOBAL LoRA parameters for server-side aggregation.
    
    This extracts only the global adapters (A_g, B_g) that should be aggregated
    via FedAvg. Local adapters are excluded and remain on the client.
    
    Args:
        model: Model with Dual LoRA adapters.
        
    Returns:
        State dict containing only global LoRA parameters (lora_a_g, lora_b_g).
    """
    global_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            global_state_dict[f"{name}.lora_a_g"] = module.lora_a_g.data.clone()
            global_state_dict[f"{name}.lora_b_g"] = module.lora_b_g.data.clone()
    return global_state_dict


def get_global_lora_and_head_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract GLOBAL LoRA parameters AND head parameters for federated communication.
    
    For Dual LoRA models, this extracts:
    - Global adapters (lora_a_g, lora_b_g) - to be aggregated on server
    - Classification/regression head parameters - to be aggregated on server
    - NOT local adapters (lora_a_l, lora_b_l) - these stay on client only
    
    This is used during federated learning to send only the aggregatable parts.
    
    Args:
        model: Model with Dual LoRA adapters.
        
    Returns:
        State dict containing global LoRA parameters and head parameters.
    """
    global_and_head_dict = {}
    
    # Extract GLOBAL LoRA parameters only (not local)
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            global_and_head_dict[f"{name}.lora_a_g"] = module.lora_a_g.data.clone()
            global_and_head_dict[f"{name}.lora_b_g"] = module.lora_b_g.data.clone()
    
    # Extract classification/regression head parameters
    if hasattr(model, 'pred_head_dict'):
        for view_name, head_module in model.pred_head_dict.items():
            for param_name, param in head_module.named_parameters():
                key = f"pred_head_dict.{view_name}.{param_name}"
                global_and_head_dict[key] = param.data.clone()
    
    return global_and_head_dict


def get_local_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LOCAL LoRA parameters (client-specific).
    
    Use this to backup/restore local parameters on the client.
    These should never be sent to or modified on the server.
    
    Args:
        model: Model with Dual LoRA adapters.
        
    Returns:
        State dict containing only local LoRA parameters (lora_a_l, lora_b_l).
    """
    local_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            local_state_dict[f"{name}.lora_a_l"] = module.lora_a_l.data.clone()
            local_state_dict[f"{name}.lora_b_l"] = module.lora_b_l.data.clone()
    return local_state_dict


def load_global_lora_state_dict(
    model: nn.Module,
    global_state_dict: dict[str, torch.Tensor],
) -> None:
    """Load GLOBAL LoRA parameters while preserving LOCAL parameters.
    
    Use this on the server after FedAvg aggregation to update only global params.
    Local parameters remain unchanged on the server and clients.
    
    Args:
        model: Model with Dual LoRA adapters.
        global_state_dict: State dict containing global parameters (from FedAvg aggregation).
    """
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            lora_a_g_key = f"{name}.lora_a_g"
            lora_b_g_key = f"{name}.lora_b_g"
            if lora_a_g_key in global_state_dict:
                module.lora_a_g.data.copy_(global_state_dict[lora_a_g_key])
            if lora_b_g_key in global_state_dict:
                module.lora_b_g.data.copy_(global_state_dict[lora_b_g_key])


def load_global_lora_and_head_state_dict(
    model: nn.Module,
    global_and_head_dict: dict[str, torch.Tensor],
) -> None:
    """Load GLOBAL LoRA and head parameters while preserving LOCAL LoRA parameters.
    
    For Dual LoRA models, this loads:
    - Aggregated global adapters (lora_a_g, lora_b_g)
    - Aggregated head parameters
    - Does NOT touch local adapters (lora_a_l, lora_b_l) - they stay unchanged
    
    Use this on clients after receiving aggregated parameters from server.
    
    Args:
        model: Model with Dual LoRA adapters.
        global_and_head_dict: State dict with global LoRA params and head params.
    """
    # Load GLOBAL LoRA parameters only (preserves local)
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            lora_a_g_key = f"{name}.lora_a_g"
            lora_b_g_key = f"{name}.lora_b_g"
            if lora_a_g_key in global_and_head_dict:
                module.lora_a_g.data.copy_(global_and_head_dict[lora_a_g_key])
            if lora_b_g_key in global_and_head_dict:
                module.lora_b_g.data.copy_(global_and_head_dict[lora_b_g_key])
    
    # Load head parameters
    if hasattr(model, 'pred_head_dict'):
        for view_name, head_module in model.pred_head_dict.items():
            for param_name, param in head_module.named_parameters():
                key = f"pred_head_dict.{view_name}.{param_name}"
                if key in global_and_head_dict:
                    param.data.copy_(global_and_head_dict[key])


def load_local_lora_state_dict(
    model: nn.Module,
    local_state_dict: dict[str, torch.Tensor],
) -> None:
    """Load LOCAL LoRA parameters (client-specific).
    
    Use this to restore client-specific local adapters.
    
    Args:
        model: Model with Dual LoRA adapters.
        local_state_dict: State dict containing local parameters.
    """
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            lora_a_l_key = f"{name}.lora_a_l"
            lora_b_l_key = f"{name}.lora_b_l"
            if lora_a_l_key in local_state_dict:
                module.lora_a_l.data.copy_(local_state_dict[lora_a_l_key])
            if lora_b_l_key in local_state_dict:
                module.lora_b_l.data.copy_(local_state_dict[lora_b_l_key])


def freeze_local_lora(model: nn.Module) -> None:
    """Freeze local LoRA parameters (for server-side aggregation).
    
    When aggregating on the server, freeze local parameters so they don't
    accidentally get updated. They should only change on the client.
    
    Args:
        model: Model with Dual LoRA adapters.
    """
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            module.lora_a_l.requires_grad = False
            module.lora_b_l.requires_grad = False
    logger.info("Froze local LoRA parameters")


def unfreeze_local_lora(model: nn.Module) -> None:
    """Unfreeze local LoRA parameters (for client-side training).
    
    On the client, both global and local parameters should be trainable.
    
    Args:
        model: Model with Dual LoRA adapters.
    """
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            module.lora_a_l.requires_grad = True
            module.lora_b_l.requires_grad = True
    logger.info("Unfroze local LoRA parameters")


def has_dual_lora(model: nn.Module) -> bool:
    """Check if model has Dual LoRA adapters.
    
    Args:
        model: Model to check.
        
    Returns:
        True if model contains DualLoRALinear modules, False otherwise.
    """
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            return True
    return False


def get_trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only trainable parameters as a state dict.
    
    This is useful for federated learning when freeze_pretrained=True,
    where only the classification head should be communicated (not the frozen backbone).
    
    Args:
        model: Model with potentially frozen backbone.
        
    Returns:
        State dict containing only parameters with requires_grad=True.
    """
    trainable_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state_dict[name] = param.data.clone()
    return trainable_state_dict


def load_trainable_state_dict(model: nn.Module, trainable_state_dict: dict[str, torch.Tensor]) -> None:
    """Load trainable parameters from a state dict.
    
    Only loads parameters that exist in the model and have requires_grad=True.
    Skipped parameters (frozen backbone) are left unchanged.
    
    Args:
        model: Model with potentially frozen backbone.
        trainable_state_dict: State dict containing only trainable parameters.
    """
    for name, param in model.named_parameters():
        if param.requires_grad and name in trainable_state_dict:
            param.data.copy_(trainable_state_dict[name])


# ── LoRa policy dispatch ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class LoRaPolicy:
    """Get/load function pair for model parameter communication.

    Centralises the decision of *which* state-dict extractor and loader to use
    based on model type, LoRA flavour, and aggregation policy.  Call sites no
    longer need their own if/elif trees.
    """
    get: Callable[[nn.Module], dict[str, torch.Tensor]]
    load: Callable[[nn.Module, dict[str, torch.Tensor]], None]


def get_lora_policy(
    model: nn.Module,
    *,
    use_lora: bool = False,
    freeze_pretrained: bool = False,
    aggregate_head: bool = True,
) -> LoRaPolicy:
    """Return the correct get/load pair for the model and configuration.

    Parameters
    ----------
    model:
        The concrete model instance (ConvViT, ConvUNetR, etc.).  The function
        inspects the model's attributes (e.g. ``has_dual_lora``,
        ``hasattr(model, "decoder_dict")``) to select the right variant.
    use_lora:
        Whether LoRA adapters are active.
    freeze_pretrained:
        Whether the pretrained backbone is frozen (head-only training).
    aggregate_head:
        Whether prediction heads should be aggregated.
        - ``True`` (default): LoRA + head sent to server.
        - ``False``: head stays local on each client.

    Note: decoder parameters are ALWAYS sent to the server for centralised
    checkpointing. The ``aggregate_decoder`` setting in ``CinemaStrategyMixin``
    controls whether they are aggregated or stripped-and-stored per-client,
    following the same pattern as ``aggregate_head`` for prediction heads.

    Returns
    -------
    LoRaPolicy(get=…, load=…)
        Appropriate extractor/loader pair for the current model state.
    """
    is_dual: bool = has_dual_lora(model)
    is_unetr: bool = hasattr(model, "decoder_dict")

    # ── No LoRA (head-only) ──────────────────────────────────────
    if not use_lora:
        if freeze_pretrained:
            return LoRaPolicy(
                get=get_trainable_state_dict,
                load=load_trainable_state_dict,
            )
        raise ValueError(
            "Full fine-tuning (use_lora=False, freeze_pretrained=False) is "
            "not handled by get_lora_policy. Call sites must handle it directly "
            "with model.state_dict() / model.load_state_dict()."
        )

    # ── ConvUNetR (LoRA ± decoder ± heads) ─────────────────────────────────────
    if is_unetr:
        # NOTE: decoder is ALWAYS sent to the server so it can be
        # checkpointed centrally. The server decides whether to aggregate
        # it (aggregate_decoder=True) or strip-and-store it per-client
        # (aggregate_decoder=False), following the same pattern as heads.
        # Bandwidth saving is achieved by not sending during intermediate
        # rounds — but currently the full decoder (39M params) is sent
        # every round. For true bandwidth optimisation, consider a
        # "send-on-final-round-only" protocol.
        if aggregate_head:
            return LoRaPolicy(
                get=get_lora_decoder_head_state_dict,
                load=load_lora_decoder_head_state_dict,
            )
        return LoRaPolicy(
            get=get_lora_decoder_state_dict,
            load=load_lora_decoder_state_dict,
        )

    # ── Dual LoRA (ConvViT) ────────────────────────────────────────────────────
    if is_dual:
        if aggregate_head:
            return LoRaPolicy(
                get=get_global_lora_and_head_state_dict,
                load=load_global_lora_and_head_state_dict,
            )
        return LoRaPolicy(
            get=get_global_lora_state_dict,
            load=load_global_lora_state_dict,
        )

    # ── Standard LoRA (ConvViT) ────────────────────────────────────────────────
    if aggregate_head:
        return LoRaPolicy(
            get=get_lora_and_head_state_dict,
            load=load_lora_and_head_state_dict,
        )
    return LoRaPolicy(
        get=get_lora_state_dict,
        load=load_lora_state_dict,
    )
