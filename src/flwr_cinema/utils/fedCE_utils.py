"""FedCE utility functions.

Computes gamma_cos (cosine similarity between local delta and excluded gradient)
and gamma_err (validation error) for FedCE client weighting. Also builds the
excluded model w_k^-i used to evaluate gamma_err on each client.
"""
import math
import time

import torch
from torch.nn.functional import cosine_similarity

import logging
from flwr.common.logger import log


def compute_flattened_delta_and_global_gradient(global_weights: dict, local_weights: dict, global_gradient: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten weight deltas and global gradients into 1D tensors for cosine similarity."""
        delta_tensors = []
        global_gradient_tensors = [] 
        common_keys = [k for k in global_weights.keys() if k in local_weights]
        log(logging.INFO, f"Computing delta and global gradient. Global weights key count: {len(global_weights)}. Local weights key count: {len(local_weights)}.")
        log(logging.INFO, f"Computing delta and global gradient. Common keys between global and local weights count: {len(common_keys)}.")
        # log(logging.INFO, 
        #     f"Key {common_keys[0]} dtypes — g: {global_weights[common_keys[0]].dtype}, "
        #     f"l: {local_weights[common_keys[0]].dtype}, "
        #     f"gg: {global_gradient[common_keys[0]].dtype}"
        # )
        for key in common_keys:
            delta = local_weights[key].detach().cpu() - global_weights[key].detach().cpu()
            delta_tensors.append(delta.reshape(-1))  # Flatten the tensor

            if global_gradient and key in global_gradient:
                grad = global_gradient[key].detach().cpu()
                global_gradient_tensors.append(grad.reshape(-1))  # Flatten the global gradient tensor

        delta_1d = torch.cat(delta_tensors, dim=0) if delta_tensors else torch.empty(0, device='cpu')
        global_gradient_1d = torch.cat(global_gradient_tensors, dim=0) if global_gradient_tensors else torch.empty(0, device='cpu')

        log(logging.INFO, f"Computed flattened delta and global gradient. Delta shape: {delta_1d.shape}, Global gradient shape: {global_gradient_1d.shape}")

        return delta_1d, global_gradient_1d # ∇F_i(w_k,i), ∇F(w_k) if provided


def calculate_excluded_delta(delta_1d: torch.Tensor, global_gradient_1d: torch.Tensor, prev_rho: float, epsilon: float = 1e-5) -> torch.Tensor:
    """Calculate the excluded delta for FedCE (authors-style, no division by 1-prev_rho)."""
    return global_gradient_1d - (prev_rho * delta_1d)  # ∇F(w_k^-i)

def calculate_gamma_cos(delta_1d: torch.Tensor, excluded_delta_1d: torch.Tensor) -> float:
    """Calculate the cosine similarity-based gamma for FedCE."""
    log(logging.INFO, f"Calculating cosine similarity. Delta shape: {delta_1d.shape}, Excluded delta shape: {excluded_delta_1d.shape}")
    cos_sim = cosine_similarity(delta_1d.unsqueeze(0), excluded_delta_1d.unsqueeze(0)).item()
    log(logging.INFO, f"Computed cosine similarity for client: {cos_sim}")
    gamma_cos = 1 - cos_sim
    return max(0, gamma_cos) # Γ_k,i(cos)

def calculate_excluded_model(global_weights: dict, local_weights: dict, prev_rho: float, epsilon: float = 1e-5) -> dict:
    """Calculate the excluded model for FedCE.

    Caps prev_rho at 0.99 to prevent numerical explosion when a single
    client dominates (denominator 1-prev_rho → 0 amplifies the excluded
    model by orders of magnitude).
    """
    if prev_rho == 0.0:
        return global_weights 

    # cap to prevent division by near-zero (1-prev_rho)
    prev_rho = min(prev_rho, 0.99)

    excluded_model = {}
    common_keys = [k for k in global_weights.keys() if k in local_weights]
    for key in common_keys:
        excluded_model[key] = (global_weights[key].detach().cpu() - (prev_rho*local_weights[key].detach().cpu())) / max((1-prev_rho), epsilon) 

    return excluded_model # w_k^-i

# def calculate_fedCE_raw_rho(global_weights: dict, local_weights: dict, global_gradient: dict, prev_rho: float, gamma_err: float, epsilon: float = 1e-5, method: str= "multiply") -> float:
#     """Calculate the raw rho for FedCE"""
#     delta_1d, global_gradient_1d = compute_flattened_delta_and_global_gradient(global_weights, local_weights, global_gradient)
#     excluded_delta_1d = calculate_excluded_delta(delta_1d, global_gradient_1d, prev_rho, epsilon)
#     gamma_cos = calculate_gamma_cos(delta_1d, excluded_delta_1d)

#     if method == "sum":
#         raw_rho = gamma_cos + gamma_err
#     else: # default to "multiply"
#         raw_rho = gamma_cos * gamma_err
#     return float(raw_rho)

# def compute_fedCE_gamma_cos(global_weights: dict, local_weights: dict, global_gradient: dict, prev_rho: float, epsilon: float = 1e-5) -> float:
#     """Compute the FedCE gamma_cos for a client."""

#     start_time = time.perf_counter()
#     delta_1d, global_gradient_1d = compute_flattened_delta_and_global_gradient(global_weights, local_weights, global_gradient)
#     excluded_delta_1d = calculate_excluded_delta(delta_1d, global_gradient_1d, prev_rho, epsilon)
#     gamma_cos = calculate_gamma_cos(delta_1d, excluded_delta_1d)
#     end_time = time.perf_counter()
#     log(logging.INFO, f"FedCE gamma_cos computation time for client: {end_time - start_time:.6f} seconds")
#     return float(gamma_cos)


def compute_fedCE_gamma_cos(global_weights: dict, local_weights: dict, global_gradient: dict, prev_rho: float, epsilon: float = 1e-5) -> float:
    """Compute the FedCE gamma_cos for a client.

    Uses authors-style excluded gradient (without division by 1-prev_rho)
    since cosine similarity is scale-invariant. This avoids numerical
    instability when prev_rho ≈ 1.
    """
    
    start_time = time.perf_counter()
    if not global_gradient:
        log(logging.WARNING, "Global gradient is empty. Cannot compute FedCE gamma_cos without global gradient information.")
        return 0.0
    
    dot = 0.0
    norm_delta = 0.0
    norm_excl = 0.0

    common_keys = [k for k in global_weights.keys() if k in local_weights and k in global_gradient]
    log(logging.INFO, f"FedCE gamma_cos streaming. Common keys: {len(common_keys)}")

    with torch.no_grad():
        for key in common_keys:
            # Mantener operación por capa para no materializar vectores enormes
            g = global_weights[key].detach().cpu()
            l = local_weights[key].detach().cpu()
            gg = global_gradient[key].detach().cpu()

            delta = (l - g).reshape(-1)
            gg = gg.reshape(-1)
            # Authors implementation excluded gradient (no division by 1-prev_rho).
            # Cosine similarity is scale-invariant, so this produces identical
            # gamma_cos while avoiding numerical instability when prev_rho near 1.
            excluded_delta = (gg - (prev_rho * delta)).reshape(-1)

            # Acumuladores escalares, memoria constante
            dot += torch.dot(delta, excluded_delta).item()
            norm_delta += torch.dot(delta, delta).item()
            norm_excl += torch.dot(excluded_delta, excluded_delta).item()

            # Romper referencias de tensores temporales
            del g, l, gg, delta, excluded_delta

    if norm_delta <= 0.0 or norm_excl <= 0.0:
        return 0.0 # TODO: Implement custom exceptions

    cos_sim = dot / (max(math.sqrt(norm_delta) * math.sqrt(norm_excl), epsilon))
    log(logging.INFO, f"FedCE gamma_cos streaming computed cosine similarity for client: {cos_sim}")
    gamma_cos = 1.0 - cos_sim
    end_time = time.perf_counter()
    log(logging.INFO, f"FedCE gamma_cos streaming computation time for client: {end_time - start_time:.6f} seconds")
    return float(max(0.0, gamma_cos))