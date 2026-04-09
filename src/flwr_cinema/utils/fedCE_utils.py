import torch
from torch.nn.functional import cosine_similarity


def compute_flattened_delta_and_global_gradient(global_weights: dict, local_weights: dict, global_gradient: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the FedCE delta between global and local weights and flatten global gradients aswell."""
        delta_tensors = []
        global_gradient_tensors = [] 
        common_keys = [k for k in global_weights.keys() if k in local_weights]
        for key in common_keys:
            delta = local_weights[key].detach().cpu() - global_weights[key].detach().cpu()
            delta_tensors.append(delta.reshape(-1))  # Flatten the tensor

            if global_gradient and key in global_gradient:
                grad = global_gradient[key].detach().cpu()
                global_gradient_tensors.append(grad.reshape(-1))  # Flatten the global gradient tensor

        delta_1d = torch.cat(delta_tensors, dim=0) if delta_tensors else torch.empty(0, device='cpu')
        global_gradient_1d = torch.cat(global_gradient_tensors, dim=0) if global_gradient_tensors else torch.empty(0, device='cpu')

        return delta_1d, global_gradient_1d # ∇F_i(w_k,i), ∇F(w_k) if provided


def calculate_excluded_delta(delta_1d: torch.Tensor, global_gradient_1d: torch.Tensor, prev_rho: float, epsilon: float = 1e-5) -> torch.Tensor:
    """Calculate the excluded delta for FedCE."""
    if prev_rho == 0.0:
         return global_gradient_1d # Base case for the first round
    return (global_gradient_1d - (prev_rho*delta_1d)) / max((1-prev_rho), epsilon) # ∇F(w_k^-i)

def calculate_gamma_cos(delta_1d: torch.Tensor, excluded_delta_1d: torch.Tensor) -> float:
    """Calculate the cosine similarity-based gamma for FedCE."""
    cos_sim = cosine_similarity(delta_1d.unsqueeze(0), excluded_delta_1d.unsqueeze(0)).item()
    gamma_cos = 1 - cos_sim
    return max(0, gamma_cos) # Γ_k,i(cos)

def calculate_excluded_model(global_weights: dict, local_weights: dict, prev_rho: float, epsilon: float = 1e-5) -> dict:
    """Calculate the excluded model for FedCE."""
    if prev_rho == 0.0:
        return global_weights # Base case for the first round
    
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

def compute_fedCE_gamma_cos(global_weights: dict, local_weights: dict, global_gradient: dict, prev_rho: float, epsilon: float = 1e-5) -> float:
    """Compute the FedCE gamma_cos for a client."""
    
    delta_1d, global_gradient_1d = compute_flattened_delta_and_global_gradient(global_weights, local_weights, global_gradient)
    excluded_delta_1d = calculate_excluded_delta(delta_1d, global_gradient_1d, prev_rho, epsilon)
    gamma_cos = calculate_gamma_cos(delta_1d, excluded_delta_1d)
    return float(gamma_cos)
