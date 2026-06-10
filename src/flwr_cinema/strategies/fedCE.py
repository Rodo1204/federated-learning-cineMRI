"""FedCE aggregation strategy.

Implements FedCE weighting: clients compute gamma_cos (cosine similarity to
global gradient) and gamma_err (validation error), the server normalizes and
combines them into per-client aggregation weights rho.

Reference: FedCE (Federated Contributive Explanation), ISBI 2025.
"""
from flwr.app import Message, ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from collections.abc import Iterable
from flwr_cinema.strategies.common import CinemaStrategyMixin
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from logging import INFO, DEBUG, ERROR
from flwr.common.logger import log

# Minimum aggregation weight to prevent zero-weight crash (aligned with authors implementation)
_FEDCE_MIN_WEIGHT = 1e-3

class FedCE(FedAvg):
    """FedAvg variant implementing FedCE client weighting.

    Clients compute gamma_cos and gamma_err; server normalizes and combines
    them into rho weights for weighted aggregation.

    Args are forwarded: CinemaStrategyMixin consumes its own (eval_steps,
    primary_metric_name, etc.), the rest go to FedAvg.
    """
    def __init__(self, **kwargs):
        combination_method = kwargs.pop("combination_method", "multiplication")
        log(INFO, f"Initializing FedCE strategy with combination method: {combination_method}")
        self.combination_method = combination_method if combination_method in ["multiplication", "sum"] else "multiplication"

        # Use normalized rho as aggregation weight key to reuse FedAvg's weighted logic.
        kwargs["weighted_by_key"] = "fedce_normalized_rho"
        kwargs["train_metrics_aggr_fn"] = lambda replies, _: aggregate_metricrecords(replies, "num_samples")
        super().__init__(**kwargs)
        self.prev_normalized_rho_by_client: dict[str, float] = {}
        self._previous_global_params: ArrayRecord | None = None  # W_{t-1} for global gradient
        self.global_gradient: ArrayRecord | None = None  # sent to clients for gamma_cos
        self.accumulated_gammas: dict[str, float] = {}
        self.first_round_uniform_rho: float | None = None  # 1/|C| for round 1

    def compute_global_gradient(self, previous_global_params: ArrayRecord, aggregated_params: ArrayRecord) -> ArrayRecord:
        """Compute the global gradient as the difference between the previous global parameters and the newly aggregated parameters."""
        prev_params = previous_global_params.to_torch_state_dict(keep_input=True)
        new_params = aggregated_params.to_torch_state_dict(keep_input=True)

        new_global_gradient = {}
        common_keys = [k for k in prev_params.keys() if k in new_params]
        for key in common_keys:
            new_global_gradient[key] = new_params[key].detach().cpu() - prev_params[key].detach().cpu()

        return ArrayRecord.from_torch_state_dict(new_global_gradient, keep_input=False)
    
    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Inject FedCE data (global gradient, prev rho) into client messages."""
    
        self._previous_global_params = arrays

        base_messages = list(super().configure_train(server_round, arrays, config, grid))
        if not base_messages:
            return base_messages
        
        if server_round == 1: 
            total_clients = len(list(grid.get_node_ids())) 
            self.first_round_uniform_rho = 1.0 / total_clients if total_clients > 0 else 0.0
            log(INFO, f"First round: setting uniform rho for all clients to {self.first_round_uniform_rho} based on total clients: {total_clients}")
            return base_messages

        for message in base_messages:
            client_id = str(message.metadata.dst_node_id)
            new_content = RecordDict()
            log(INFO, f"Configuring train message for client {client_id} with content keys: {message.content.keys()}")
            for key, record in message.content.items():
                if isinstance(record, ConfigRecord):
                    new_content[key] = ConfigRecord(record)
                else:
                    new_content[key] = record

            strategy_data = ConfigRecord()
            strategy_data["prev_fedce_rho"] = float(self.prev_normalized_rho_by_client.get(client_id, self.first_round_uniform_rho))
            strategy_data["primary_metric_name"] = self.primary_metric_name
            strategy_data["primary_metric_direction"] = self.metric_direction
            new_content["strategy_data"] = strategy_data
            log(INFO, f"Configuring train message for client {client_id} with prev_fedce_rho: {strategy_data['prev_fedce_rho']}")

            if self.global_gradient is not None:
                new_content["global_gradient"] = self.global_gradient
            message.content = new_content

        return base_messages

    def aggregate_train(self, server_round, replies):
        """Aggregate with FedCE rho weights computed from gamma_cos and gamma_err."""
        replies_list = list(replies)
        if server_round == 1:
            # Round 1: no gamma data yet — use uniform weights (num_samples or 1/|C|).
            for message in replies_list:
                client_id = str(message.metadata.src_node_id)
                message.content.metric_records["metrics"]["fedce_normalized_rho"] = message.content.metric_records["metrics"].get("num_samples", self.first_round_uniform_rho) 
            agg_arrays, agg_metrics = super().aggregate_train(server_round, replies_list)

        else: 
            raw_cos, raw_err = {}, {}
            sum_cos, sum_err = 0.0, 0.0

            for message in replies_list:
                client_id = str(message.metadata.src_node_id)
                metrics = message.content.metric_records.get("metrics", MetricRecord())
                gamma_cos = metrics.get("fedce_gamma_cos", None)
                gamma_err = metrics.get("fedce_gamma_err", None)
                if gamma_cos is not None and gamma_err is not None:
                    raw_cos[client_id] = gamma_cos
                    raw_err[client_id] = gamma_err
                    sum_cos += gamma_cos
                    sum_err += gamma_err
            
            # Normalize and combine gammas into per-client rho.
            for client_id in raw_cos.keys():
                normalized_cos = (raw_cos[client_id] / sum_cos) if sum_cos > 0 else 0.0
                normalized_err = (raw_err[client_id] / sum_err) if sum_err > 0 else 0.0
                combined_gamma = normalized_cos + normalized_err if self.combination_method == "sum" else normalized_cos * normalized_err
                self.accumulated_gammas[client_id] = self.accumulated_gammas.get(client_id, 0.0) + combined_gamma
            
            z_k = sum(self.accumulated_gammas.values())
            self.prev_normalized_rho_by_client.clear()

            for client_id, gamma_sum in self.accumulated_gammas.items():
                self.prev_normalized_rho_by_client[client_id] = gamma_sum / z_k if z_k > 0 else 0.0

            # Clip minimum weight and re-normalize (aligned with authors: np.clip(weights, a_min=1e-3)).
            if self.prev_normalized_rho_by_client:
                for cid in self.prev_normalized_rho_by_client:
                    self.prev_normalized_rho_by_client[cid] = max(self.prev_normalized_rho_by_client[cid], _FEDCE_MIN_WEIGHT)
                z_clipped = sum(self.prev_normalized_rho_by_client.values())
                if z_clipped > 0:
                    for cid in self.prev_normalized_rho_by_client:
                        self.prev_normalized_rho_by_client[cid] /= z_clipped

            for message in replies_list:
                client_id = str(message.metadata.src_node_id)
                rho = self.prev_normalized_rho_by_client.get(client_id)
                if rho is None:
                    rho = self.first_round_uniform_rho or _FEDCE_MIN_WEIGHT
                    log(INFO, f"Client {client_id} not in prev_normalized_rho_by_client, using fallback rho={rho}")
                message.content.metric_records["metrics"]["fedce_normalized_rho"] = rho

            agg_arrays, agg_metrics = super().aggregate_train(server_round, replies_list)

        # Compute global gradient for next round's gamma_cos.
        if self._previous_global_params and agg_arrays:
            self.global_gradient = self.compute_global_gradient(self._previous_global_params, agg_arrays)

        return agg_arrays, agg_metrics
    
    def aggregate_evaluate(self, server_round, replies):
        """Evaluate with num_samples weighting, then restore FedCE weighting."""
        try:
            agg_metric = super().aggregate_evaluate(server_round, replies)
        finally:
            self.weighted_by_key = "fedce_normalized_rho" # Switch back to FedCE weighting for the next training round

        return agg_metric


class FedCEStrategy(CinemaStrategyMixin, FedCE):
    """FedCE + CinemaStrategyMixin (step validation, early stopping, checkpoints).

    MRO: CinemaStrategyMixin -> FedCE -> FedAvg.
    All params forwarded via **kwargs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    