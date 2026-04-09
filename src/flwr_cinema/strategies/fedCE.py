from flwr.app import Message, ArrayRecord, ConfigRecord, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from collections.abc import Iterable
from flwr_cinema.strategies.common import CinemaStrategyMixin

class FedCE(FedAvg):
    """Federated strategy implementing FedCE aggregation, combined with CineMA-specific logic (step-based validation, 
    early stopping, checkpointing).
    
    All parameters are forwarded through **kwargs — CinemaStrategyMixin consumes
    its own (eval_steps, primary_metric_name, etc.) and passes the rest to FedAvg
    (fraction_train, weighted_by_key, etc.).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_normalized_rho_by_client: dict[str, float] = {} # Store previous normalized rho values for each client to use in the next round's calculations
        self._previous_global_params: ArrayRecord | None = None # Store global parameters to send to clients for FedCE calculations
        self.global_gradient: ArrayRecord | None = None # Store global gradient to send to clients for FedCE calculations

    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        raw_cos, raw_err = {}, {}
        sum_cos, sum_err = 0.0, 0.0

        for message in valid_replies:
            client_id = str(message.metadata.src_node_id)
            metrics = message.content.metric_records.get("metrics", MetricRecord())
            gamma_cos = metrics.get("fedce_gamma_cos", None)
            gamma_err = metrics.get("fedce_gamma_err", None)
            if gamma_cos is not None and gamma_err is not None:
                raw_cos[client_id] = gamma_cos
                raw_err[client_id] = gamma_err
                sum_cos += gamma_cos
                sum_err += gamma_err
        


        # Hay que hacer la normalizacion de los gammas aqui no en el cliente, q paper mas confuso xdios :(


class FedStrategy(CinemaStrategyMixin, FedAvg):
    """Federated strategy combining CineMA-specific logic (step-based validation, 
    early stopping, checkpointing) with FedCE aggregation.
    
    All parameters are forwarded through **kwargs — CinemaStrategyMixin consumes
    its own (eval_steps, primary_metric_name, etc.) and passes the rest.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_normalized_rho_by_client: dict[str, float] = {} # Store previous normalized rho values for each client to use in the next round's calculations
        self._previous_global_params: ArrayRecord | None = None # Store global parameters to send to clients for FedCE calculations
        self.global_gradient: ArrayRecord | None = None # Store global gradient to send to clients for FedCE calculations

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Override configure_train to include FedCE-specific data (global gradients, previous rhos) in the messages sent to clients."""
    
        # Set flags and shared data in the config to indicate to clients that they should perform FedCE-specific calculations
        config["use_fedce"] = True # Flag to indicate to clients that they should perform FedCE-specific calculations
        config["fedce_primary_metric_name"] = self.primary_metric_name # Pass primary metric name

        # First call the parent method to get the base messages with global weights
        base_messages = list(super().configure_train(server_round, arrays, config, grid))

        if not base_messages:
            return base_messages  # No clients selected, return empty
        
        for message in base_messages:
            client_id = str(message.metadata.dst_node_id)

            strategy_data = message.content.config_records.get("strategy_data", ConfigRecord())
            strategy_data["prev_fedce_rho"] = float(self.prev_normalized_rho_by_client.get(client_id, 0.0)) # Include previous rho for this client, default to 0.0 for the first round
            message.content.config_records["strategy_data"] = strategy_data

            if self.global_gradient is not None:
                message.content.array_records["global_gradient"] = self.global_gradient # Include global gradient for FedCE calculations on the client side

        return base_messages
    
    def aggregate_train(self, server_round, replies):
        return super().aggregate_train(server_round, replies)

    