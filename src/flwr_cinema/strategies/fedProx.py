from flwr.serverapp.strategy import FedProx
from flwr_cinema.strategies.common import CinemaStrategyMixin


class FedProxStrategy(CinemaStrategyMixin, FedProx):
    """Federated strategy combining CineMA-specific logic (step-based validation, 
    early stopping, checkpointing) with Flower's FedProx aggregation.
    
    All parameters are forwarded through **kwargs — CinemaStrategyMixin consumes
    its own (eval_steps, primary_metric_name, etc.) and passes the rest to FedProx
    (fraction_train, weighted_by_key, etc.).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)