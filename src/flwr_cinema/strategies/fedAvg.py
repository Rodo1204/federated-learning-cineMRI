from flwr.serverapp.strategy import FedAvg
from flwr_cinema.strategies.common import CinemaStrategyMixin


class FedStrategy(CinemaStrategyMixin, FedAvg):
    """Federated strategy combining CineMA-specific logic (step-based validation, 
    early stopping, checkpointing) with Flower's FedAvg aggregation.
    
    All parameters are forwarded through **kwargs — CinemaStrategyMixin consumes
    its own (eval_steps, primary_metric_name, etc.) and passes the rest to FedAvg
    (fraction_train, weighted_by_key, etc.).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)