"""FedProx aggregation strategy with CinemaStrategyMixin.

Clients add a proximal term (mu * ||w - w^t||^2) to their local loss,
penalizing deviation from the global model.
"""
from flwr.serverapp.strategy import FedProx
from flwr_cinema.strategies.common import CinemaStrategyMixin


class FedProxStrategy(CinemaStrategyMixin, FedProx):
    """FedProx + CinemaStrategyMixin (step validation, early stopping, checkpoints).

    MRO: CinemaStrategyMixin -> FedProx -> FedAvg.
    All params forwarded via **kwargs.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)