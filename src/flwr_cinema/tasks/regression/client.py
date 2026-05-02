from flwr_cinema.tasks.classification.client import TaskClient


class RegressionClient(TaskClient):
    """Regression federated client.

    Reuses the generic task client implementation in TaskClient.
    """
