import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log
import logging
from omegaconf import DictConfig
from models.cinema.cinema import ConvViT

from flwr_cinema.task import Net, load_data
from flwr_cinema.task import test as test_fn
from flwr_cinema.task import train as train_fn

# Flower ClientApp
app = ClientApp()

# def setup_logging(log_dir: Path, ):

def get_model(config: DictConfig) -> torch.nn.Module:
    """Return a new instance of the model."""
    freeze_backbone = config.model.get("freeze_backbone", True)

    model = ConvViT.from_pretrained(config=config, freeze_backbone=freeze_backbone)

    if config.model.get("freeze_backbone", False):
        if hasattr(model, 'pred_head_dict'):
            for head_params in model.pred_head_dict.parameters():
                head_params.requires_grad = True

    return model

def get_client_class(config: DictConfig):
    """Dynamically import FederatedClient based on task type.
    
    Args:
        config: Configuration with config.task field.
        
    Returns:
        FederatedClient class from appropriate module (classification or regression).
    """
    task = getattr(config, "task", "regression")
    
    if task == "classification":
        from flwr_cinema.tasks.classification.client import FederatedClient
    else:
        from flwr_cinema.tasks.regression.client import FederatedClient
    
    return FederatedClient

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
