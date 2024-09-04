import logging
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    output_path: str,
    model: nn.Module,
    dataloader_test: Iterable,
    class_names: List,
    device: torch.device = torch.device("cpu"),
) -> None:
    """A single forward pass to evluate the val set after training an epoch

    Args:
        model: Model to train
        criterion: Loss function; only used to inspect the loss on the val set,
                    not used for backpropagation
        dataloader_val: Dataloader for the validation set
        device: Device to run the model on
    """
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (true positives, cls_confs, cls_labels)
    for steps, (samples, targets) in enumerate(dataloader_test):
        samples = samples.to(device)


    return None


def load_model_state_dict(model: nn.Module, weights_path: str):
    """Load the weights of a trained or pretrained model from the state_dict file;
    this could be from a fully trained model or a partially trained model that you want
    to resume training from.

    Args:
        model: The torch model to load the weights into
        weights_path:
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Select device for inference

    state_dict  = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict["model"])

    return model
