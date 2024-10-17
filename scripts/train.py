import datetime
import logging
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from diffusion.data.cifar import build_cifar
from diffusion.models import Unet, DDPM
from diffusion.trainer import Trainer
from diffusion.utils import reproduce

model_map: Dict[str, Any] = {"unet": Unet}

optimizer_map = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

loss_map = {
    "cross_entropy": nn.CrossEntropyLoss(),
}

scheduler_map = {
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "lambda_lr": torch.optim.lr_scheduler.LambdaLR,  # Multiply the initial lr by a factor determined by a user-defined function; it does NOT multiply the factor by the current lr, always the initial lr
}

# Initialize the root logger
log = logging.getLogger(__name__)


def main(base_config_path: str, model_config_path: Optional[str] = None):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired configuration file
        model_config_path: TODO: path to the detection model configuration file

    """
    # Load configuration files
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    if model_config_path is not None:
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
    else:
        model_name = base_config["model_name"]
        model_config = base_config["model_params"][model_name]

    # Initialize paths
    output_path = (
        Path(base_config["output_path"])
        / base_config["exp_name"]
        / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "training.log"

    # Dictionary of logging parameters; used to log training and evaluation progress after certain intervals
    logging_intervals = base_config["logging"]

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    log.info("Initializing...\n")

    # Apply reproducibility seeds
    reproduce.reproducibility(**base_config["reproducibility"])

    # Set cuda parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {"batch_size": base_config["train"]["batch_size"], "shuffle": True}
    val_kwargs = {
        "batch_size": base_config["validation"]["batch_size"],
        "shuffle": False,
    }

    if use_cuda:
        log.info("Using %d GPU(s): ", len(base_config["cuda"]["gpus"]))
        for gpu in range(len(base_config["cuda"]["gpus"])):
            log.info("    -%s", torch.cuda.get_device_name(gpu))

        cuda_kwargs = {
            "num_workers": base_config["dataset"]["num_workers"],
            "pin_memory": True,
        }

        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
    else:
        log.info("Using CPU")

    # Create train and val dataset; cifar does not have a val set so we can use test for this
    common_dataset_kwargs = {
        "root": base_config["dataset"]["root"],
        "debug_mode": base_config["debug_mode"],
    }

    dataset_name = base_config["dataset"]["name"]
    if dataset_name == "cifar10":
        dataset_train = build_cifar("cifar10", "train", **common_dataset_kwargs)
        dataset_val = build_cifar("cifar10", "test", **common_dataset_kwargs)
    elif dataset_name == "cifar100":
        dataset_train = build_cifar("cifar10", "train", **common_dataset_kwargs)
        dataset_val = build_cifar("cifar10", "val", **common_dataset_kwargs)
    else:
        ValueError(f"Dataset {dataset_name} not recognized.")

    dataloader_train = DataLoader(
        dataset_train,
        drop_last=True,
        **train_kwargs,
    )
    dataloader_val = DataLoader(
        dataset_val,
        drop_last=True,
        **val_kwargs,
    )

    # Initialize model
    model = model_map[model_name](**model_config)
    model.to(device)

    ## TODO: Apply weights initialization

    # Extract the train arguments from base config
    train_args = base_config["train"]

    # Extract the learning parameters such as lr, optimizer params and lr scheduler
    learning_config = train_args["optimization_config"]
    learning_params = base_config[learning_config]

    # Initialize training objects
    optimizer = _init_training_objects(
        model_params=model.parameters(),
        optimizer=learning_params["optimizer"],
        learning_rate=learning_params["learning_rate"],
        weight_decay=learning_params["weight_decay"],
    )

    denoise_model = Unet(**base_config["model_params"]["unet"])
    diffusion_model = DDPM(**base_config["model_params"]["ddpm"])

    trainer = Trainer(
        output_path=str(output_path),
        device=device,
        logging_intervals=logging_intervals,
        ema_decay=base_config["model_params"]["ema_decay"],
        ckpt_steps = train_args["ckpt_steps"],
        eval_intervals=train_args["eval_intervals"],
        num_eval_samples=train_args["num_eval_samples"],
        logging_interval=base_config["logging_interval"]
    )

    # Save configuration files
    reproduce.save_configs(
        config_dicts=[base_config, model_config],
        save_names=["base_config.json", "model_config.json"],
        output_path=output_path / "reproduce",
    )

    # Build trainer args used for the training
    trainer_args = {
        "model": model,
        # "criterion": criterion,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        # "optimizer": optimizer,
        **train_args["epochs"],
    }
    trainer.train(**trainer_args)


def _init_training_objects(
    model_params: Iterable,
    optimizer: str = "sgd",
    scheduler: str = "step_lr",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    lr_drop: int = 200,
):
    optimizer = optimizer_map[optimizer](
        model_params, lr=learning_rate, weight_decay=weight_decay
    )

    return optimizer


if __name__ == "__main__":
    Fire(main)
