import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml
from fire import Fire
from torch import nn
from torch.utils.data import DataLoader

from diffusion.data import create_dataset
from diffusion.models import DDPM, Unet, vae_map
from diffusion.models.layers import init_weights
from diffusion.solvers import solver_configs
from diffusion.trainer import Trainer
from diffusion.utils import reproduce

model_map: Dict[str, Any] = {"unet": Unet}

# TODO: change this to ML collections configs
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


def main(base_config_path: str, model_config_path: str = None):
    """Entrypoint for the project

    Args:
        base_config_path: path to the desired configuration file
        model_config_path: TODO: path to the detection model configuration file

    """
    # Load configuration files
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    dev_mode = base_config["dev_mode"]

    # Initialize paths
    output_path = (
        Path(base_config["output_dir"])
        / base_config["exp_name"]
        / f"{datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "training.log"

    # Configure logger that prints to a log file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    if dev_mode:
        log.info("NOTE: executing in dev mode")
        base_config["train"]["batch_size"] = 2

    log.info("Initializing...\n")

    log.info("writing outputs to %s", str(output_path))

    # Apply reproducibility seeds
    reproduce.reproducibility(**base_config["reproducibility"])

    # Extract solver config; TODO
    # solver_config = solver_configs[base_config["train"]["solver_config"]]()

    # Set dataset parameters
    train_kwargs = {
        "batch_size": base_config["train"]["batch_size"],
        "shuffle": True,
        "num_workers": base_config["dataset"]["num_workers"] if not dev_mode else 0,
    }
    val_kwargs = {
        "batch_size": base_config["validation"]["batch_size"],
        "shuffle": False,
        "num_workers": base_config["dataset"]["num_workers"] if not dev_mode else 0,
    }

    # Set device specific characteristics
    use_cpu = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using %d GPU(s): ", len(base_config["cuda"]["gpus"]))
        for gpu in range(len(base_config["cuda"]["gpus"])):
            log.info("    -%s", torch.cuda.get_device_name(gpu))
    elif torch.mps.is_available():
        base_config["dataset"]["root"] = base_config["dataset"]["root_mac"]
        del base_config_path["dataset"]["root_mac"]
        device = torch.device("mps")
        log.info("Using: %s", device)
    else:
        use_cpu = True
        device = torch.device("cpu")
        log.info("Using CPU")

    if not use_cpu:
        gpu_kwargs = {
            "pin_memory": True,
        }

        train_kwargs.update(gpu_kwargs)
        val_kwargs.update(gpu_kwargs)

    # Create train and val dataset
    dataset_params = base_config["dataset"]
    dataset_name = dataset_params.get("name")
    dataset_kwargs = dataset_params["params"]

    if dataset_name is not None:
        dataset_train = create_dataset(
            dataset_name,
            split="train",
            root=dataset_params["root"],
            dev_mode=dev_mode,
            **dataset_kwargs,
        )
        dataset_val = create_dataset(
            dataset_name,
            split="val",
            root=dataset_params["root"],
            dev_mode=dev_mode,
            **dataset_kwargs,
        )
    else:
        raise ValueError("dataset name not specified in the config file")

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

    exit()

    # Extract the train arguments from base config
    train_args = base_config["train"]

    # Extract the learning parameters such as lr, optimizer params and lr scheduler
    learning_config = train_args["learning_config"]
    learning_params = base_config[learning_config]

    ############## START HERE build vae ############

    # Initalize models
    diffusion_model = DDPM(
        denoise_model, device=device, **model_config["model_params"]["ddpm"]
    ).to(device)

    # Compute and log the number of params in the model
    reproduce.model_info(diffusion_model)

    optimizer, lr_scheduler = build_solvers(
        model.parameters(), solver_config.optimizer, solver_config.lr_scheduler
    )

    trainer = Trainer(
        output_path=str(output_path),
        device=device,
        ema_decay=model_config["model_params"]["ema_decay"],
        ckpt_steps=train_args["ckpt_steps"],
        eval_intervals=train_args["eval_intervals"],
        num_samples=train_args["num_samples"],
        logging_intervals=base_config["logging_intervals"],
        sample_batch_size=train_args["sampling_batch_size"],
    )

    # Save configuration files
    reproduce.save_configs(
        config_dicts=[base_config, model_config],
        save_names=["base_config.yaml", "model_config.yaml"],
        output_path=output_path / "reproduce",
    )

    # Build trainer args used for the training
    trainer_args = {
        "diffusion_model": diffusion_model,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "optimizer": optimizer,
        "checkpoint_path": train_args["checkpoint_path"],
        "start_step": train_args["start_step"],
        "steps": train_args["steps"],
    }

    # Train the ddpm model
    trainer.train(**trainer_args)


if __name__ == "__main__":
    Fire(main)
