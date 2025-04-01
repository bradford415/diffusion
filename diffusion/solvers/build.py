from collections.abc import Iterable

from diffusion.solvers import optimizer_map, scheduler_map


def build_solvers(
    model_params: Iterable,
    optimizer_params: dict[str, any],
    scheduler_params: dict[str, any],
):
    """Builds the optimizer and learning rate scheduler based on the provided parameters
    from solver.config

    Args:
        optimizer_params: the parameters used to build the optimizer
        scheduler_params: the parameters used to build the learning rate scheduler
        optimizer: the optimizer used during training
    """
    optimizer_name = optimizer_params["name"]
    scheduler_name = scheduler_params["name"]

    # Delete name key so we can easily unpack the parameters
    del optimizer_params["name"]
    del scheduler_params["name"]

    # Build optimizer
    if optimizer_name in optimizer_map:
        optimizer = optimizer_map[optimizer_name](model_params, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # TODO: impelement configs for warmup_cosine_decay
    # Build scheduler
    if scheduler_name in scheduler_map:
        scheduler = scheduler_map[scheduler_name](optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    return optimizer, scheduler

    if scheduler_name == "warmup_cosine_decay":  # TODO: put this in config
        raise NotImplementedError
        # return warmup_cosine_decay(
        #     optimizer,
        #     warmup_steps=scheduler_params["warmup_steps"],
        #     total_steps=scheduler_params["total_steps"],
        # )
        return scheduler_mapreduce_lr_on_plateau(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name }")
