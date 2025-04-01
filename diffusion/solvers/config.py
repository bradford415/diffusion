import ml_collections

__all__ = ["kl_vae_f_4_d_8_lsun_bedroom"]


def kl_vae_f_4_d_8_lsun_bedroom():
    """Returns the solver parameters used for the Autoencoder KL with f=4, d=8 on the LSUN Bedroom dataset

    Parameters defined at: TODO
    """
    config = ml_collections.ConfigDict()

    # Optimizer params
    # TODO: update with the correct params
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = "sgd"
    config.optimizer.lr = 0.1
    config.optimizer.weight_decay = 1e-4  # 0.0001
    config.optimizer.momentum = 0.9

    # Scheduler params
    # TODO
    config.step_lr_on = "epochs"  # step the lr after n "epochs" or "steps"
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.name = "reduce_lr_on_plateau"
    config.lr_scheduler.mode = "min"
    config.lr_scheduler.factor = 0.1
    config.lr_scheduler.patience = 5  # epochs of no improvement

    return config
