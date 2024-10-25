import copy
import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import functional as F
from tqdm import tqdm

from diffusion.visualize import save_gen_images

log = logging.getLogger(__name__)


class Trainer:
    """Trainer TODO: comment"""

    def __init__(
        self,
        output_path: str,
        device: torch.device = torch.device("cpu"),
        max_grad_norm: float = 1.0,
        ema_decay: float = 0.9999,
        ckpt_steps: int = 1000,
        eval_intervals: int = 40,
        eval_batch_size: int = 4,
        num_eval_samples: int = 25,
        logging_intervals: int = 20,
    ):
        """Constructor for the Trainer class

        Args:
            output_path: Path to save the train outputs
            device: which device to use
            max_grad_norm: the l2 magnitude to clip the gradients
            ema_decay: TODO
            eval_intervals: number of steps to evaluate the model after during training
            num_eval_samples: number of samples to evaluate; must be a perfect square
            logging_interval: number of steps to log the training progress after

        """
        ## TODO: PROBALBY REMOVE THESE Initialize training objects
        # self.optimizer = optimizer_map[optimizer]
        # self.lr_scheduler = "test"

        self.device = device

        self.output_paths = output_path

        self.ckpt_steps = ckpt_steps

        # Logging params
        self.eval_intervals = eval_intervals
        self.log_intervals = logging_intervals

        self.max_grad_norm = max_grad_norm

        self.eval_batch_size = eval_batch_size
        self.num_eval_samples = num_eval_samples

        self.ema_decay = ema_decay

        # TODO: Implement FID evaluator

    def train(
        self,
        diffusion_model: nn.Module,
        dataloader_train: data.DataLoader,
        optimizer: torch.optim.Optimizer,
        start_step: int = 1,
        steps: int = 700000,
        ckpt_steps: int = 1000,
    ):
        """Trains a ddpm model

        Specifically, this method trains a model for n epochs and evaluates on the validation set.
        A model checkpoint is saved at user-specified intervals

        Args:
            diffusion_model: the diffusion model (ddpm) to train; this class should contain the denoise_model
                             (unet) as an attribute
            dataloader_train: torch dataloader to loop through the train dataset
            optimizer: Optimizer which determines how to update the weights
            start_step: the step to start the training on; starting at 1 is a good default because it makes
                        checkpointing and calculations more intuitive
            steps: number of stpes to train for; unless starting from a checkpoint, this will be the number of epochs to train for
            ckpt_steps: save the model after n steps
        """
        ema_model = copy.deepcopy(diffusion_model)

        # Infinitely cycle through the dataloader to train by steps
        # i.e., once all the samples have been sampled, it will start over again
        dataloader_train = self._cylce(dataloader_train)

        log.info("\nTraining started\n")
        total_train_start_time = time.time()

        # TODO: Visualize the first batch for each dataloader; manually verifies data augmentation correctness
        # self._visualize_batch(dataloader_train, "train", class_names)
        # self._visualize_batch(dataloader_val, "val", class_names)

        # Starting the epoch at 1 makes calculations more intuitive
        for step in range(start_step, steps + 1):
            diffusion_model.train()  # TODO: I think this will also set the unet model to train mode but I should verify

            # Train one epoch
            self._train_one_step(
                diffusion_model,
                dataloader_train,
                optimizer,
            )

            # Update the EMA model's weights
            self._calculate_ema(diffusion_model, ema_model, self.ema_decay)

            # Evaluate the diffusion model at a specified interval
            if step % self.eval_interval == 0:
                # Evaluate the model on the validation set
                log.info("\nEvaluating — step %d", step)
                ######################## TODO start here  and fill out evaluate
                metrics_output = self._evaluate(ema_model)

            # TODO: also save and replace best model based on fid score

            # Save the model every ckpt_steps
            if steps % ckpt_steps == 0:
                ckpt_path = self.output_paths["output_dir"] / f"checkpoint{step:07}.pt"
                self._save_model(
                    diffusion_model,
                    optimizer,
                    ema_model,
                    step,
                    save_path=ckpt_path,
                )

        # Entire training time
        total_time = time.time() - total_train_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log.info(
            "Training time for %d steps (h:mm:ss): %s ",
            start_step - steps,
            total_time_str,
        )

        log.info("training complete")

    def _train_one_step(
        self,
        diffusion_model: nn.Module,
        dataloader_train: Iterable,
        optimizer: torch.optim.Optimizer,
    ):
        """Train one step (one batch of samples)

        Args:
            model: Model to train
            dataloader_train: Dataloader for the training set
            optimizer: Optimizer to update the models weights
        """
        samples = next(dataloader_train)

        # the torch cifar dataset returns the image and label; we don't need the label for diffusion
        if len(samples) == 2:
            samples = samples[0]

        samples = samples.to(self.device)

        # Forward pass through diffusion model; noises and denoises the image; diffusion_model contains the
        # the unet model for denoising
        loss = diffusion_model(samples)

        loss.backward()

        # Clips the magnitude (l2) of the gradients to max_norm if the magnitude is greater than this value;
        # this works by calculating the unit vector of the gradients (grad_vec / ||grad_vec||) and if this
        # is greater than max_norm, the gradients are clipped by dividing by ||grad_vec|| which will create
        # a unit vector; a unit vector has a l2 magnitude of 1 so if we take the l2 of all the gradients after
        # clipping the magnitude should be 1; if max_norm is not 1, a unit vector is created and then multiplied
        # by this max_norm value; this works because the gradients are now a unit vector so multiplying by
        # this constant will the scale the clipped gradients linearly; the gradients of the models parameters
        # are clipped (i.e., parameter.grad) not the parameters itself, therefore, clip_grad_norm_
        # should be placed after loss.backward but before optimizer.step
        nn.utils.clip_grad_norm_(
            diffusion_model.parameters(), max_norm=self.max_grad_norm
        )

        optimizer.step()
        optimizer.zero_grad()

    @torch.inference_mode
    def _evaluate(
        self,
        ema_model: nn.Module,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Denoises pure noise to generate and save images

        Args:
            ema model: model to sample from; typically only the ema model is used

        Returns:
            A Tuple of the (prec, rec, ap, f1, and class) per class
        """
        # Eval is only performed with the ema model
        ema_model.eval()

        # Split the number of samples to generate into a list of batches
        eval_batch_sizes = self._num_samples_to_batches(self.num_eval_samples)

        generated_images = []
        for batch_size in eval_batch_sizes:
            generated_images.append(ema_model.sample_generation(batch_size=batch_size))

        for image_set in enumerate(generated_images):
            save_gen_images(image_set, self.num_eval_samples**2, "generated_images.png")

        # TODO: Calculate fid score
        return None

    def _calculate_ema(source_model: nn.Module, target_model: nn.Module, decay: float):
        """Calculate the exponential moving average (ema) from a source model's weights
        and the current target_model's weights; the updated weights are stored in target_model

        TODO: write about the benefit here

        Args:
            source_model: the model to calcuate the ema on
            target_model: the model to store the ema of the weights
            decay: the ema decay TODO clarify this better
        """
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay + source_dict[key].data * (1 - decay)
            )

    def _num_samples_to_batches(self, num_samples: int):
        """Create a list of batch sizes and the remaining batch size at the last index;
        this is useful to pass the number of eval samples by batch

        Example: num_samples = 25 and batch_size = 16 -> [16, 9]

        Args:
            num_samples: number of samples to generate images of
        """
        groups = num_samples // self.eval_batch_size
        remainder = num_samples % self.eval_batch_size
        batch_arr = [self.eval_batch_size] * groups
        if remainder > 0:
            batch_arr.append(remainder)
        return batch_arr

    def _cylce(self, dataloader: data.DataLoader):
        """This function infinitely cycles through a torch dataloader. This is useful
        when you want to train by steps rather than epochs.

        It functions exactly the same as training by epochs; i.e., the dataloader
        will still sample all of the samples in the dataset first and once it reaches the end of
        the dataset it will restart; if shuffle is specified in the dataloader, the shuffling will
        still be used randomly to mix samples in batches just like training by epochs

        Args:
         dataloader: A torch dataloader
        """
        while True:
            for data in dataloader:
                yield data

    def _save_model(
        self, diffusion_model, optimizer, ema_model, current_step, save_path
    ):
        """TODO"""
        torch.save(
            {
                "model": diffusion_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema_model": ema_model.state_dict(),
                "step": current_step,
            },
            save_path,
        )

    def _visualize_batch(
        self, dataloader: data.DataLoader, split: str, class_names: List[str]
    ):
        """Visualize a batch of images after data augmentation; sthis helps manually verify
        the data augmentations are working as intended on the images and boxes

        Args:
            dataloader: Train or val dataloader
            split: "train" or "val"
            class_names: List of class names in the ontology
        """
        valid_splits = {"train", "val"}
        if split not in valid_splits:
            raise ValueError("split must either be in valid_splits")

        samples, targets = next(iter(dataloader))
        plots.visualize_norm_img_tensors(
            samples,
            targets,
            class_names,
            self.output_paths["output_dir"] / f"{split}-images",
        )
