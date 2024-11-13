import copy
import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import functional as F

from diffusion.data.transforms import to_numpy_image
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
        eval_batch_size: int = 16,
        num_eval_samples: int = 16,
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

        self.output_path = output_path

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
        dataloader_val: data.DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str] = None,
        start_step: int = 1,
        steps: int = 700000,
    ):
        """Trains a ddpm model

        Specifically, this method trains a model for n epochs and evaluates on the validation set.
        A model checkpoint is saved at user-specified intervals

        Args:
            diffusion_model: the diffusion model (ddpm) to train; this class should contain the denoise_model
                             (unet) as an attribute
            dataloader_train: torch dataloader to loop through the train dataset
            dataloader_train: torch dataloader to loop through the val dataset
            optimizer: Optimizer which determines how to update the weights
            checkpoint_path: path to the weights file to resume training from 
            start_step: the step to start the training on; starting at 1 is a good default because it makes
                        checkpointing and calculations more intuitive
            steps: number of stpes to train for; unless starting from a checkpoint, this will be the number of epochs to train for
        """
        ema_model = copy.deepcopy(diffusion_model)
        
        if checkpoint_path is not None:
            start_step = self._load_model(checkpoint_path=checkpoint_path, diffusion_model=diffusion_model, optimizer=optimizer, ema_model=ema_model)

        # Infinitely cycle through the dataloader to train by steps
        # i.e., once all the samples have been sampled, it will start over again
        dataloader_train = self._cycle(dataloader_train)
        dataloader_val = self._cycle(dataloader_val)

        log.info("\nTraining started\n")
        total_train_start_time = time.time()

        # Starting the epoch at 1 makes calculations more intuitive
        for step in range(start_step, steps + 1):
            diffusion_model.train()  # TODO: I think this will also set the unet model to train mode but I should verify

            # Train one epoch
            train_loss = self._train_one_step(
                diffusion_model, dataloader_train, optimizer, step
            )

            if step % self.log_intervals == 0:
                val_loss = self._evaluate_one_step(diffusion_model, dataloader_val)
                log.info(
                    "step: %4d \ttrain loss: %4.5f \tval loss: %4.5f",
                    step,
                    train_loss,
                    val_loss,
                )

            # Update the EMA model's weights
            self._calculate_ema(diffusion_model, ema_model, self.ema_decay)

            # Evaluate the diffusion model at a specified interval
            if step % self.eval_intervals == 0:
                # Evaluate the model on the validation set
                log.info("\nsampling — step %d", step)

                # Generate images
                self._sample(ema_model, step)

            # TODO: also save and replace best model based on fid score

            # Save the model every ckpt_steps
            if step % self.ckpt_steps == 0:
                breakpoint
                ckpt_path = Path(self.output_path) / "checkpoints"
                ckpt_path.mkdir(parents=True, exist_ok=True) 
                self._save_model(
                    diffusion_model,
                    optimizer,
                    ema_model,
                    step,
                    save_path=ckpt_path / f"checkpoint{step:07}.pt",
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
        current_step: int,
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

        # Visualize a batch of images to verify the agumentations are correct
        if current_step == 1:
            self._visualize_batch(samples)

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

        return loss

    @torch.inference_mode
    def _evaluate_one_step(
        self,
        diffusion_model: nn.Module,
        dataloader_val: Iterable,
    ):
        """TODO

        Args:
            model: Model to train
            dataloader_train: Dataloader for the validation set
        """
        diffusion_model.eval()
        samples = next(dataloader_val)

        # the torch cifar dataset returns the image and label; we don't need the label for diffusion
        if len(samples) == 2:
            samples = samples[0]

        samples = samples.to(self.device)

        # Forward pass through diffusion model; noises and denoises the image; diffusion_model contains the
        # the unet model for denoising
        loss = diffusion_model(samples)

        return loss

    @torch.inference_mode
    def _sample(
        self,
        ema_model: nn.Module,
        current_step: int,
    ):
        """Denoises pure noise to generate and save images

        Args:
            ema model: model to sample from; typically only the ema model is used

        Returns:
        """
        # Eval is only performed with the ema model
        ema_model.eval()

        # TODO
        gen_images_output = Path(self.output_path) / "samples" / f"step_{current_step}"
        gen_images_output.mkdir(parents=True, exist_ok=True)

        # Split the number of samples to generate into a list of batches
        eval_batch_sizes = self._num_samples_to_batches(self.num_eval_samples)
        log.info(
            "Generating %d images using the following batch sizes: %s",
            self.num_eval_samples,
            eval_batch_sizes,
        )
        generated_images = []
        for index, batch_size in enumerate(eval_batch_sizes):
            log.info("Processing batch %d/%d", index + 1, len(eval_batch_sizes))
            generated_images.append(ema_model.sample_generation(batch_size=batch_size))

        all_images = torch.cat(generated_images, dim=0)

        # # Convert generated images to viewable shape
        # all_images = all_images.permute(0, 2, 3, 1) # (b, c, h, w) -> (b, h, w, c)
        # all_images *= 255.0
        # all_images = all_images.detach().cpu().numpy().astype(np.uint8)

        all_images = to_numpy_image(all_images)

        # for index, image_set in enumerate(generated_images):
        save_gen_images(
            all_images,
            self.num_eval_samples**0.5,
            str(gen_images_output / "generated_images.png"),
        )

    def _calculate_ema(
        self, source_model: nn.Module, target_model: nn.Module, decay: float
    ):
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

    def _cycle(self, dataloader: data.DataLoader):
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
        """Save the ddpm and ema model
        
        Args:
            diffusion_model: the diffusion model being trained
            optimizer: the optimizer used during training
            ema_model: ema which is used for the sampling process
            current_step: the current step the training is on when
                          the model is saved
        """
        torch.save(
            {
                "model": diffusion_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema_model": ema_model.state_dict(),
                "step": current_step + 1, # + 1 bc when we resume training we want to start at the next step
            },
            save_path,
        )
        
    def _load_model(
        self, checkpoint_path: str, diffusion_model, optimizer, ema_model
    ):
        """Load the ddpm model to resume training or generate new images from
        
        Args:
            checkpoint_path: path to the weights file to resume training from 
            diffusion_model: the diffusion model being trained
            optimizer: the optimizer used during training
            ema_model: ema which is used for the sampling process
            current_step: the current step the training is on when
                          the model is saved
        """
        # Load the torch weights
        weights = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # load the state dictionaries for the necessary training modules
        diffusion_model.load_state_dict(weights["model"])
        optimizer.load_state_dict(weights["optimizer"])
        ema_model.load_state_dict(weights["ema_model"])
        start_step = weights["step"]
        
        log.info("NOTE: A checkpoint file was provided, the model will resume training loading model at step %d", start_step)        
        
        return start_step


    def _visualize_batch(
        self, samples: torch.Tensor,
    ):
        """Visualize a batch of images after data augmentation; sthis helps manually verify
        the data augmentations are working as intended on the images and boxes

        Args:
            samples: tensor of a batch of images (b, c, h, w)
        """
        samples = to_numpy_image(samples)
        
        save_dir = Path(self.output_path) / "train_images"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_gen_images(samples, sqrt_num=np.sqrt(samples.shape[0]), save_name=str(save_dir / "train_batch.png"))

