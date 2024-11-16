import math
import random
from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from diffusion.data.transforms import Unnormalize


class DDPM(nn.Module):
    """DDPM model which is responsible for noising images and denoising with unet

    In general, diffusion models randomly apply noise to an image at certain timestep t and
    the denoising model (unet) tries to predict the noise that was added (it does NOT predict the purified image).
    The loss function compares the ground truth noise that was added to the predicted noise at the timestep,
    therefore, the denoise model tries to learn how to predict the noise at any timestep. Once the model has
    been trained, it can sample (inference) pure random gaussian noise and iteratively denoise the image
    by the reverse defusion process  starting at the maximum timestep (normally T=1000). Once it reaches T=0,
    an image should be generated.

    DDPM works in 3 steps:
        1. Forward process (algorithm 1):
            q(x_{1:T} | x_{0}) is the forward diffusion process which adds noise according to a variance
            schedule β1, . . . , βT; this allows noise to be added at a specific timestep instantly rather
            than iteratively
        2. Reverse process (algorithm 1):
            p_θ(x_{0:T}) is the reverse process TODO flesh this out
        3. Sampling (algorithm 2): This is basically the evaluation after training where we generate new images;
           the model takes random noise and applies the learned reserse steps to iteratively
           refine the noise into a coherent sample


    """

    def __init__(
        self,
        denoise_model,
        *,
        image_size: Union[List, Tuple],
        timesteps=1000,
        objective="pred_noise",
        variance_schedule="sigmoid",
        device=torch.device("cpu"),
    ):
        """
        Args:
            model: A torch Unet model; theoretically this could be any encoder-decoder
                   model which downsamples and upsamples back to the original input size
            image_size: tuple of orignal image size in the form of (height, width)
            timesteps: max number of timesteps for diffusion; default is T=1000
            objective: the objective the unet model will predict; the ddpm paper predicts noise but
                       supposedly pred_v is better
            variance_schedule: the variance schedule to use; the variance at each timestep is the betas
        """
        super().__init__()

        self.denoise_model = denoise_model

        # Input image channels (e.g., rgb = 3)
        self.channels = self.denoise_model.channels

        self.image_size = image_size

        # TODO: quick comment
        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if variance_schedule == "linear":
            variance_schedule_fn = linear_beta_schedule
        elif variance_schedule == "cosine":
            variance_schedule_fn = cosine_beta_schedule
        elif variance_schedule == "sigmoid":
            variance_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown variance schedule {variance_schedule}")

        # Below we are essenttially defining the necessary alpha and beta variables of the
        # same shape so that we can sample all of the required variables for the forward
        # process at a specific time step

        # Define the variance schedule; minumum and maximum noise to add
        betas = variance_schedule_fn(timesteps)
        betas = betas.to(device)

        # Define alphas for the forward process;
        # these are defined in the ddpm paper in equation 4 and the paragraph above
        # α_t = 1 - β and α¯_t = cumulative product of α at timestep t - NOTE: α¯ = alphas_cumprod
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Define α_{t-1}
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Number of timesteps for diffusion; typically T = 1000
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        # Store betas and alphas as part of the model
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Compute sqrt(α¯) from equation 4
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))

        # Compute sqrt(1 - α¯) for sampling x_t (the noised image at timestep t from figure 2)
        # and parametizing the mean, mu
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # START HERE, I don't think this log param is used for anything
        # register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))

        # sqrt_recip_alphas_cumprod & sqrt_recipm1_alphas_cumprod are used to compute
        # equation 15 which estimates the denoised image x_0;
        # x_0 is used to compute the predicted mean in equation 7

        # Equation 15  left term after distributing 1 / sqrt(α¯);
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))

        # Equation 15 right term after distributing 1 / sqrt(α¯);
        # sqrt(1-α)/sqrt(α) => sqrt((1-α)/α) => sqrt(1/α - α/α) => sqrt(1/α - 1)
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # Compute the posterior variance (equation 7) q(x_{t-1} | x_t, x_0)
        # which is a parameter of the Normal distribution (equation 6)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        # NOTE: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        #       e.g., setting 0 to a very small number avoids log of 0
        register_buffer(
            "posterior_log_variance_clipped",
            # torch.log(posterior_variance.clamp(min=1e-20)),
            torch.log(
                torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
            ),
        )

        # Used to calculate the posterior mean from equation 7
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # unnormalize [-1, 1] -> [0, 1]; After sampling, return data to [0,1] to visualize
        self.unnormalize = Unnormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, timestep, pred_noise):
        """Use equation 15 to estimate the denoised image at timestep 0, x_0,
        from the noise prediction (epsilon). This is NOT the final image generation, it is used to
        help calculate the noise mean.

        NOTE: The parameters below seem to be rearranged compared to the eq 15;
              a short simplification is shown in the attribute definition to
              show where they come from; specifically sqrt_recipm1_alphas_cumprod
              was the attribute which confused me the most

        Args:
            x_t: noisy image at timestep t (b, c, h, w); during sampling the noisy image is
                 drawn from a random normal distribution, not noise added to a real image
            timestep: Current noise denoise timestep (b,)
            pred_noise: predicted noise from the unet model (b, c, h, w)

        """
        assert x_t.shape == pred_noise.shape

        return (
            extract_values(self.sqrt_recip_alphas_cumprod, timestep, x_t.shape) * x_t
            - extract_values(self.sqrt_recipm1_alphas_cumprod, timestep, x_t.shape)
            * pred_noise
        )

    # TODO: Implement at a later point
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # TODO: Implement at a later point
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, timestep):
        """Compute the posterior mean (mu) and posterior variance (beta)
        from equation 7

        Args:
            x_start: the estimated denoised image, x_0, from equation 15
            x_t: the noisy image at timestep t being denoised
            timestep: current denoising timestep

        Returns:
            The posterior_mean, posterior_variance, & log posterior_variance
            TODO: Put the shapes
        """
        # Calculate the posterior mean (mu) using equation 7
        posterior_mean = (
            extract_values(self.posterior_mean_coef1, timestep, x_t.shape) * x_start
            + extract_values(self.posterior_mean_coef2, timestep, x_t.shape) * x_t
        )

        # Posterior variance (beta) from equation 7
        posterior_variance = extract_values(
            self.posterior_variance, timestep, x_t.shape
        )

        posterior_log_variance_clipped = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            "fixedlarge": torch.log(
                torch.cat([self.posterior_variance[1:2], self.betas[1:]])
            ),
            "fixedsmall": self.posterior_log_variance_clipped,
        }["fixedlarge"]
        posterior_log_variance_clipped = extract_values(
            posterior_log_variance_clipped, timestep, x_t.shape
        )
        # posterior_log_variance_clipped = extract_values(
        #     self.posterior_log_variance_clipped, timestep, x_t.shape
        # )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self, sampled_noise, timestep, clip_x_start=False, rederive_pred_noise=False
    ):
        """Predict the models training objective; for ddpm this is the noise of the image;
        this works by first predicting the objective from the trained unet model and then
        using the equations from the ddpm paper to reconstruct the image

        Args:
            sampled_noise: random noise from normal distribution (b, c, h, w)
            timestep: a single denoising timestep (b,)

        Returns:
            A tuple of:
                1. the predicted noise in the data (b, c, h, w)
                2.
        """
        # model predictions (b, c, h, w);
        model_output = self.denoise_model(sampled_noise, timestep)

        # Since this is a unet model the input and output shape should be the same
        assert sampled_noise.shape == model_output.shape

        # TODO comment
        if self.objective == "pred_noise":  # noise is also epsilon
            pred_noise = model_output
            x_start = self.predict_start_from_noise(sampled_noise, timestep, pred_noise)

        # TODO implement x0 and v at a later point
        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, sampled_noise, timestep, clip_denoised=True):
        """Calculates the posterior_mean and grabs the posterior_variance
        and posterior_log_variance. These parameters are used to compute
        x_{t_1} so we can iteratively denoise the randomly sampled noise.

        Args:
            noise: random noise from normal distribution (b, c, h, w)
            timestep: a single denoising timestep (b,); unlike training this will
                      be the same value for the entire batch
            clip_denoised: TODO

        """
        # predict the models training objective; for ddpm paper this is the noise
        pred_noise, x_start = self.model_predictions(sampled_noise, timestep)

        # clip to the same scale the data was trained on [-1, 1]
        # if clip_denoised:
        #    pred_x_start.clamp_(-1.0, 1.0)

        # Calculates the posterior_mean & extracts posterior_variance, posterior_log_variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=sampled_noise, timestep=timestep
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, sample_noise, timestep: int):
        """TODO

        Args:
            sample_noise: a batch of noise at a timestep used to generate images (b, c, h, w);
                          at the start of sampling this is pure noise from a normal distribution
            timestep: the current denoising timestep (1,); unlike training this will just
                      be a scalar

        Returns:
            1. the predicted image at the previous timestep x_{t-1}
            2.
        """
        b, c, h, w = sample_noise.shape

        # repeat the timestep for the batch size (b,)
        batched_times = torch.full((b,), timestep, device=self.device, dtype=torch.long)

        # Calculate the posterior mean and grab the variances computed at the start
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            sampled_noise=sample_noise, timestep=batched_times, clip_denoised=True
        )

        # Generate random noise, z, to add to the mean (b, c, h, w); algorithm 2 line 4
        noise = (
            torch.randn_like(sample_noise) if timestep > 0 else 0.0
        )  # no noise if t == 0

        # Compute the image/sample at the previous timestep x_{t-1} (Algorithm 2 line 4);
        # the entire left term is the mean
        # TODO: not entirely sure where 0.5 and e come from; i think the e just cancels out the log
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape: Tuple, return_all_timesteps=False):
        """Iteratively denoises randomly sampled noise from a gaussian distribution;
        loops for (num_timesteps, 0]

        Args:
            shape: shape to randomly sample noise of (b, c, h, w)
            return_all_timesteps: whether to return the denoised sample at all timesteps;
                                  if false only the final denoised image is returned i.e., t=0

        Return:
            TODO

        """
        batch, device = shape[0], self.device

        # Create batch of noise from a normal distribution (b, c, h, w)
        denoised_img = torch.randn(shape, device=device)

        # Stores the denoised sample at every timestep
        images = [denoised_img]

        # Loop from (num_timesteps, 0]
        for timestep in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Denoising samples",
            total=self.num_timesteps,
            ncols=100,
        ):
            denoised_img, x_start = self.p_sample(denoised_img, timestep)
            images.append(denoised_img)

        x_0 = denoised_img if not return_all_timesteps else torch.stack(images, dim=1)

        x_0 = torch.clip(x_0, min=-1.0, max=1.0)

        # unnormalizes the generated data and converts it to a PIL image;
        # see data.transforms.Unnormalize for more details
        # ret = self.unnormalize(ret)
        return x_0

    @torch.inference_mode()
    def sample_generation(self, batch_size=16):
        """Generate images from noise samples; in the ddpm paper this is
        the sample algorithm 2, this is similar to inferencing; this method is typically
        only called from the ema_model, not the

        This is basically the evaluation/inference function but I think diffusion models
        do not really use the term evaluation.

        Args:
            batch_size: number of images to generate
        """
        (h, w), channels = self.image_size, self.channels

        # Randomly sample a batch of noise and iteratively denoise it to generate an image
        images = self.p_sample_loop(
            (batch_size, channels, h, w), return_all_timesteps=False
        )

        return images

    def q_sample(self, img_start, t, noise):
        """Noise a batch of images according to the variance schedule (forward diffusion)

        This function performs q(x_t | x_0) (eq. 4) which allows us to sample
        from an arbitrary timestep; this means we don't have to iteratively add noise;
        the authors of ddpm found it simpler and more beneficial to sample from a simplified
        objective so the actual equation used for sampling is the reparametized eq. 4 explained
        on page 3 and found in eq. 12 and algorithm 1

        Args:
            img_start: Initial image (x) without noise
            noise: Random gaussian noise (b, c, h, w)
        """

        return (
            extract_values(self.sqrt_alphas_cumprod, t, img_start.shape) * img_start
            + extract_values(self.sqrt_one_minus_alphas_cumprod, t, img_start.shape)
            * noise
        )

    def q_mean_variance(self, x_0, x_t, t):
        # TODO: might remove from different repo
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_log_var_clipped

    def p_losses(self, img_start, timestep, noise=None):
        """Main method that adds noise to the image and denoises it through the unet model

        Args:
            img_start: Initial input image to the model without any noise (b, c, h, w)
            timestep: Random timestep sampled from [0, num_timesteps); shape (b,)
            noise: TODO: remove this if it's not used anywhere
        """
        b, c, h, w = img_start.shape

        # Sample random noise from a normal distribution (b, c, h, w)
        if noise is None:
            noise = torch.randn_like(img_start)

        # Noise a batch of images
        noised_images = self.q_sample(img_start=img_start, t=timestep, noise=noise)

        # Predict the noise added to the image
        model_out = self.denoise_model(noised_images, timestep)

        # TODO: Incorporate different objectives but for now use the original
        #       objective which is to predict the noise added to the image
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = img_start
        elif self.objective == "pred_v":
            v = self.predict_v(img_start, timestep, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # Calculate the element-wise squared error loss (pred - true)^2 (b, c, h, w)
        loss = F.mse_loss(model_out, target, reduction="none")

        # Calculate the mean across c, h, & w (b, c*h*w); this allows us to weight each sample
        # e.g., ddpm paper mentions that reducing the weight of samples at a low timestep could be effective;
        # many implementations weight these samples but for now we'll keep it simple and won't apply weighting
        loss = loss.mean(axis=[1, 2, 3])

        return loss.mean()

    def forward(self, image):
        """Forward pass of DDPM; the main diffusion logic is perform in self.p_losses()

        Args:
            image: Batch of original images without noise (b, c, h, w)
        """
        (
            b,
            c,
            h,
            w,
        ) = image.shape
        device = image.device
        img_size = (h, w)
        assert (
            h == img_size[0] and w == img_size[1]
        ), f"height and width of image must be {img_size}"

        # Batch of random (uniformly sampled) timesteps [0, num_timesteps); shape (b,)
        timestep = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(image, timestep)


def extract_values(values: torch.Tensor, timestep: torch.Tensor, x_shape):
    """Extract parameter values at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

    This is mostly used to grab the parameters calculated in the GaussianDiffusion.__init__
    that are used for the forward and reverse diffusion process

    Args:
        values: A tensor of values to extract a specific timestep from (b, num_timesteps)
                TODO: verify this shape
        timestep: TODO verify this shape

    Returns:
        A tensor of the extracted values; shape is (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes
    """
    # torch.gather visual: https://stackoverflow.com/questions/50999977/what-does-gather-do-in-pytorch-in-layman-terms
    out = torch.gather(values, index=timestep, dim=0).float()

    return out.view([timestep.shape[0]] + [1] * (len(x_shape) - 1))


# Variance schedulers to gradually add noise for the foward diffusion process;
# the schedulers set the betas (minimum and maximum noise) which is the diffusion rate
# TODO: should probably extract this to its own module
def linear_beta_schedule(timesteps: int = 1000) -> torch.Tensor:
    """Linear schedule proposed in original ddpm paper

    Args:
        timesteps: Total number of timesteps

    Returns:
        A tensor of the beta schedul (timesteps,)
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
