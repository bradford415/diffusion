import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class GaussianDiffusion(nn.Module):
    """DDPM model which is responsible for noising images and denoising with unet 
    
    q(x_{1:T} | x_{0}) is the forward diffusion process which adds noise according to a variance schedule β1, . . . , βT
    p_θ(x_{0:T}) is the reverse process 
    
    """
    def __init__(
        self,
        model,
        *,
        image_size: Tuple,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_v",
        beta_schedule="sigmoid",
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.0,
        auto_normalize=True,
        offset_noise_strength=0.0,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5,
    ):
        """
        Args:
            model: A torch Unet model; theoretically this could be any encoder-decoder
                   model which downsamples and upsamples back to the original input size
            image_size: TODO: verify it is (h, w) tuple of orignal image size in the form of (height, width)

        """
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert (
            not hasattr(model, "random_or_learned_sinusoidal_cond")
            or not model.random_or_learned_sinusoidal_cond
        )

        self.model = model

        # Input image channels (e.g., rgb = 3)
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")


        # Below we are essenttially defining the necessary alpha and beta variables of the 
        # same shape so that we can sample all of the required variables for the forward 
        # process at a specific time step
        
        # Define the beta schedule; minumum and maximum noise to add
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        # Define alphas for the forward process; 
        # these are defined in the ddpm paper in equation 4 and the paragraph above
        # α¯_t = 1 - β and α¯_t = cumulative product of α at timestep t
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Define α_{t-1}
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Number of timesteps for diffusion; typically T = 1000
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # TODO: Understand and comment sampling_timesteps;
        self.sampling_timesteps = (
            sampling_timesteps if sampling_timesteps is not None else timesteps
        )

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        # Store betas and alphas as part of the model
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Compute calculations for diffusion q(x_{1:T} | x_0) and store in the model; 
        # equation 2 in ddpm paper
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        
        ################ START HERE, Understand the reset of these params ##########
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )


        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

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

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False
    ):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        return sample_fn(
            (batch_size, channels, h, w), return_all_timesteps=return_all_timesteps
        )

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(
            rearrange(t, "b ... -> b (...)") for t in (x_start, noise)
        )
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    def q_sample(self, x_start, t, noise):
        """Extract 

        Args:
            noise: Random gaussian noise (b, c, h, w)
        """

        return ( ######### START HERE, understand what alphas really mean
            extract_values(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_values(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_mean_variance(self, x_0, x_t, t):
        # TODO: Comment this once I understand it better
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def p_losses(self, x_start, timestep, noise=None):
        ############# START HERE ################################
        """Main method that adds noise to the image and denoises it through the unet model
        
        Args:
            x_start: 
            timestep:
            noise: TODO: remove this if it's not used anywhere
        """
        b, c, h, w = x_start.shape

        # Sample random noise from a normal distribution (b, c, h, w) 
        if noise is None:
            noise = torch.randn_like(x_start)

        # noise sample
        x = self.q_sample(x_start=x_start, t=timestep, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, timestep).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, timestep, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, timestep, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, timestep, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """Forward pass of DDPM; the main diffusion logic is perform in self.p_losses()
        Args:
        """
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *img.shape,
            img.device,
            self.image_size,
        )
        assert (
            h == img_size[0] and w == img_size[1]
        ), f"height and width of image must be {img_size}"

        # Batch of random (uniformly sampled) timesteps [0, num_timesteps); shape (b,)
        timestep = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Normalization is done in the dataset class so I don't think we need this
        #img = self.normalize(img)
        return self.p_losses(img, timestep, *args, **kwargs)
    
    
def extract_values(values, timestep, x_shape):
    """Extract coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    
    Args:
        values: A tensor of values to extract from
        timestep: 

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
