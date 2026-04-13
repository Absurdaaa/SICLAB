from __future__ import annotations

import torch
from torch import nn


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_t = (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )
        return x_t, noise

    def p_losses(self, x_start: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x_t, noise = self.q_sample(x_start, timesteps)
        noise_pred = self.model(x_t, timesteps)
        return torch.mean((noise - noise_pred) ** 2)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        betas_t = extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, timesteps, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, timesteps) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = extract(self.posterior_variance, timesteps, x.shape)

        nonzero_mask = (timesteps != 0).float().reshape(x.shape[0], *((1,) * (x.ndim - 1)))
        noise = torch.randn_like(x)
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, num_samples: int, image_size: int, device: torch.device, channels: int = 3) -> torch.Tensor:
        x = torch.randn(num_samples, channels, image_size, image_size, device=device)
        return self.sample_from_noise(x)

    @torch.no_grad()
    def sample_from_noise(self, x: torch.Tensor) -> torch.Tensor:
        for timestep in reversed(range(self.timesteps)):
            t = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x
