from __future__ import annotations

import copy

import torch
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel


def build_unet(
    image_size: int,
    in_channels: int,
    base_channels: int,
    channel_multipliers: tuple[int, ...],
    layers_per_block: int,
    attention_levels: tuple[int, ...],
) -> UNet2DModel:
    block_out_channels = tuple(base_channels * mult for mult in channel_multipliers)
    down_block_types = []
    up_block_types = []
    for level in range(len(block_out_channels)):
        if level in attention_levels:
            down_block_types.append("AttnDownBlock2D")
            up_block_types.append("AttnUpBlock2D")
        else:
            down_block_types.append("DownBlock2D")
            up_block_types.append("UpBlock2D")

    return UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=in_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(reversed(up_block_types)),
    )


class ConsistencyModel(torch.nn.Module):
    def __init__(self, unet: UNet2DModel, sigma_data: float = 0.5) -> None:
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, clip_output: bool = True) -> torch.Tensor:
        sigma = sigma.reshape(-1).to(x.device, x.dtype)
        c_skip = (self.sigma_data**2) / (sigma.pow(2) + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma.pow(2) + self.sigma_data**2)
        model_out = self.unet(x, sigma).sample
        out = c_skip[:, None, None, None] * x + c_out[:, None, None, None] * model_out
        return out.clamp(-1.0, 1.0) if clip_output else out


def build_ddpm_scheduler(num_train_timesteps: int, beta_schedule: str) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        clip_sample=False,
        prediction_type="epsilon",
    )


def build_ddim_scheduler(num_train_timesteps: int, beta_schedule: str) -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        clip_sample=False,
        prediction_type="epsilon",
    )


def sigma_from_scheduler(scheduler: DDPMScheduler | DDIMScheduler, timesteps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    alpha_bar = alphas_cumprod[timesteps.long()].clamp_min(1e-8)
    return torch.sqrt((1.0 - alpha_bar) / alpha_bar)


def ddim_step_between(
    scheduler: DDIMScheduler,
    teacher: UNet2DModel,
    x_t: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)
    alpha_t = alpha_bar[t.long()].reshape(-1, 1, 1, 1).clamp_min(1e-8)
    alpha_s = alpha_bar[s.long()].reshape(-1, 1, 1, 1).clamp_min(1e-8)
    eps_pred = teacher(x_t, t).sample
    pred_x0 = (x_t - torch.sqrt(1.0 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
    x_s = torch.sqrt(alpha_s) * pred_x0 + torch.sqrt(1.0 - alpha_s) * eps_pred
    return x_s.clamp(-1.0, 1.0)


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)


__all__ = [
    "ConsistencyModel",
    "build_ddim_scheduler",
    "build_ddpm_scheduler",
    "build_unet",
    "clone_model",
    "ddim_step_between",
    "sigma_from_scheduler",
]

