from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ddpm_cifar.diffusion import extract


@torch.no_grad()
def predict_x0(diffusion, model, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    eps_pred = model(x_t, timesteps)
    sqrt_alpha = extract(diffusion.sqrt_alphas_cumprod, timesteps, x_t.shape)
    sqrt_one_minus = extract(diffusion.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
    x0_pred = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha
    return x0_pred.clamp(-1.0, 1.0)


def distribution_matching_loss(
    diffusion,
    mu_real_model,
    mu_fake_model,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    noise = noise if noise is not None else torch.randn_like(x)
    x_t, _ = diffusion.q_sample(x, timesteps, noise=noise)

    with torch.no_grad():
        pred_fake = predict_x0(diffusion, mu_fake_model, x_t, timesteps)
        pred_real = predict_x0(diffusion, mu_real_model, x_t, timesteps)

    weighting_factor = torch.abs(x - pred_real).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
    grad = (pred_fake - pred_real) / weighting_factor
    target = (x - grad).detach()
    loss = 0.5 * F.mse_loss(x, target)
    return loss, {"dm_loss": float(loss.detach())}


def denoising_loss(
    diffusion,
    mu_fake_model,
    x: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float]]:
    x_t, _ = diffusion.q_sample(x.detach(), timesteps)
    pred_fake = predict_x0(diffusion, mu_fake_model, x_t, timesteps)
    weight = 1.0 / extract(diffusion.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape).pow(2).clamp_min(1e-5)
    loss = torch.mean(weight * (pred_fake - x.detach()) ** 2)
    return loss, {"denoise_loss": float(loss.detach())}


def generator_loss(
    diffusion,
    student,
    mu_real_model,
    mu_fake_model,
    z: torch.Tensor,
    z_ref: torch.Tensor,
    generator_sigma: torch.Tensor,
    timesteps: torch.Tensor,
    reg_weight: float,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    x = student(z, generator_sigma)
    dm_loss, dm_stats = distribution_matching_loss(diffusion, mu_real_model, mu_fake_model, x, timesteps)

    with torch.no_grad():
        y_ref = diffusion.sample_from_noise(z_ref).detach()
    x_ref = student(z_ref, generator_sigma[: z_ref.shape[0]])
    reg_loss = F.l1_loss(x_ref, y_ref)
    total = dm_loss + reg_weight * reg_loss
    stats = {
        **dm_stats,
        "reg_loss": float(reg_loss.detach()),
        "gen_loss": float(total.detach()),
    }
    return total, stats, x.detach()


__all__ = ["distribution_matching_loss", "denoising_loss", "generator_loss", "predict_x0"]
