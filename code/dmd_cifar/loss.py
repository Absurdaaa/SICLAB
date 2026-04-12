from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ddpm_cifar.diffusion import extract


@torch.no_grad()
def predict_x0_from_teacher(diffusion, x0_candidate: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x0_candidate)
    x_t, _ = diffusion.q_sample(x0_candidate, timesteps, noise=noise)
    eps_pred = diffusion.model(x_t, timesteps)
    sqrt_alpha = extract(diffusion.sqrt_alphas_cumprod, timesteps, x_t.shape)
    sqrt_one_minus = extract(diffusion.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
    x0_pred = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha
    return x0_pred.clamp(-1.0, 1.0)


def distillation_loss(
    diffusion,
    student,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_reg: float,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    x_student = student(noise)
    x_teacher = predict_x0_from_teacher(diffusion, x_student, timesteps)
    match_loss = F.mse_loss(x_student, x_teacher)
    range_loss = torch.mean(torch.relu(torch.abs(x_student) - 1.0) ** 2)
    loss = match_loss + alpha_reg * range_loss
    stats = {
        "match_loss": float(match_loss.detach()),
        "range_loss": float(range_loss.detach()),
    }
    return loss, stats, x_student

__all__ = ["distillation_loss", "predict_x0_from_teacher"]
