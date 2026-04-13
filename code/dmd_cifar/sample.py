from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dmd_cifar.config import DistillConfig
from dmd_cifar.student import OneStepGenerator
from ddpm_cifar.utils import ensure_dir, get_device, save_image_grid
from sample import load_diffusion


def load_student(checkpoint_path: str | Path, device: torch.device) -> tuple[OneStepGenerator, DistillConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = DistillConfig(**checkpoint["config"])
    model = OneStepGenerator(
        in_channels=config.in_channels,
        base_channels=config.student_base_channels,
        channel_multipliers=config.student_channel_multipliers,
        num_res_blocks=config.student_num_res_blocks,
        dropout=config.student_dropout,
        attention_levels=config.student_attention_levels,
    ).to(device)
    state_dict = checkpoint.get("ema_student", checkpoint["student"])
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a distilled one-step generator.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to distillation checkpoint.")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "distill_outputs" / "sample.png"),
        help="Output image grid path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, config = load_student(args.checkpoint, device)
    teacher_diffusion, _ = load_diffusion(config.teacher_checkpoint, device)
    step = max(0, min(config.generator_sigma_timestep, teacher_diffusion.timesteps - 1))
    sqrt_alpha = teacher_diffusion.sqrt_alphas_cumprod[step].to(device=device, dtype=torch.float32)
    sqrt_one_minus = teacher_diffusion.sqrt_one_minus_alphas_cumprod[step].to(device=device, dtype=torch.float32)
    sigma_value = (sqrt_one_minus / sqrt_alpha.clamp_min(1e-6)).clamp_min(1e-4)
    sigma = torch.full((args.num_samples,), float(sigma_value), device=device)
    noise = torch.randn(args.num_samples, config.in_channels, config.image_size, config.image_size, device=device) * sigma[:, None, None, None]
    samples = model(noise, sigma)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    save_image_grid(samples, output_path)
    print(f"Saved distilled samples to {output_path}")


if __name__ == "__main__":
    main()
