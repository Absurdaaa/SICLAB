from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ddpm_cifar.config import TrainConfig
from ddpm_cifar.diffusion import GaussianDiffusion
from ddpm_cifar.model import UNet
from ddpm_cifar.utils import ensure_dir, get_device, save_image_grid


def load_diffusion(checkpoint_path: str | Path, device: torch.device) -> tuple[GaussianDiffusion, TrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TrainConfig(**checkpoint["config"])
    model = UNet(
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        channel_multipliers=config.channel_multipliers,
        num_res_blocks=config.num_res_blocks,
        time_emb_dim=config.time_emb_dim,
    )
    diffusion = GaussianDiffusion(model, config.timesteps, config.beta_start, config.beta_end).to(device)
    state_dict = checkpoint.get("ema_model", checkpoint["model"])
    diffusion.model.load_state_dict(state_dict)
    diffusion.eval()
    return diffusion, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a trained DDPM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "sample.png"),
        help="Output image grid path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device(args.device)
    diffusion, config = load_diffusion(args.checkpoint, device)
    samples = diffusion.sample(
        num_samples=args.num_samples,
        image_size=config.image_size,
        channels=config.in_channels,
        device=device,
    )
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    save_image_grid(samples, output_path)
    print(f"Saved samples to {output_path}")
