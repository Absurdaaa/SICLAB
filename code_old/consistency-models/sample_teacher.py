from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import TeacherConfig
from modeling import build_ddpm_scheduler, build_unet
from teacher_train import sample_teacher
from utils import ensure_dir, get_device, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained diffusers teacher.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs_teacher" / "sample.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = TeacherConfig(**checkpoint["config"])
    model = build_unet(
        config.image_size,
        config.in_channels,
        config.base_channels,
        config.channel_multipliers,
        config.layers_per_block,
        config.attention_levels,
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    model.eval()
    scheduler = build_ddpm_scheduler(config.num_train_timesteps, config.beta_schedule)
    samples = sample_teacher(model, scheduler, args.num_samples, config.image_size, config.in_channels, device)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    save_image_grid(samples, output_path)
    print(f"Saved teacher samples to {output_path}")


if __name__ == "__main__":
    main()

