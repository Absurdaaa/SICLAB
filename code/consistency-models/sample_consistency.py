from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import DistillConfig
from distill_train import sample_consistency
from modeling import ConsistencyModel, build_ddim_scheduler, build_unet
from utils import ensure_dir, get_device, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a distilled consistency model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs_consistency" / "sample.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = DistillConfig(**checkpoint["config"])
    unet = build_unet(
        config.image_size,
        config.in_channels,
        config.base_channels,
        config.channel_multipliers,
        config.layers_per_block,
        config.attention_levels,
    ).to(device)
    model = ConsistencyModel(unet, sigma_data=config.sigma_data).to(device)
    model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    model.eval()
    scheduler = build_ddim_scheduler(config.num_train_timesteps, config.beta_schedule)
    samples = sample_consistency(
        model,
        scheduler,
        args.num_samples,
        config.image_size,
        config.in_channels,
        device,
        steps=args.steps,
    )
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    save_image_grid(samples, output_path)
    print(f"Saved consistency samples to {output_path}")


if __name__ == "__main__":
    main()

