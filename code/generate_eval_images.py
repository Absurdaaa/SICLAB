from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from ddpm_cifar.utils import ensure_dir, get_device, save_single_image
from sample import load_diffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images to a directory for FID/IS evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "generated_eval_images"),
        help="Directory to save generated images.",
    )
    parser.add_argument("--num-images", type=int, default=10000, help="Number of images to generate.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Sampling batch size. Increase on larger GPUs.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    diffusion, config = load_diffusion(args.checkpoint, device)
    output_dir = ensure_dir(args.output_dir)

    image_index = 0
    total_steps = (args.num_images + args.batch_size - 1) // args.batch_size
    for _ in tqdm(range(total_steps), desc="Generate eval images"):
        current_batch = min(args.batch_size, args.num_images - image_index)
        if current_batch <= 0:
            break
        samples = diffusion.sample(
            num_samples=current_batch,
            image_size=config.image_size,
            channels=config.in_channels,
            device=device,
        )
        for sample in samples:
            save_single_image(sample, output_dir / f"{image_index:05d}.png")
            image_index += 1

    print(f"generated: {image_index}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
