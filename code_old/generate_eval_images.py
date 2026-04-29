from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
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
        help="Per-process sampling batch size. In single-process mode this is the normal batch size.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_runtime(device_arg: str) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process generation requested but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, world_size

    return get_device(device_arg), 0, 1


def cleanup_runtime() -> None:
    if is_distributed():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    device, rank, world_size = setup_runtime(args.device)
    diffusion, config = load_diffusion(args.checkpoint, device)
    output_dir = ensure_dir(args.output_dir)

    assigned_indices = list(range(rank, args.num_images, world_size))
    total_steps = (len(assigned_indices) + args.batch_size - 1) // args.batch_size
    progress = tqdm(
        range(total_steps),
        desc=f"Generate eval images rank {rank}",
        disable=(rank != 0),
    )

    for step in progress:
        start = step * args.batch_size
        end = min(start + args.batch_size, len(assigned_indices))
        current_indices = assigned_indices[start:end]
        if not current_indices:
            continue

        samples = diffusion.sample(
            num_samples=len(current_indices),
            image_size=config.image_size,
            channels=config.in_channels,
            device=device,
        )
        for sample, image_index in zip(samples, current_indices):
            save_single_image(sample, output_dir / f"{image_index:05d}.png")

    if is_distributed():
        dist.barrier()

    if rank == 0:
        print(f"generated: {args.num_images}")
        print(f"output_dir: {output_dir}")

    cleanup_runtime()


if __name__ == "__main__":
    main()
