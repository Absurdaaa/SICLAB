from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from config import DistillConfig, TeacherConfig
from distill_train import load_teacher, sample_consistency
from modeling import ConsistencyModel, build_ddim_scheduler, build_ddpm_scheduler, build_unet
from teacher_train import sample_teacher
from utils import ensure_dir, get_device, save_single_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation images for teacher or consistency model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--model-type", type=str, required=True, choices=["teacher", "consistency"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100, help="Per-process generation batch size.")
    parser.add_argument("--steps", type=int, default=1, help="Only used for consistency model.")
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


def load_consistency(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    return model, scheduler, config


def load_teacher_for_eval(checkpoint_path: str | Path, device: torch.device):
    model, config_dict = load_teacher(checkpoint_path, device)
    config = TeacherConfig(**config_dict)
    scheduler = build_ddpm_scheduler(config.num_train_timesteps, config.beta_schedule)
    return model, scheduler, config


def main() -> None:
    args = parse_args()
    device, rank, world_size = setup_runtime(args.device)
    output_dir = ensure_dir(args.output_dir)

    if args.model_type == "teacher":
        model, scheduler, config = load_teacher_for_eval(args.checkpoint, device)
    else:
        model, scheduler, config = load_consistency(args.checkpoint, device)

    assigned_indices = list(range(rank, args.num_images, world_size))
    total_steps = (len(assigned_indices) + args.batch_size - 1) // args.batch_size
    progress = tqdm(range(total_steps), desc=f"Generate eval rank {rank}", disable=(rank != 0))

    for step in progress:
        start = step * args.batch_size
        end = min(start + args.batch_size, len(assigned_indices))
        current_indices = assigned_indices[start:end]
        if not current_indices:
            continue

        if args.model_type == "teacher":
            samples = sample_teacher(
                model,
                scheduler,
                len(current_indices),
                config.image_size,
                config.in_channels,
                device,
            )
        else:
            samples = sample_consistency(
                model,
                scheduler,
                len(current_indices),
                config.image_size,
                config.in_channels,
                device,
                steps=args.steps,
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
