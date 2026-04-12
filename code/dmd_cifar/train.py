from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast
    HAS_TORCH_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    HAS_TORCH_AMP = False

from dmd_cifar.config import DistillConfig
from dmd_cifar.loss import distillation_loss
from dmd_cifar.student import OneStepGenerator
from ddpm_cifar.dataset import CIFAR10BatchDataset
from ddpm_cifar.utils import ensure_dir, get_device, save_image_grid, set_seed, unwrap_model, update_ema
from sample import load_diffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a one-step distilled generator from a diffusion teacher.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "distill_config.json"),
        help="Path to JSON config file.",
    )
    return parser.parse_args()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_runtime(config: DistillConfig) -> tuple[torch.device, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if config.use_ddp and world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP launch detected but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), world_size
    return get_device(config.device), 1


def cleanup_runtime() -> None:
    if is_distributed():
        dist.destroy_process_group()


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return autocast(enabled=False)
    if HAS_TORCH_AMP:
        return autocast(device_type=device.type, enabled=True)
    return autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    if HAS_TORCH_AMP:
        return GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


def build_student(config: DistillConfig) -> OneStepGenerator:
    return OneStepGenerator(
        in_channels=config.in_channels,
        base_channels=config.student_base_channels,
        channel_multipliers=config.student_channel_multipliers,
        num_res_blocks=config.student_num_res_blocks,
        dropout=config.student_dropout,
        attention_levels=config.student_attention_levels,
    )


def save_checkpoint(output_dir: Path, epoch: int, student, ema_student, optimizer, config: DistillConfig) -> None:
    checkpoint = {
        "epoch": epoch,
        "student": unwrap_model(student).state_dict(),
        "ema_student": unwrap_model(ema_student).state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
    }
    torch.save(checkpoint, output_dir / f"distill_epoch_{epoch:04d}.pt")


def train(config: DistillConfig) -> None:
    device, world_size = setup_runtime(config)
    set_seed(config.seed + get_rank())

    output_dir = Path(config.output_dir)
    if is_main_process():
        ensure_dir(output_dir)
        ensure_dir(output_dir / "samples")
        ensure_dir(output_dir / "checkpoints")
        with open(output_dir / "distill_config.json", "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
    if is_distributed():
        dist.barrier()

    teacher_diffusion, teacher_cfg = load_diffusion(config.teacher_checkpoint, device)
    teacher_diffusion.eval()
    teacher_diffusion.requires_grad_(False)

    dataset = CIFAR10BatchDataset(teacher_cfg.data_dir, split="train")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=get_rank(), shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    student = build_student(config).to(device)
    ema_student = copy.deepcopy(student).eval().to(device)
    if world_size > 1:
        student = DDP(student, device_ids=[device.index], output_device=device.index)
        if is_main_process():
            print(f"Using DDP on {world_size} GPUs for distillation")

    optimizer = AdamW(unwrap_model(student).parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = make_grad_scaler(enabled=(config.use_amp and device.type == "cuda"))

    for epoch in range(1, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        unwrap_model(student).train()
        running_loss = 0.0
        progress = tqdm(dataloader, desc=f"Distill {epoch}/{config.epochs}", disable=not is_main_process(), leave=False)

        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(
                config.min_timestep,
                config.max_timestep + 1,
                (batch.shape[0],),
                device=device,
                dtype=torch.long,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=(config.use_amp and device.type == "cuda")):
                loss, stats, _ = distillation_loss(teacher_diffusion, student, noise, timesteps, alpha_reg=config.alpha_reg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(unwrap_model(student).parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_student, student, config.ema_decay)

            running_loss += loss.item()
            if is_main_process():
                progress.set_postfix(loss=f"{loss.item():.4f}", match=f"{stats['match_loss']:.4f}")

        epoch_loss = torch.tensor(running_loss / len(dataloader), device=device)
        if is_distributed():
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            epoch_loss /= world_size
        if is_main_process():
            print(f"epoch={epoch} distill_loss={epoch_loss.item():.6f}")

        if is_main_process() and epoch % config.sample_every == 0:
            ema_student.eval()
            samples = ema_student(
                torch.randn(
                    config.num_sample_images,
                    config.in_channels,
                    config.image_size,
                    config.image_size,
                    device=device,
                )
            )
            save_image_grid(samples, output_dir / "samples" / f"epoch_{epoch:04d}.png")

        if is_main_process() and epoch % config.save_every == 0:
            save_checkpoint(output_dir / "checkpoints", epoch, student, ema_student, optimizer, config)

    cleanup_runtime()


def main() -> None:
    args = parse_args()
    config = DistillConfig.from_json(args.config)
    train(config)


if __name__ == "__main__":
    main()
