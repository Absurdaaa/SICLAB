from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast
    HAS_TORCH_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    HAS_TORCH_AMP = False

from config import TeacherConfig
from data import CIFAR10BatchDataset
from modeling import build_ddpm_scheduler, build_unet, clone_model
from utils import ensure_dir, get_device, save_image_grid, set_seed, unwrap_model, update_ema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM teacher with diffusers.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "teacher_config.json"),
    )
    return parser.parse_args()


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


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_runtime(config: TeacherConfig) -> tuple[torch.device, int]:
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


@torch.no_grad()
def sample_teacher(unet, scheduler, num_samples: int, image_size: int, channels: int, device: torch.device) -> torch.Tensor:
    unet = unwrap_model(unet)
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)
    for step in reversed(range(scheduler.config.num_train_timesteps)):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        model_output = unet(x, t).sample
        x = scheduler.step(model_output, step, x).prev_sample
    return x.clamp(-1.0, 1.0)


def train(config: TeacherConfig) -> None:
    device, world_size = setup_runtime(config)
    set_seed(config.seed + get_rank())

    output_dir = ensure_dir(Path(__file__).resolve().parent / config.output_dir)
    if is_main_process():
        ensure_dir(output_dir / "samples")
        ensure_dir(output_dir / "checkpoints")
        with open(output_dir / "teacher_config.json", "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
    if is_distributed():
        dist.barrier()

    dataset = CIFAR10BatchDataset(Path(__file__).resolve().parent / config.data_dir, split="train")
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

    unet = build_unet(
        config.image_size,
        config.in_channels,
        config.base_channels,
        config.channel_multipliers,
        config.layers_per_block,
        config.attention_levels,
    ).to(device)
    ema_unet = clone_model(unet).eval().to(device)
    ema_unet.requires_grad_(False)
    if world_size > 1:
        unet = DDP(unet, device_ids=[device.index], output_device=device.index)

    scheduler = build_ddpm_scheduler(config.num_train_timesteps, config.beta_schedule)
    optimizer = AdamW(unwrap_model(unet).parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = make_grad_scaler(enabled=(config.use_amp and device.type == "cuda"))

    for epoch in range(1, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        unwrap_model(unet).train()
        running_loss = 0.0
        progress = tqdm(dataloader, desc=f"Teacher {epoch}/{config.epochs}", leave=False, disable=not is_main_process())
        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, config.num_train_timesteps, (batch.shape[0],), device=device, dtype=torch.long)
            noisy_images = scheduler.add_noise(batch, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=(config.use_amp and device.type == "cuda")):
                pred = unet(noisy_images, timesteps).sample
                loss = torch.mean((pred - noise) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_unet, unet, config.ema_decay)

            running_loss += loss.item()
            if is_main_process():
                progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = torch.tensor(running_loss / len(dataloader), device=device)
        if is_distributed():
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            epoch_loss /= world_size
        if is_main_process():
            print(f"epoch={epoch} loss={epoch_loss.item():.6f}")

        if is_main_process() and epoch % config.sample_every == 0:
            ema_unet.eval()
            samples = sample_teacher(
                ema_unet,
                scheduler,
                config.num_sample_images,
                config.image_size,
                config.in_channels,
                device,
            )
            save_image_grid(samples, output_dir / "samples" / f"epoch_{epoch:04d}.png")

        if is_main_process() and epoch % config.save_every == 0:
            checkpoint = {
                "epoch": epoch,
                "model": unwrap_model(unet).state_dict(),
                "ema_model": ema_unet.state_dict(),
                "config": config.to_dict(),
            }
            torch.save(checkpoint, output_dir / "checkpoints" / f"teacher_epoch_{epoch:04d}.pt")
    cleanup_runtime()


def main() -> None:
    args = parse_args()
    train(TeacherConfig.from_json(args.config))


if __name__ == "__main__":
    main()
