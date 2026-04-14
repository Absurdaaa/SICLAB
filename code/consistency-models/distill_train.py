from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
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

from config import DistillConfig
from data import CIFAR10BatchDataset
from modeling import (
    ConsistencyModel,
    build_ddim_scheduler,
    build_unet,
    clone_model,
    ddim_step_between,
    sigma_from_scheduler,
)
from utils import ensure_dir, get_device, save_image_grid, set_seed, unwrap_model, update_ema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a diffusers teacher into a consistency model.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "consistency_config.json"),
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


def load_teacher(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    teacher_cfg = checkpoint["config"]
    unet = build_unet(
        teacher_cfg["image_size"],
        teacher_cfg["in_channels"],
        teacher_cfg["base_channels"],
        tuple(teacher_cfg["channel_multipliers"]),
        teacher_cfg["layers_per_block"],
        tuple(teacher_cfg["attention_levels"]),
    ).to(device)
    state_dict = checkpoint.get("ema_model", checkpoint["model"])
    unet.load_state_dict(state_dict)
    unet.eval()
    unet.requires_grad_(False)
    return unet, teacher_cfg


def get_bins(global_step: int, total_steps: int, bins_min: int, bins_max: int) -> int:
    progress = min(max(global_step / max(total_steps, 1), 0.0), 1.0)
    return int(math.ceil(bins_min + progress * (bins_max - bins_min)))


def make_timestep_pairs(batch_size: int, num_train_timesteps: int, bins: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.linspace(0, num_train_timesteps - 1, steps=bins, device=device).round().long()
    idx = torch.randint(0, bins - 1, (batch_size,), device=device)
    s = grid[idx]
    t = grid[idx + 1]
    return s, t


def sample_consistency(model: ConsistencyModel, scheduler, num_samples: int, image_size: int, channels: int, device: torch.device, steps: int) -> torch.Tensor:
    model = unwrap_model(model)
    if steps <= 1:
        grid = torch.tensor([scheduler.config.num_train_timesteps - 1], device=device, dtype=torch.long)
    else:
        grid = torch.linspace(0, scheduler.config.num_train_timesteps - 1, steps=steps, device=device).round().long()
        grid = torch.flip(grid, dims=[0])
    sigma_max = sigma_from_scheduler(scheduler, grid[:1]).item()
    x = torch.randn(num_samples, channels, image_size, image_size, device=device) * sigma_max
    for idx, t_value in enumerate(grid):
        t = torch.full((num_samples,), int(t_value.item()), device=device, dtype=torch.long)
        sigma = sigma_from_scheduler(scheduler, t)
        x = model(x, sigma)
        if idx < len(grid) - 1:
            next_t = torch.full((num_samples,), int(grid[idx + 1].item()), device=device, dtype=torch.long)
            next_sigma = sigma_from_scheduler(scheduler, next_t)
            x = x + next_sigma[:, None, None, None] * torch.randn_like(x)
    return x.clamp(-1.0, 1.0)


def train(config: DistillConfig) -> None:
    device, world_size = setup_runtime(config)
    set_seed(config.seed + get_rank())

    output_dir = ensure_dir(Path(__file__).resolve().parent / config.output_dir)
    if is_main_process():
        ensure_dir(output_dir / "samples")
        ensure_dir(output_dir / "checkpoints")
        with open(output_dir / "consistency_config.json", "w", encoding="utf-8") as f:
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

    teacher, teacher_cfg = load_teacher(Path(__file__).resolve().parent / config.teacher_checkpoint, device)
    ddim_scheduler = build_ddim_scheduler(config.num_train_timesteps, config.beta_schedule)

    student_unet = build_unet(
        config.image_size,
        config.in_channels,
        config.base_channels,
        config.channel_multipliers,
        config.layers_per_block,
        config.attention_levels,
    ).to(device)
    student = ConsistencyModel(student_unet, sigma_data=config.sigma_data).to(device)
    ema_student = clone_model(student).eval().to(device)
    ema_student.requires_grad_(False)
    if world_size > 1:
        student = DDP(student, device_ids=[device.index], output_device=device.index)

    optimizer = AdamW(unwrap_model(student).parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = make_grad_scaler(enabled=(config.use_amp and device.type == "cuda"))
    total_steps = config.epochs * len(dataloader)
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        unwrap_model(student).train()
        running_loss = 0.0
        progress = tqdm(dataloader, desc=f"Consistency {epoch}/{config.epochs}", leave=False, disable=not is_main_process())
        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            bins = get_bins(global_step, total_steps, config.bins_min, config.bins_max)
            s, t = make_timestep_pairs(batch.shape[0], config.num_train_timesteps, bins, device)
            noise = torch.randn_like(batch)

            alpha_bar = ddim_scheduler.alphas_cumprod.to(device=device, dtype=batch.dtype)
            alpha_t = alpha_bar[t].reshape(-1, 1, 1, 1)
            x_t = torch.sqrt(alpha_t) * batch + torch.sqrt(1.0 - alpha_t) * noise

            with torch.no_grad():
                x_s = ddim_step_between(ddim_scheduler, teacher, x_t, t, s)
                sigma_s = sigma_from_scheduler(ddim_scheduler, s)
                target = ema_student(x_s, sigma_s)

            sigma_t = sigma_from_scheduler(ddim_scheduler, t)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=(config.use_amp and device.type == "cuda")):
                pred = student(x_t, sigma_t)
                if config.loss_type == "l1":
                    loss = F.l1_loss(pred, target)
                else:
                    loss = F.mse_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_student, student, config.ema_decay)

            running_loss += loss.item()
            global_step += 1
            if is_main_process():
                progress.set_postfix(loss=f"{loss.item():.4f}", bins=bins)

        epoch_loss = torch.tensor(running_loss / len(dataloader), device=device)
        if is_distributed():
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            epoch_loss /= world_size
        if is_main_process():
            print(f"epoch={epoch} loss={epoch_loss.item():.6f}")

        if is_main_process() and epoch % config.sample_every == 0:
            ema_student.eval()
            samples = sample_consistency(
                ema_student,
                ddim_scheduler,
                config.num_sample_images,
                config.image_size,
                config.in_channels,
                device,
                steps=1,
            )
            save_image_grid(samples, output_dir / "samples" / f"epoch_{epoch:04d}.png")

        if is_main_process() and epoch % config.save_every == 0:
            checkpoint = {
                "epoch": epoch,
                "model": unwrap_model(student).state_dict(),
                "ema_model": ema_student.state_dict(),
                "config": config.to_dict(),
            }
            torch.save(checkpoint, output_dir / "checkpoints" / f"consistency_epoch_{epoch:04d}.pt")
    cleanup_runtime()


def main() -> None:
    args = parse_args()
    train(DistillConfig.from_json(args.config))


if __name__ == "__main__":
    main()
