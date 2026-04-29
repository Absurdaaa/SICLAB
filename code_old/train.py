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

from ddpm_cifar.config import TrainConfig
from ddpm_cifar.dataset import CIFAR10BatchDataset
from ddpm_cifar.diffusion import GaussianDiffusion
from ddpm_cifar.model import UNet
from ddpm_cifar.utils import (
    ensure_dir,
    get_device,
    save_image_grid,
    set_seed,
    unwrap_model,
    update_ema,
)


def build_diffusion(config: TrainConfig) -> tuple[GaussianDiffusion, GaussianDiffusion]:
    model = UNet(
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        channel_multipliers=config.channel_multipliers,
        num_res_blocks=config.num_res_blocks,
        time_emb_dim=config.time_emb_dim,
        dropout=config.dropout,
        attention_levels=config.attention_levels,
    )
    ema_model = copy.deepcopy(model)
    diffusion = GaussianDiffusion(model, config.timesteps, config.beta_start, config.beta_end)
    ema_diffusion = GaussianDiffusion(ema_model, config.timesteps, config.beta_start, config.beta_end)
    return diffusion, ema_diffusion


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    diffusion: GaussianDiffusion,
    ema_diffusion: GaussianDiffusion,
    optimizer: AdamW,
    config: TrainConfig,
) -> None:
    train_model = unwrap_model(diffusion.model)
    ema_model = unwrap_model(ema_diffusion.model)
    checkpoint = {
        "epoch": epoch,
        "model": train_model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
    }
    torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch:04d}.pt")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(config: TrainConfig) -> tuple[torch.device, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if config.use_ddp and world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP launch detected but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        return device, world_size

    return get_device(config.device), 1


def cleanup_distributed() -> None:
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


def train(config: TrainConfig) -> None:
    device, world_size = setup_distributed(config)
    set_seed(config.seed + get_rank())

    output_dir = Path(config.output_dir)
    if is_main_process():
        ensure_dir(output_dir)
        ensure_dir(output_dir / "samples")
        ensure_dir(output_dir / "checkpoints")
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
    if is_distributed():
        dist.barrier()

    dataset = CIFAR10BatchDataset(config.data_dir)
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

    diffusion, ema_diffusion = build_diffusion(config)
    diffusion = diffusion.to(device)
    ema_diffusion = ema_diffusion.to(device)

    if world_size > 1:
        diffusion.model = DDP(diffusion.model, device_ids=[device.index], output_device=device.index)
        if is_main_process():
            print(f"Using DDP on {world_size} GPUs")

    ema_diffusion.eval()

    optimizer = AdamW(
        unwrap_model(diffusion.model).parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = make_grad_scaler(enabled=(config.use_amp and device.type == "cuda"))

    for epoch in range(1, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        diffusion.train()
        running_loss = 0.0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{config.epochs}",
            leave=False,
            disable=not is_main_process(),
        )
        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            timesteps = torch.randint(0, config.timesteps, (batch.shape[0],), device=device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=(config.use_amp and device.type == "cuda")):
                loss = diffusion.p_losses(batch, timesteps)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(unwrap_model(diffusion.model).parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_diffusion.model, diffusion.model, config.ema_decay)

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
            ema_diffusion.eval()
            samples = ema_diffusion.sample(
                num_samples=config.num_sample_images,
                image_size=config.image_size,
                channels=config.in_channels,
                device=device,
            )
            save_image_grid(samples, output_dir / "samples" / f"epoch_{epoch:04d}.png")

        if is_main_process() and epoch % config.save_every == 0:
            save_checkpoint(output_dir / "checkpoints", epoch, diffusion, ema_diffusion, optimizer, config)

    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 DDPM model.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "train_config.json"),
        help="Path to JSON config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig.from_json(args.config)
    train(config)
