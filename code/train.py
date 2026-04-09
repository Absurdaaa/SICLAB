from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm_cifar.config import TrainConfig
from ddpm_cifar.dataset import CIFAR10BatchDataset
from ddpm_cifar.diffusion import GaussianDiffusion
from ddpm_cifar.model import UNet
from ddpm_cifar.utils import ensure_dir, get_device, save_image_grid, set_seed, update_ema


def build_diffusion(config: TrainConfig) -> tuple[GaussianDiffusion, GaussianDiffusion]:
    model = UNet(
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        channel_multipliers=config.channel_multipliers,
        num_res_blocks=config.num_res_blocks,
        time_emb_dim=config.time_emb_dim,
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
    checkpoint = {
        "epoch": epoch,
        "model": diffusion.model.state_dict(),
        "ema_model": ema_diffusion.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
    }
    torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch:04d}.pt")


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = get_device(config.device)
    output_dir = ensure_dir(config.output_dir)
    ensure_dir(output_dir / "samples")
    ensure_dir(output_dir / "checkpoints")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    dataset = CIFAR10BatchDataset(config.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    diffusion, ema_diffusion = build_diffusion(config)
    diffusion = diffusion.to(device)
    ema_diffusion = ema_diffusion.to(device)
    ema_diffusion.eval()

    optimizer = AdamW(
        diffusion.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(1, config.epochs + 1):
        diffusion.train()
        running_loss = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in progress:
            batch = batch.to(device)
            timesteps = torch.randint(0, config.timesteps, (batch.shape[0],), device=device)

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.p_losses(batch, timesteps)
            loss.backward()
            clip_grad_norm_(diffusion.model.parameters(), config.grad_clip)
            optimizer.step()
            update_ema(ema_diffusion.model, diffusion.model, config.ema_decay)

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"epoch={epoch} loss={epoch_loss:.6f}")

        if epoch % config.sample_every == 0:
            ema_diffusion.eval()
            samples = ema_diffusion.sample(
                num_samples=config.num_sample_images,
                image_size=config.image_size,
                channels=config.in_channels,
                device=device,
            )
            save_image_grid(samples, output_dir / "samples" / f"epoch_{epoch:04d}.png")

        if epoch % config.save_every == 0:
            save_checkpoint(output_dir / "checkpoints", epoch, diffusion, ema_diffusion, optimizer, config)


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
