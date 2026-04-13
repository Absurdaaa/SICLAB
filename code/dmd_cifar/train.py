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
from dmd_cifar.loss import denoising_loss, generator_loss
from dmd_cifar.student import OneStepGenerator
from ddpm_cifar.dataset import CIFAR10BatchDataset
from ddpm_cifar.diffusion import GaussianDiffusion
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


def save_full_checkpoint(
    output_dir: Path,
    epoch: int,
    student,
    ema_student,
    mu_fake_diffusion,
    optimizer_g,
    optimizer_d,
    config: DistillConfig,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "student": unwrap_model(student).state_dict(),
        "ema_student": unwrap_model(ema_student).state_dict(),
        "mu_fake_model": unwrap_model(mu_fake_diffusion.model).state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "config": config.to_dict(),
    }
    torch.save(checkpoint, output_dir / f"distill_full_epoch_{epoch:04d}.pt")


def get_generator_sigma(
    diffusion,
    timestep: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma_max: float,
) -> torch.Tensor:
    step = max(0, min(timestep, diffusion.timesteps - 1))
    sqrt_alpha = diffusion.sqrt_alphas_cumprod[step].to(device=device, dtype=dtype)
    sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[step].to(device=device, dtype=dtype)
    sigma = (sqrt_one_minus / sqrt_alpha.clamp_min(1e-6)).clamp(1e-4, sigma_max)
    return torch.full((batch_size,), float(sigma), device=device, dtype=dtype)


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
    mu_fake_diffusion = GaussianDiffusion(
        copy.deepcopy(teacher_diffusion.model),
        teacher_cfg.timesteps,
        teacher_cfg.beta_start,
        teacher_cfg.beta_end,
    ).to(device)
    mu_fake_diffusion.model.requires_grad_(True)

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
        student = DDP(
            student,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
        mu_fake_diffusion.model = DDP(
            mu_fake_diffusion.model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
        if is_main_process():
            print(f"Using DDP on {world_size} GPUs for distillation")

    optimizer_g = AdamW(unwrap_model(student).parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_d = AdamW(unwrap_model(mu_fake_diffusion.model).parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler_g = make_grad_scaler(enabled=(config.use_amp and device.type == "cuda"))
    scaler_d = make_grad_scaler(enabled=(config.use_amp and device.type == "cuda"))

    for epoch in range(1, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        unwrap_model(student).train()
        unwrap_model(mu_fake_diffusion.model).train()
        running_gen_loss = 0.0
        running_denoise_loss = 0.0
        progress = tqdm(dataloader, desc=f"Distill {epoch}/{config.epochs}", disable=not is_main_process(), leave=False)

        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            generator_sigma = get_generator_sigma(
                teacher_diffusion,
                config.generator_sigma_timestep,
                batch.shape[0],
                device,
                batch.dtype,
                config.generator_sigma_max,
            )
            z = torch.randn_like(batch) * generator_sigma[:, None, None, None]
            if config.reg_weight > 0.0 and config.teacher_pair_batch_size > 0:
                z_ref_sigma = get_generator_sigma(
                    teacher_diffusion,
                    config.generator_sigma_timestep,
                    config.teacher_pair_batch_size,
                    device,
                    batch.dtype,
                    config.generator_sigma_max,
                )
                z_ref = (
                    torch.randn(
                        config.teacher_pair_batch_size,
                        config.in_channels,
                        config.image_size,
                        config.image_size,
                        device=device,
                        dtype=batch.dtype,
                    )
                    * z_ref_sigma[:, None, None, None]
                )
            else:
                z_ref = torch.empty(0, config.in_channels, config.image_size, config.image_size, device=device, dtype=batch.dtype)
            timesteps = torch.randint(
                config.min_timestep,
                config.max_timestep + 1,
                (z.shape[0],),
                device=device,
                dtype=torch.long,
            )

            optimizer_g.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=(config.use_amp and device.type == "cuda")):
                gen_loss, gen_stats, x_for_denoiser = generator_loss(
                    teacher_diffusion,
                    student,
                    teacher_diffusion.model,
                    mu_fake_diffusion.model,
                    z,
                    z_ref,
                    generator_sigma,
                    timesteps,
                    reg_weight=config.reg_weight,
                )
            if not torch.isfinite(gen_loss):
                raise RuntimeError(
                    "Non-finite generator loss detected. "
                    f"sigma={generator_sigma[0].item():.4f}, "
                    f"timestep_range=[{int(timesteps.min())}, {int(timesteps.max())}]"
                )
            scaler_g.scale(gen_loss).backward()
            scaler_g.unscale_(optimizer_g)
            clip_grad_norm_(unwrap_model(student).parameters(), config.grad_clip)
            scaler_g.step(optimizer_g)
            scaler_g.update()
            update_ema(ema_student, student, config.ema_decay)

            optimizer_d.zero_grad(set_to_none=True)
            x_for_denoiser = x_for_denoiser.detach()
            denoise_timesteps = torch.randint(1, config.max_timestep + 1, (x_for_denoiser.shape[0],), device=device, dtype=torch.long)
            with autocast_context(device, enabled=(config.use_amp and device.type == "cuda")):
                diff_loss, diff_stats = denoising_loss(
                    teacher_diffusion,
                    mu_fake_diffusion.model,
                    x_for_denoiser,
                    denoise_timesteps,
                )
            if not torch.isfinite(diff_loss):
                raise RuntimeError(
                    "Non-finite denoising loss detected. "
                    f"sigma={generator_sigma[0].item():.4f}, "
                    f"timestep_range=[{int(denoise_timesteps.min())}, {int(denoise_timesteps.max())}]"
                )
            scaler_d.scale(diff_loss).backward()
            scaler_d.unscale_(optimizer_d)
            clip_grad_norm_(unwrap_model(mu_fake_diffusion.model).parameters(), config.grad_clip)
            scaler_d.step(optimizer_d)
            scaler_d.update()

            running_gen_loss += gen_loss.item()
            running_denoise_loss += diff_loss.item()
            if is_main_process():
                progress.set_postfix(
                    gen=f"{gen_loss.item():.4f}",
                    dm=f"{gen_stats['dm_loss']:.4f}",
                    reg=f"{gen_stats['reg_loss']:.4f}",
                    d=f"{diff_loss.item():.4f}",
                )

        epoch_gen_loss = torch.tensor(running_gen_loss / len(dataloader), device=device)
        epoch_denoise_loss = torch.tensor(running_denoise_loss / len(dataloader), device=device)
        if is_distributed():
            dist.all_reduce(epoch_gen_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_denoise_loss, op=dist.ReduceOp.SUM)
            epoch_gen_loss /= world_size
            epoch_denoise_loss /= world_size
        if is_main_process():
            print(
                f"epoch={epoch} gen_loss={epoch_gen_loss.item():.6f} "
                f"denoise_loss={epoch_denoise_loss.item():.6f}"
            )

        if is_main_process() and epoch % config.sample_every == 0:
            ema_student.eval()
            sample_sigma = get_generator_sigma(
                teacher_diffusion,
                config.generator_sigma_timestep,
                config.num_sample_images,
                device,
                torch.float32,
                config.generator_sigma_max,
            )
            samples = ema_student(
                torch.randn(
                    config.num_sample_images,
                    config.in_channels,
                    config.image_size,
                    config.image_size,
                    device=device,
                ) * sample_sigma[:, None, None, None],
                sample_sigma,
            )
            save_image_grid(samples, output_dir / "samples" / f"epoch_{epoch:04d}.png")

        if is_main_process() and epoch % config.save_every == 0:
            save_checkpoint(output_dir / "checkpoints", epoch, student, ema_student, optimizer_g, config)
            save_full_checkpoint(
                output_dir / "checkpoints",
                epoch,
                student,
                ema_student,
                mu_fake_diffusion,
                optimizer_g,
                optimizer_d,
                config,
            )

    cleanup_runtime()


def main() -> None:
    args = parse_args()
    config = DistillConfig.from_json(args.config)
    train(config)


if __name__ == "__main__":
    main()
