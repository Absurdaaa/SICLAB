from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import json


@dataclass
class TeacherConfig:
    data_dir: str = "../data/cifar-10-batches-py"
    output_dir: str = "./outputs_teacher"
    image_size: int = 32
    in_channels: int = 3
    base_channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 2)
    layers_per_block: int = 2
    attention_levels: tuple[int, ...] = (2, 3)
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"
    seed: int = 42
    device: str = "auto"
    use_ddp: bool = True
    use_amp: bool = True
    sample_every: int = 5
    save_every: int = 5
    num_sample_images: int = 16

    @classmethod
    def from_json(cls, path: str | Path) -> "TeacherConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DistillConfig:
    data_dir: str = "../data/cifar-10-batches-py"
    teacher_checkpoint: str = "./outputs_teacher/checkpoints/teacher_epoch_0100.pt"
    output_dir: str = "./outputs_consistency"
    image_size: int = 32
    in_channels: int = 3
    base_channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 2)
    layers_per_block: int = 2
    attention_levels: tuple[int, ...] = (2, 3)
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    ema_decay_target: float = 0.9999
    ema_decay_schedule: str = "paper"
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    bins_min: int = 20
    bins_max: int = 150
    bins_rho: float = 7.0
    bins_schedule: str = "paper"
    sigma_data: float = 0.5
    loss_type: str = "lpips_l1"
    lpips_net: str = "alex"
    lpips_weight: float = 1.0
    l1_weight: float = 1.0
    l2_weight: float = 1.0
    seed: int = 42
    device: str = "auto"
    use_ddp: bool = True
    use_amp: bool = True
    sample_every: int = 5
    save_every: int = 5
    num_sample_images: int = 16

    @classmethod
    def from_json(cls, path: str | Path) -> "DistillConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["TeacherConfig", "DistillConfig"]
