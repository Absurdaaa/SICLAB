from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import json


@dataclass
class DistillConfig:
    teacher_checkpoint: str = "./outputs/checkpoints/checkpoint_epoch_0300.pt"
    output_dir: str = "./distill_outputs"
    image_size: int = 32
    in_channels: int = 3
    student_base_channels: int = 128
    student_channel_multipliers: tuple[int, ...] = (2, 2, 2)
    student_num_res_blocks: int = 2
    student_dropout: float = 0.13
    student_attention_levels: tuple[int, ...] = ()
    timesteps: int = 1000
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    reg_weight: float = 0.25
    min_timestep: int = 10
    max_timestep: int = 999
    generator_sigma_timestep: int = 999
    teacher_pair_batch_size: int = 16
    save_every: int = 5
    sample_every: int = 5
    num_sample_images: int = 16
    seed: int = 42
    device: str = "auto"
    use_ddp: bool = True
    use_amp: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "DistillConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

__all__ = ["DistillConfig"]
