from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import json


@dataclass
class TrainConfig:
    data_dir: str = "../data/cifar-10-batches-py"
    output_dir: str = "./outputs"
    image_size: int = 32
    in_channels: int = 3
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    time_emb_dim: int = 256
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    save_every: int = 5
    sample_every: int = 5
    num_sample_images: int = 16
    seed: int = 42
    device: str = "auto"
    use_data_parallel: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
