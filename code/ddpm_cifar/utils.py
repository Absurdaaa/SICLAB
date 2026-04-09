from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int | None = None) -> None:
    images = images.detach().cpu().clamp(-1.0, 1.0)
    images = ((images + 1.0) * 127.5).to(torch.uint8)
    b, c, h, w = images.shape
    nrow = nrow or int(math.sqrt(b))
    nrow = max(1, nrow)
    ncol = math.ceil(b / nrow)

    grid = torch.zeros(c, ncol * h, nrow * w, dtype=torch.uint8)
    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = image

    array = grid.permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)
