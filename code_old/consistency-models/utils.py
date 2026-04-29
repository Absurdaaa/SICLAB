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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_model = unwrap_model(ema_model)
    model = unwrap_model(model)
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        for name, param in model.named_parameters():
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

        ema_buffers = dict(ema_model.named_buffers())
        for name, buffer in model.named_buffers():
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


def tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).to(torch.uint8)
    return image.permute(1, 2, 0).numpy()


def save_single_image(image: torch.Tensor, path: str | Path) -> None:
    Image.fromarray(tensor_to_uint8_image(image)).save(path)


__all__ = [
    "ensure_dir",
    "get_device",
    "save_image_grid",
    "save_single_image",
    "set_seed",
    "tensor_to_uint8_image",
    "unwrap_model",
    "update_ema",
]
