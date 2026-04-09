from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CIFAR10BatchDataset(Dataset):
    """Load CIFAR-10 from the official python batch files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images = self._load_batches()

    def _load_batches(self) -> torch.Tensor:
        batch_files = sorted(self.root.glob("data_batch_*"))
        if not batch_files:
            raise FileNotFoundError(f"No CIFAR batch files found under {self.root}")
        
        # 读取每个batch
        arrays = []
        for path in batch_files:
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            arrays.append(batch[b"data"])

        data = np.concatenate(arrays, axis=0)
        # 数据预处理：将像素值从[0, 255]缩放到[0, 1]
        data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        # 将像素值从[0, 1]缩放到[-1, 1]
        data = data * 2.0 - 1.0
        return torch.from_numpy(data)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]
