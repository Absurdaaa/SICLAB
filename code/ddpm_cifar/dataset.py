from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CIFAR10BatchDataset(Dataset):
    """Load CIFAR-10 from the official python batch files."""

    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.images = self._load_batches()

    def _load_batches(self) -> torch.Tensor:
        if self.split == "train":
            batch_files = sorted(self.root.glob("data_batch_*"))
        elif self.split == "test":
            batch_files = [self.root / "test_batch"]
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        missing = [path for path in batch_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing CIFAR batch files: {missing}")

        arrays = []
        for path in batch_files:
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            arrays.append(batch[b"data"])

        data = np.concatenate(arrays, axis=0)
        data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        data = data * 2.0 - 1.0
        return torch.from_numpy(data)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]
