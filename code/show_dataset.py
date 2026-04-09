from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ddpm_cifar.dataset import CIFAR10BatchDataset
from ddpm_cifar.utils import ensure_dir, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize samples from the CIFAR-10 batch dataset.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "cifar-10-batches-py"),
        help="Path to the CIFAR-10 python batch directory.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=16,
        help="Number of images to save in the grid.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "dataset_preview.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CIFAR10BatchDataset(args.data_dir)

    num_images = min(args.num_images, len(dataset))
    images = torch.stack([dataset[i] for i in range(num_images)])

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    save_image_grid(images, output_path)

    print(f"dataset size: {len(dataset)}")
    print(f"image shape: {tuple(dataset[0].shape)}")
    print(f"value range: [{float(dataset[0].min()):.1f}, {float(dataset[0].max()):.1f}]")
    print(f"saved preview to: {output_path}")


if __name__ == "__main__":
    main()
