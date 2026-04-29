from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ddpm_cifar.dataset import CIFAR10BatchDataset
from ddpm_cifar.utils import ensure_dir, save_single_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CIFAR-10 images to a directory for FID/IS evaluation.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "cifar-10-batches-py"),
        help="Path to the CIFAR-10 python batch directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "real_test_images"),
        help="Directory to save exported images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of images to export. 0 means export all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CIFAR10BatchDataset(args.data_dir, split=args.split)
    output_dir = ensure_dir(args.output_dir)

    total = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    for index in tqdm(range(total), desc=f"Export {args.split} images"):
        save_single_image(dataset[index], output_dir / f"{index:05d}.png")

    print(f"split: {args.split}")
    print(f"exported: {total}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
