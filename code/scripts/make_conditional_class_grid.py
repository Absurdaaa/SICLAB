#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def list_class_dirs(samples_root: Path) -> Dict[int, Path]:
    class_dirs: Dict[int, Path] = {}
    for class_id in range(10):
        path = samples_root / f"class_{class_id}"
        if path.is_dir():
            class_dirs[class_id] = path
    if len(class_dirs) != 10:
        missing = [str(i) for i in range(10) if i not in class_dirs]
        raise ValueError(f"Missing class directories under {samples_root}: {missing}")
    return class_dirs


def _load_npz_array(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "samples" in data:
            arr = data["samples"]
        else:
            arr = data[data.files[0]]
    return arr


def load_generated_samples(class_dir: Path) -> np.ndarray:
    arrays: List[np.ndarray] = []
    for root, _, files in os.walk(class_dir):
        for name in sorted(files):
            if name.endswith(".npz"):
                arrays.append(_load_npz_array(Path(root) / name))
    if not arrays:
        raise ValueError(f"No npz sample files found under {class_dir}")
    samples = np.concatenate(arrays, axis=0)
    if samples.dtype != np.uint8:
        samples = np.clip(samples, 0, 255).astype(np.uint8)
    return samples


def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def sample_per_class(
    class_dirs: Dict[int, Path],
    rows: int,
    seed: int,
    strategy: str,
) -> Dict[int, np.ndarray]:
    rng = random.Random(seed)
    selected: Dict[int, np.ndarray] = {}
    for class_id, class_dir in sorted(class_dirs.items()):
        samples = load_generated_samples(class_dir)
        if samples.shape[0] < rows:
            raise ValueError(
                f"class_{class_id} only has {samples.shape[0]} samples, need at least {rows}"
            )
        if strategy == "first":
            indices = list(range(rows))
        elif strategy == "random":
            indices = rng.sample(range(samples.shape[0]), rows)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        selected[class_id] = samples[indices]
    return selected


def build_grid(
    selected: Dict[int, np.ndarray],
    output_path: Path,
    cell_size: int,
    padding: int,
    header_height: int,
    font_size: int,
    font_path: str | None,
    title: str,
) -> None:
    rows = next(iter(selected.values())).shape[0]
    cols = 10

    width = padding + cols * (cell_size + padding)
    height = header_height + padding + rows * (cell_size + padding)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_path, font_size)
    title_font = load_font(font_path, font_size + 4)

    if title:
        draw.text((padding, 8), title, fill="black", font=title_font)

    for col in range(cols):
        x0 = padding + col * (cell_size + padding)
        label = CIFAR10_CLASSES[col]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x0 + (cell_size - text_w) // 2
        text_y = header_height - font_size - 8
        draw.text((text_x, text_y), label, fill="black", font=font)

        for row in range(rows):
            y0 = header_height + padding + row * (cell_size + padding)
            tile = Image.fromarray(selected[col][row]).convert("RGB")
            tile = tile.resize((cell_size, cell_size), Image.BICUBIC)
            canvas.paste(tile, (x0, y0))
            draw.rectangle(
                (x0, y0, x0 + cell_size, y0 + cell_size),
                outline=(210, 210, 210),
                width=1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved grid to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cell-size", type=int, default=96)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--header-height", type=int, default=56)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=["first", "random"], default="first")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    samples_root = Path(args.samples_root).expanduser()
    output_path = Path(args.output).expanduser()
    class_dirs = list_class_dirs(samples_root)
    selected = sample_per_class(class_dirs, args.rows, args.seed, args.strategy)
    build_grid(
        selected=selected,
        output_path=output_path,
        cell_size=args.cell_size,
        padding=args.padding,
        header_height=args.header_height,
        font_size=args.font_size,
        font_path=args.font_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()
