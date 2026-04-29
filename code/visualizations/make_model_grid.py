#!/usr/bin/env python3
"""Build a side-by-side comparison grid from model sample .npz files.

Each input model contributes one row with up to 9 images. The row label is
rendered on the left so teacher/student comparisons are easy to read.

Expected .npz format:
  - `all_samples`: aggregated samples saved by the eval pipeline
  - or `samples`: a single shard of samples

Example:
  python visualizations/make_model_grid.py \
    --input Teacher=/path/to/teacher/ckpt_1_samples.npz \
    --input Student=/path/to/student/ckpt_25_samples.npz \
    --output visualizations/teacher_vs_student.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_samples(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        if "all_samples" in data:
            samples = data["all_samples"]
        elif "samples" in data:
            samples = data["samples"]
        elif "arr_0" in data:
            samples = data["arr_0"]
        else:
            raise KeyError(
                f"No `all_samples`, `samples`, or `arr_0` key found in {npz_path}; "
                f"available keys: {list(data.keys())}"
            )

    if samples.ndim != 4:
        raise ValueError(f"Expected samples with shape [N, H, W, C], got {samples.shape}")
    return samples


def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def to_pil(img: np.ndarray) -> Image.Image:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def make_row_canvas(
    label: str,
    samples: np.ndarray,
    n_cols: int = 9,
    cell_size: int = 96,
    label_width: int = 260,
    padding: int = 12,
    font_path: str | None = None,
    font_size: int = 28,
) -> Image.Image:
    n_show = min(len(samples), n_cols)
    row_w = label_width + padding * 2 + n_cols * (cell_size + padding)
    row_h = padding * 2 + cell_size
    canvas = Image.new("RGB", (row_w, row_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_path, font_size)

    # Label area
    text_x = 18
    text_y = row_h // 2 - font_size // 2 - 2
    draw.text((text_x, text_y), label, fill="black", font=font)
    draw.line((label_width, 0, label_width, row_h), fill=(220, 220, 220), width=2)

    # Images
    for i in range(n_cols):
        x0 = label_width + padding + i * (cell_size + padding)
        y0 = padding
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        draw.rectangle((x0, y0, x1, y1), outline=(200, 200, 200), width=1)

        if i < n_show:
            img = to_pil(samples[i]).convert("RGB")
            img = img.resize((cell_size, cell_size), Image.BICUBIC)
            canvas.paste(img, (x0, y0))
    return canvas


def stack_rows(rows: List[Image.Image], margin: int = 20) -> Image.Image:
    if not rows:
        raise ValueError("No rows to stack")
    width = max(img.width for img in rows)
    height = sum(img.height for img in rows) + margin * (len(rows) + 1)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    y = margin
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + margin
    return canvas


def parse_input(s: str) -> Tuple[str, Path]:
    if "=" not in s:
        raise ValueError(f"Input must be in the form Label=/path/to/file.npz, got: {s}")
    label, path = s.split("=", 1)
    label = label.strip()
    path = Path(path.strip()).expanduser()
    if not label:
        raise ValueError(f"Empty label in input: {s}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Model input in the form Label=/path/to/samples.npz. Repeatable.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path, e.g. visualizations/comparison.png",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Number of images per row. Default: 9.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=96,
        help="Size of each thumbnail cell in pixels.",
    )
    parser.add_argument(
        "--label-font-size",
        type=int,
        default=28,
        help="Font size for row labels.",
    )
    parser.add_argument(
        "--label-font",
        default=None,
        help="Optional path to a .ttf font file for row labels.",
    )
    args = parser.parse_args()

    rows: List[Image.Image] = []
    for item in args.input:
        label, path = parse_input(item)
        samples = load_samples(path)
        rows.append(
            make_row_canvas(
                label=label,
                samples=samples,
                n_cols=args.cols,
                cell_size=args.cell_size,
                font_path=args.label_font,
                font_size=args.label_font_size,
            )
        )

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = stack_rows(rows)
    grid.save(out_path)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
