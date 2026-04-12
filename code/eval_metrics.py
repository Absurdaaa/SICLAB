from __future__ import annotations

import argparse
from pathlib import Path

from torch_fidelity import calculate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID and Inception Score for generated images.")
    parser.add_argument("--real-dir", type=str, required=True, help="Directory of real images.")
    parser.add_argument("--generated-dir", type=str, required=True, help="Directory of generated images.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for metric computation.")
    parser.add_argument(
        "--no-isc",
        action="store_true",
        help="Disable Inception Score and only compute FID.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_dir = Path(args.real_dir)
    generated_dir = Path(args.generated_dir)

    if not real_dir.is_dir():
        raise FileNotFoundError(f"Real image directory not found: {real_dir}")
    if not generated_dir.is_dir():
        raise FileNotFoundError(f"Generated image directory not found: {generated_dir}")

    metrics = calculate_metrics(
        input1=str(generated_dir),
        input2=str(real_dir),
        cuda=args.device.startswith("cuda"),
        isc=not args.no_isc,
        fid=True,
        batch_size=args.batch_size,
        device=args.device,
        samples_find_deep=True,
    )

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
