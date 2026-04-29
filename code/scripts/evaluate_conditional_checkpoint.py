#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_ckpt(workdir: Path, ckpt_arg: str) -> str:
    if ckpt_arg != "latest":
        return ckpt_arg

    ckpt_dir = workdir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"No checkpoint directory found: {ckpt_dir}")

    ckpts = []
    for path in ckpt_dir.iterdir():
        if path.is_file() and path.name.startswith("checkpoint_"):
            try:
                ckpts.append(int(path.name.split("checkpoint_")[1]))
            except ValueError:
                continue
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_* files found under {ckpt_dir}")
    return str(max(ckpts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--classifier-ckpt", required=True)
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--samples-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--mode", default="legacy_tensorflow")
    parser.add_argument("--stats-cache", default="")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    ckpt = resolve_ckpt(workdir, args.ckpt)

    samples_root = (
        Path(args.samples_root).expanduser().resolve()
        if args.samples_root
        else workdir / f"conditional_samples_ckpt_{ckpt}"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else workdir / f"conditional_metrics_ckpt_{ckpt}"
    )

    if not samples_root.is_dir():
        raise FileNotFoundError(f"Samples root not found: {samples_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    eval_script = PROJECT_ROOT / "scripts" / "evaluate_conditional_generation.py"
    cmd = [
        args.python_bin,
        str(eval_script),
        "--samples-root",
        str(samples_root),
        "--classifier-ckpt",
        args.classifier_ckpt,
        "--output-dir",
        str(output_dir),
        "--data-root",
        args.data_root,
        "--device",
        args.device,
        "--classifier-batch-size",
        str(args.classifier_batch_size),
        "--fid-batch-size",
        str(args.fid_batch_size),
        "--grid-size",
        str(args.grid_size),
        "--mode",
        args.mode,
    ]
    if args.stats_cache:
        cmd.extend(["--stats-cache", args.stats_cache])

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
