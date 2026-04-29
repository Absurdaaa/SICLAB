#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.models as models
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from cleanfid import fid

from jcm.metrics import ResizeDataset


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


def build_classifier(num_classes=10):
    model = models.resnet18(num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def load_classifier(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_classifier().to(device)
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict)
    model.eval()
    return model


def classifier_transform():
    return Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def list_class_dirs(samples_root: str) -> Dict[int, str]:
    class_dirs = {}
    for class_id in range(10):
        path = os.path.join(samples_root, f"class_{class_id}")
        if os.path.isdir(path):
            class_dirs[class_id] = path
    if not class_dirs:
        raise ValueError(
            "No class directories found. Expected subdirectories like class_0 ... class_9"
        )
    return class_dirs


def _load_npz_array(path: str) -> np.ndarray:
    with np.load(path) as data:
        if "samples" in data:
            arr = data["samples"]
        else:
            arr = data[data.files[0]]
    return arr


def load_generated_samples(class_dir: str) -> np.ndarray:
    arrays: List[np.ndarray] = []
    for root, _, files in os.walk(class_dir):
        for name in sorted(files):
            if name.endswith(".npz"):
                arrays.append(_load_npz_array(os.path.join(root, name)))
    if not arrays:
        raise ValueError(f"No npz sample files found under {class_dir}")
    samples = np.concatenate(arrays, axis=0)
    if samples.dtype != np.uint8:
        samples = np.clip(samples, 0, 255).astype(np.uint8)
    return samples


def save_grid(samples: np.ndarray, output_path: str, grid_size: int) -> None:
    num_images = min(samples.shape[0], grid_size * grid_size)
    images = samples[:num_images]
    tile_h, tile_w = images.shape[1], images.shape[2]
    grid = np.zeros((grid_size * tile_h, grid_size * tile_w, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = image
    Image.fromarray(grid).save(output_path)


def classify_samples(
    model: torch.nn.Module,
    samples: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    transform = classifier_transform()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, samples.shape[0], batch_size):
            batch = samples[start : start + batch_size]
            tensor = torch.stack([transform(Image.fromarray(x)) for x in batch], dim=0)
            logits = model(tensor.to(device, non_blocking=True))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_feature_stats(
    samples: np.ndarray,
    feat_model,
    device: torch.device,
    batch_size: int,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = ResizeDataset(samples, mode=mode)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )
    features = []
    for batch in loader:
        features.append(fid.get_batch_features(batch, feat_model, device))
    features = np.concatenate(features, axis=0)
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def load_or_compute_real_stats(
    data_root: str,
    class_id: int,
    feat_model,
    device: torch.device,
    batch_size: int,
    mode: str,
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"class_{class_id}_stats.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["mu"], cached["sigma"]

    ds = CIFAR10(data_root, train=True, download=True)
    real_images = np.stack(
        [np.asarray(image) for image, label in ds if label == class_id], axis=0
    ).astype(np.uint8)
    mu, sigma = compute_feature_stats(real_images, feat_model, device, batch_size, mode)
    np.savez(cache_path, mu=mu, sigma=sigma)
    return mu, sigma


def plot_confusion(confusion: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--classifier-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--mode", default="legacy_tensorflow")
    parser.add_argument("--stats-cache", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats_cache = args.stats_cache or os.path.join(args.output_dir, "real_stats_cache")

    classifier = load_classifier(args.classifier_ckpt, device)
    feat_model = fid.build_feature_extractor(args.mode, device, use_dataparallel=False)

    class_dirs = list_class_dirs(args.samples_root)
    confusion = np.zeros((10, 10), dtype=np.int64)
    per_class = {}

    for class_id, class_dir in sorted(class_dirs.items()):
        samples = load_generated_samples(class_dir)
        preds = classify_samples(
            classifier, samples, device, batch_size=args.classifier_batch_size
        )
        for pred in preds:
            confusion[class_id, int(pred)] += 1

        acc = float((preds == class_id).mean())
        mu_fake, sigma_fake = compute_feature_stats(
            samples, feat_model, device, args.fid_batch_size, args.mode
        )
        mu_real, sigma_real = load_or_compute_real_stats(
            args.data_root,
            class_id,
            feat_model,
            device,
            args.fid_batch_size,
            args.mode,
            stats_cache,
        )
        cfid = float(fid.frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real))

        save_grid(
            samples,
            os.path.join(args.output_dir, f"class_{class_id}_grid.png"),
            args.grid_size,
        )

        per_class[class_id] = {
            "class_name": CIFAR10_CLASSES[class_id],
            "num_samples": int(samples.shape[0]),
            "conditional_accuracy": acc,
            "cfid": cfid,
        }
        print(
            f"class={class_id} name={CIFAR10_CLASSES[class_id]} "
            f"acc={acc:.4f} cfid={cfid:.4f} n={samples.shape[0]}"
        )

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    overall_acc = correct / total if total else 0.0
    mean_cfid = float(np.mean([item["cfid"] for item in per_class.values()]))

    summary = {
        "overall_conditional_accuracy": overall_acc,
        "mean_cfid": mean_cfid,
        "num_classes_evaluated": len(per_class),
        "per_class": per_class,
        "class_names": CIFAR10_CLASSES,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target\\pred"] + CIFAR10_CLASSES)
        for idx, row in enumerate(confusion):
            writer.writerow([CIFAR10_CLASSES[idx]] + row.tolist())

    with open(os.path.join(args.output_dir, "per_class_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "num_samples", "conditional_accuracy", "cfid"])
        for class_id, metrics in sorted(per_class.items()):
            writer.writerow(
                [
                    class_id,
                    metrics["class_name"],
                    metrics["num_samples"],
                    metrics["conditional_accuracy"],
                    metrics["cfid"],
                ]
            )

    plot_confusion(confusion, os.path.join(args.output_dir, "confusion_matrix.png"))
    print(f"overall_conditional_accuracy={overall_acc:.4f}")
    print(f"mean_cfid={mean_cfid:.4f}")


if __name__ == "__main__":
    main()
