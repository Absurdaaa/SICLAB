#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from train_cifar10_classifier import CIFAR10_CLASSES, build_classifier


def build_test_loader(data_root: str, batch_size: int, num_workers: int) -> DataLoader:
    test_tf = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_ds = CIFAR10(data_root, train=False, download=True, transform=test_tf)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_checkpoint(path: str, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location=device)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loader = build_test_loader(args.data_root, args.batch_size, args.num_workers)
    model = load_checkpoint(args.checkpoint, device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = np.zeros((10, 10), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            for target, pred in zip(labels_np, preds_np):
                confusion[int(target), int(pred)] += 1

    test_loss = total_loss / total_count
    test_acc = total_correct / total_count

    per_class: List[Dict[str, float]] = []
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        class_total = int(confusion[class_id].sum())
        class_correct = int(confusion[class_id, class_id])
        class_acc = class_correct / class_total if class_total else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "num_samples": class_total,
                "accuracy": class_acc,
            }
        )

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "num_samples": total_count,
        "per_class": per_class,
        "class_names": CIFAR10_CLASSES,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(
        os.path.join(args.output_dir, "per_class_accuracy.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "num_samples", "accuracy"])
        for row in per_class:
            writer.writerow(
                [row["class_id"], row["class_name"], row["num_samples"], row["accuracy"]]
            )

    with open(
        os.path.join(args.output_dir, "confusion_matrix.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["target\\pred"] + CIFAR10_CLASSES)
        for idx, row in enumerate(confusion):
            writer.writerow([CIFAR10_CLASSES[idx]] + row.tolist())

    plot_confusion(confusion, os.path.join(args.output_dir, "confusion_matrix.png"))

    print(f"test_loss={test_loss:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    for row in per_class:
        print(
            f"class={row['class_id']} name={row['class_name']} "
            f"acc={row['accuracy']:.4f} n={row['num_samples']}"
        )


if __name__ == "__main__":
    main()
