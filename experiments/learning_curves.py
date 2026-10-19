"""Plot recorded training histories without training or modifying a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train.snapshot import load_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("results/verified/models"))
    parser.add_argument("--output", type=Path, default=Path("results/verified/learning_curves.png"))
    args = parser.parse_args()
    sources = [("Dense", "baseline_s0p0_seed42.npz"),
               ("Magnitude, 90% target", "magnitude_s0p9_seed42.npz"),
               ("Saliency, 90% target", "saliency_s0p9_seed42.npz")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    records = []
    for column, (label, filename) in enumerate(sources):
        _, metadata = load_snapshot(args.models / filename)
        result = metadata["result"]
        history = result["history"]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        for row, metric in enumerate(("loss", "accuracy")):
            ax = axes[row, column]
            for split in ("train", "test"):
                values = np.asarray(history[f"{split}_{metric}"])
                if values.shape != epochs.shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid history: {filename}, {split}_{metric}")
                ax.plot(epochs, values, label="Training" if split == "train" else "Held-out")
            if result["config"]["mode"] == "prune":
                ax.axvspan(result["config"]["prune_start_epoch"],
                           result["config"]["prune_end_epoch"], alpha=0.1, color="gray")
            ax.set_ylabel("Cross-entropy" if metric == "loss" else "Accuracy")
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(label)
                ax.legend()
            else:
                ax.set_xlabel("Epoch")
                ax.set_ylim(0, 1.02)
        records.append({"source": filename, "model_sha256": metadata["model_sha256"],
                        "config": result["config"], "history": history})
    fig.suptitle("Recorded learning curves — seed 42; shaded interval: gradual pruning")
    fig.text(0.5, 0.015, "Training loss: mini-batch mean during updates. Held-out loss and accuracies: end of epoch.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} and raw history JSON")


if __name__ == "__main__":
    main()
