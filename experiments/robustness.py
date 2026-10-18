"""Phase 5: Robustness Experiment under Challenging Data Distributions.

Evaluates magnitude vs. first-order Taylor saliency pruning under challenging
conditions (higher noise, overlapping class boundaries) across 5 independent seeds:
seeds = [101, 102, 103, 104, 105].

Does not modify the original train/dataset.py or benchmark baselines.
All outputs are saved to results/optimization/.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn.layers import MLP
from nn.optimizers import Adam
from prune.importance import compute_layer_scores
from prune.scheduler import prune_to_sparsity, target_sparsity_at_step
from train.dataset import iterate_minibatches, train_test_split
from train.train import evaluate, prune_epoch_range

ROBUST_SEEDS = (101, 102, 103, 104, 105)
SPARSITY_TARGETS = (0.0, 0.50, 0.75, 0.90, 0.95)


def make_challenging_spirals(
    samples_per_class: int = 200,
    num_classes: int = 3,
    noise: float = 0.35,  # Higher noise -> substantial arm overlap
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate spirals with increased Gaussian angular noise and tighter winding."""
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.0, 1.0, samples_per_class)
    features = np.empty((samples_per_class * num_classes, 2), dtype=np.float64)
    labels = np.empty(samples_per_class * num_classes, dtype=np.int64)

    for class_index in range(num_classes):
        start = class_index * samples_per_class
        stop = start + samples_per_class
        # Angle winding 5.5 (tighter than baseline 4.0) + noise 0.35 (vs baseline 0.20)
        angle = 5.0 * class_index + 5.5 * radius + rng.normal(0.0, noise, samples_per_class)
        features[start:stop, 0] = radius * np.sin(angle)
        features[start:stop, 1] = radius * np.cos(angle)
        labels[start:stop] = class_index

    return features, labels


def train_robust_model(
    mode: str = "baseline",
    method: str = "magnitude",
    sparsity: float = 0.0,
    epochs: int = 80,
    seed: int = 101,
    hidden_sizes: Sequence[int] = (64, 64),
    learning_rate: float = 0.01,
    batch_size: int = 32,
    noise: float = 0.35,
) -> dict:
    """Train on the challenging spiral dataset."""
    data_seq, split_seq, init_seq, shuffle_seq = np.random.SeedSequence(seed).spawn(4)
    features, labels = make_challenging_spirals(
        samples_per_class=200, num_classes=3, noise=noise, seed=int(data_seq.generate_state(1)[0])
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_fraction=0.2, seed=int(split_seq.generate_state(1)[0])
    )

    layer_sizes = [features.shape[1], *hidden_sizes, 3]
    model = MLP(layer_sizes, np.random.default_rng(init_seq), activation="relu")
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    shuffle_rng = np.random.default_rng(shuffle_seq)
    prune_start, prune_end = prune_epoch_range(epochs, 0.1, 0.6)

    for epoch in range(1, epochs + 1):
        for bx, by in iterate_minibatches(train_x, train_y, batch_size, shuffle_rng):
            optimizer.zero_grad()
            from engine.tensor import Tensor
            from engine import operations as ops
            loss = ops.softmax_cross_entropy(model(Tensor(bx)), by)
            loss.backward()
            optimizer.step()

        if mode == "prune" and prune_start <= epoch <= prune_end:
            step_target = target_sparsity_at_step(epoch, sparsity, prune_start, prune_end)
            scores = compute_layer_scores(model, method, train_x, train_y)
            prune_to_sparsity(model, optimizer, scores, step_target)

    test_loss, test_acc = evaluate(model, test_x, test_y)
    train_loss, train_acc = evaluate(model, train_x, train_y)
    achieved_sparsity, active_conn, total_prunable = model.sparsity()

    return {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "train_accuracy": float(train_acc),
        "train_loss": float(train_loss),
        "achieved_sparsity": float(achieved_sparsity),
        "active_connections": int(active_conn),
        "total_prunable": int(total_prunable),
    }


def run_robustness_sweep(
    sparsities: Sequence[float] = SPARSITY_TARGETS,
    seeds: Sequence[int] = ROBUST_SEEDS,
    epochs: int = 80,
    noise: float = 0.35,
) -> dict:
    """Run sweep over 5 seeds under challenging data conditions."""
    raw_runs: list[dict] = []
    print(f"Running robustness sweep on challenging dataset (noise={noise}) across seeds {list(seeds)}...")

    # Cache 0.0 baseline per seed
    baseline_cache: dict[int, dict] = {}
    for seed in seeds:
        baseline_cache[seed] = train_robust_model(
            mode="baseline", sparsity=0.0, epochs=epochs, seed=seed, noise=noise
        )

    for method in ("magnitude", "saliency"):
        for sparsity in sparsities:
            for seed in seeds:
                if sparsity == 0.0:
                    run_result = baseline_cache[seed]
                else:
                    run_result = train_robust_model(
                        mode="prune", method=method, sparsity=sparsity, epochs=epochs, seed=seed, noise=noise
                    )

                raw_runs.append({
                    "method": method,
                    "target_sparsity": float(sparsity),
                    "seed": int(seed),
                    "achieved_sparsity": float(run_result["achieved_sparsity"]),
                    "active_connections": int(run_result["active_connections"]),
                    "total_prunable": int(run_result["total_prunable"]),
                    "test_accuracy": float(run_result["test_accuracy"]),
                    "test_loss": float(run_result["test_loss"]),
                    "train_accuracy": float(run_result["train_accuracy"]),
                    "train_loss": float(run_result["train_loss"]),
                })

    summary: dict[str, dict[str, dict]] = {"magnitude": {}, "saliency": {}}
    for method in ("magnitude", "saliency"):
        for target in sparsities:
            matching = [r for r in raw_runs if r["method"] == method and r["target_sparsity"] == target]
            accs = [r["test_accuracy"] for r in matching]
            losses = [r["test_loss"] for r in matching]
            n = len(accs)
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs, ddof=1)) if n > 1 else 0.0
            # 95% t-interval half-width for n=5 (t_crit = 2.776)
            ci95_acc = float(2.776 * (std_acc / np.sqrt(n))) if n == 5 else 0.0

            summary[method][f"{target:.2f}"] = {
                "target_sparsity": float(target),
                "achieved_sparsity_mean": float(np.mean([r["achieved_sparsity"] for r in matching])),
                "test_accuracy_mean": mean_acc,
                "test_accuracy_std": std_acc,
                "test_accuracy_ci95": ci95_acc,
                "test_loss_mean": float(np.mean(losses)),
                "test_loss_std": float(np.std(losses, ddof=1)) if n > 1 else 0.0,
                "num_seeds": n,
            }

    return {
        "metadata": {
            "noise": noise,
            "seeds": list(seeds),
            "sparsities": list(sparsities),
            "epochs": epochs,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "raw_runs": raw_runs,
        "summary": summary,
    }


def generate_robustness_plot(data: dict, output_path: Path) -> None:
    """Plot Accuracy vs. Sparsity under challenging noise conditions."""
    summary = data["summary"]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    styles = {
        "magnitude": {"color": "#1f77b4", "marker": "o", "label": "Magnitude (|W|)"},
        "saliency": {"color": "#d62728", "marker": "s", "label": "Taylor Saliency (|g * W|)"},
    }

    for method, style in styles.items():
        targets = sorted(summary[method].keys(), key=lambda k: float(k))
        sparsities = [summary[method][t]["achieved_sparsity_mean"] for t in targets]
        acc_means = [summary[method][t]["test_accuracy_mean"] for t in targets]
        acc_stds = [summary[method][t]["test_accuracy_std"] for t in targets]

        ax.errorbar(
            sparsities,
            acc_means,
            yerr=acc_stds,
            fmt=f"-{style['marker']}",
            color=style["color"],
            label=style["label"],
            capsize=4,
            capthick=1.5,
            linewidth=2,
            markersize=6,
        )

    ax.axhline(1.0 / 3.0, color="gray", linestyle="--", linewidth=1.2, label="Chance Accuracy (33.3%)")
    ax.set_title("Robustness Stress Test: Accuracy vs. Sparsity (High Noise = 0.35)", fontsize=12, pad=12)
    ax.set_xlabel("Achieved Sparsity", fontsize=11)
    ax.set_ylabel("Test Accuracy (Mean +/- 1 Std across 5 seeds)", fontsize=11)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlim(-0.02, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved robustness plot to: {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Robustness experiment under noise.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/optimization"))
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = run_robustness_sweep(noise=args.noise, epochs=args.epochs)

    json_path = output_dir / "robustness_results.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nSaved robustness raw data to: {json_path}")

    plot_path = output_dir / "robustness_pareto_plot.png"
    generate_robustness_plot(data, plot_path)

    print("\n========================= ROBUSTNESS SUMMARY (NOISE = 0.35) =========================")
    print(f"{'Method':<12} | {'Target':<7} | {'Achieved':<9} | {'Test Acc (Mean +/- Std)':<24} | {'95% CI Half-Width':<18}")
    print("-" * 79)
    for method in ("magnitude", "saliency"):
        for target in SPARSITY_TARGETS:
            info = data["summary"][method][f"{target:.2f}"]
            achieved = info["achieved_sparsity_mean"]
            acc_str = f"{info['test_accuracy_mean']*100:.1f}% +/- {info['test_accuracy_std']*100:.1f}%"
            ci_str = f"+/- {info['test_accuracy_ci95']*100:.2f}%"
            print(f"{method:<12} | {target:<7.2f} | {achieved:<9.4f} | {acc_str:<24} | {ci_str:<18}")
    print("=====================================================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
