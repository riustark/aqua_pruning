"""Multi-sparsity benchmark sweep, Pareto trade-off plot, and sparse cost analysis.

Run from the project root:

    python -m experiments.benchmark --seeds 3 --epochs 100

This module:
1. Sweeps target sparsities [0.0, 0.50, 0.75, 0.90, 0.95] for both magnitude
   and gradient-informed saliency pruning across multiple fixed seeds.
2. Measures theoretical work (MACs) and empirical wall-clock latency for both
   dense BLAS and a genuine sparse-aware forward path.
3. Generates a Pareto plot (Accuracy vs. Achieved Sparsity) saved to results/.
4. Saves complete raw JSON results and summary statistics.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated environments
import matplotlib.pyplot as plt
import numpy as np

from nn.layers import MLP
from train.train import train_baseline, train_pruned

DEFAULT_SPARSITIES = (0.0, 0.50, 0.75, 0.90, 0.95)
DEFAULT_SEEDS = (42, 43, 44)


# ----------------------------------------------------------------------
# Sparse-aware forward execution and cost analysis
# ----------------------------------------------------------------------
class SparseLinearRepresentation:
    """Pre-extracted sparse representation of a pruned Linear layer.

    Stores only non-zero coordinates and weights, genuinely skipping
    all pruned connections during forward evaluation.
    """

    def __init__(self, weight: np.ndarray, mask: np.ndarray, bias: np.ndarray) -> None:
        self.in_features = weight.shape[0]
        self.out_features = weight.shape[1]
        self.bias = bias.copy()

        # Coordinate format of active connections
        rows, cols = np.where(mask)
        self.active_rows = rows.astype(np.int32)
        self.active_cols = cols.astype(np.int32)
        self.active_weights = weight[rows, cols].copy()
        self.num_active = len(self.active_weights)
        self.total_weights = weight.size

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate forward pass skipping inactive connections."""
        num_samples = inputs.shape[0]
        output = np.tile(self.bias, (num_samples, 1))
        # Accumulate only active connection products
        for row, col, weight in zip(self.active_rows, self.active_cols, self.active_weights):
            output[:, col] += inputs[:, row] * weight
        return output


def sparse_aware_forward(
    sparse_layers: list[SparseLinearRepresentation],
    inputs: np.ndarray,
    activation: str = "relu",
) -> np.ndarray:
    """Forward pass through network that genuinely skips inactive weights."""
    hidden = inputs
    last_idx = len(sparse_layers) - 1
    for idx, layer in enumerate(sparse_layers):
        hidden = layer.forward(hidden)
        if idx < last_idx:
            hidden = np.maximum(hidden, 0.0) if activation == "relu" else np.tanh(hidden)
    return hidden


def measure_forward_costs(
    model: MLP,
    batch_size: int = 120,
    iterations: int = 200,
    seed: int = 0,
    inputs: np.ndarray | None = None,
    warmup_reps: int = 20,
) -> dict:
    """Compare theoretical work (MACs) and wall-clock latency for dense vs sparse execution.

    Args:
        model: Trained/pruned MLP.
        batch_size: Number of samples per forward evaluation.
        iterations: Number of timed repetitions.
        seed: Random seed for test input generation.

    Returns:
        Dict with MAC counts, theoretical reduction, and measured timings.
    """
    from experiments.optimized_inference import OptimizedInferenceEngine
    from train.snapshot import model_fingerprint

    if iterations <= 0 or warmup_reps < 0 or batch_size <= 0:
        raise ValueError("Need positive iterations/batch size and nonnegative warmup.")
    if inputs is None:
        inputs = np.random.default_rng(seed).normal(
            size=(batch_size, model.layers[0].in_features))
    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] != model.layer_sizes[0]:
        raise ValueError("Inputs must be a nonempty batch matching model input size.")
    before = model_fingerprint(model)
    engine = OptimizedInferenceEngine(model)
    backends = ("dense", "dense_cached", "sparse_coord", "sparse_csc_vectorized")
    # Compare every inference path against the actual training forward function.
    reference = model(inputs).data
    errors = {}
    for backend in backends:
        logits = engine.predict_logits(inputs, backend)
        np.testing.assert_allclose(logits, reference, rtol=1e-10, atol=1e-10)
        if not np.array_equal(logits.argmax(axis=1), reference.argmax(axis=1)):
            raise AssertionError(f"Class predictions differ for {backend}.")
        errors[backend] = float(np.max(np.abs(logits - reference)))
    for _ in range(warmup_reps):
        for backend in backends:
            engine.predict_logits(inputs, backend)
    durations = {backend: [] for backend in backends}
    # Rotate backend order to distribute timing drift across implementations.
    for repeat in range(iterations):
        offset = repeat % len(backends)
        for backend in backends[offset:] + backends[:offset]:
            start = time.perf_counter_ns()
            engine.predict_logits(inputs, backend)
            durations[backend].append((time.perf_counter_ns() - start) / 1000.0)
    stats = {backend: {
        "median_us": float(np.median(values)),
        "p95_us": float(np.percentile(values, 95)),
        "min_us": float(np.min(values)), "max_us": float(np.max(values)),
    } for backend, values in durations.items()}
    if before != model_fingerprint(model):
        raise AssertionError("Inference benchmark changed model weights or masks.")
    dense_macs_per_sample = sum(layer.weight.size for layer in model.layers)
    sparse_macs_per_sample = sum(layer.num_active() for layer in model.layers)
    dense_ms = stats["dense"]["median_us"] / 1000.0
    sparse_ms = stats["sparse_coord"]["median_us"] / 1000.0
    blas_config = io.StringIO()
    with contextlib.redirect_stdout(blas_config):
        np.show_config()
    return {
        "dense_macs_per_sample": int(dense_macs_per_sample),
        "sparse_macs_per_sample": int(sparse_macs_per_sample),
        "macs_reduction_fraction": float(1.0 - sparse_macs_per_sample / dense_macs_per_sample),
        "dense_wall_time_ms": dense_ms,
        "sparse_wall_time_ms": sparse_ms,
        "slowdown_factor": float(sparse_ms / max(dense_ms, 1e-9)),
        "model_sha256": before,
        "batch_size": int(inputs.shape[0]), "warmup_reps": warmup_reps,
        "benchmark_reps": iterations, "latency_statistic": "median",
        "timing_order": "rotating backends per repetition",
        "timings": stats, "max_abs_logit_error": errors,
        "class_predictions_match": True,
        "array_payload_bytes": engine.memory_footprints(),
        "memory_scope": "NumPy array payload per representation; excludes Python overhead and total engine memory",
        "conversion_times_ms": {
            "dense_cached": engine.cached_build_time_ms,
            "sparse_coord": engine.coord_build_time_ms,
            "sparse_csc_vectorized": engine.csc_build_time_ms,
        },
        "environment": {
            "platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "numpy": np.__version__,
            "thread_settings": {name: os.environ.get(name, "default") for name in
                                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "numpy_build_config": blas_config.getvalue(),
        },
    }


# ----------------------------------------------------------------------
# Multi-sparsity sweep runner
# ----------------------------------------------------------------------
def run_benchmark_sweep(
    sparsities: Sequence[float] = DEFAULT_SPARSITIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = 100,
    hidden_sizes: Sequence[int] = (64, 64),
    learning_rate: float = 0.01,
    artifact_dir: Path | None = None,
) -> dict:
    """Execute the multi-sparsity experiment across methods and seeds.

    Runs both 'magnitude' and 'saliency' at matching budgets. At sparsity 0.0,
    a single dense baseline model is evaluated for both to ensure identical baselines.
    """
    raw_runs: list[dict] = []
    print(f"Starting sweep over sparsities {list(sparsities)} with seeds {list(seeds)} ({epochs} epochs)...")

    # Cache baseline runs at sparsity 0.0 so we do not repeat identical dense training
    baseline_cache: dict[int, dict] = {}
    artifacts: list[dict] = []
    def snapshot_path(method: str, target: float, seed: int) -> Path | None:
        if artifact_dir is None:
            return None
        name = f"{method}_s{str(float(target)).replace('.', 'p')}_seed{seed}.npz"
        return artifact_dir / name

    for seed in seeds:
        print(f"  Training dense baseline (seed {seed})...")
        baseline_cache[seed] = train_baseline(
            epochs=epochs, seed=seed, hidden_sizes=hidden_sizes, learning_rate=learning_rate,
            snapshot_path=snapshot_path("baseline", 0.0, seed),
        )

    for method in ("magnitude", "saliency"):
        print(f"\nEvaluating method: {method}")
        for target_sparsity in sparsities:
            for seed in seeds:
                if target_sparsity == 0.0:
                    run_result = baseline_cache[seed]
                else:
                    print(f"  Pruning to {target_sparsity:.2f} using {method} (seed {seed})...")
                    run_result = train_pruned(
                        method=method,
                        sparsity=target_sparsity,
                        epochs=epochs,
                        seed=seed,
                        hidden_sizes=hidden_sizes,
                        learning_rate=learning_rate,
                        snapshot_path=snapshot_path(method, target_sparsity, seed),
                    )

                final = run_result["final"]
                raw_runs.append({
                    "method": method,
                    "target_sparsity": float(target_sparsity),
                    "seed": int(seed),
                    "achieved_sparsity": float(final["sparsity"]),
                    "active_connections": int(final["active_connections"]),
                    "total_prunable": int(final["total_prunable"]),
                    "test_accuracy": float(final["test_accuracy"]),
                    "test_loss": float(final["test_loss"]),
                    "train_accuracy": float(final["train_accuracy"]),
                    "train_loss": float(final["train_loss"]),
                })
                if artifact_dir is not None:
                    from train.snapshot import load_snapshot
                    path = snapshot_path("baseline" if target_sparsity == 0 else method,
                                         target_sparsity, seed)
                    _, metadata = load_snapshot(path)
                    artifacts.append({
                        "method": method, "target_sparsity": float(target_sparsity),
                        "seed": int(seed), "path": path.relative_to(artifact_dir.parent).as_posix(),
                        "model_sha256": metadata["model_sha256"],
                    })

    # Aggregate statistics per (method, target_sparsity)
    summary: dict[str, dict[str, dict]] = {"magnitude": {}, "saliency": {}}
    for method in ("magnitude", "saliency"):
        for target in sparsities:
            matching = [
                r for r in raw_runs
                if r["method"] == method and r["target_sparsity"] == target
            ]
            test_accs = [r["test_accuracy"] for r in matching]
            test_losses = [r["test_loss"] for r in matching]
            achieved = [r["achieved_sparsity"] for r in matching]
            active = [r["active_connections"] for r in matching]

            summary[method][f"{target:.2f}"] = {
                "target_sparsity": float(target),
                "achieved_sparsity_mean": float(np.mean(achieved)),
                "active_connections_mean": float(np.mean(active)),
                "test_accuracy_mean": float(np.mean(test_accs)),
                "test_accuracy_std": float(np.std(test_accs)),
                "test_loss_mean": float(np.mean(test_losses)),
                "test_loss_std": float(np.std(test_losses)),
                "num_seeds": len(matching),
            }

    result = {
        "metadata": {
            "sparsities": [float(s) for s in sparsities],
            "seeds": [int(s) for s in seeds],
            "epochs": epochs,
            "hidden_sizes": list(hidden_sizes),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "raw_runs": raw_runs,
        "summary": summary,
    }
    if artifact_dir is not None:
        result["model_artifacts"] = artifacts
    return result


# ----------------------------------------------------------------------
# Pareto plot generation
# ----------------------------------------------------------------------
def generate_pareto_plot(benchmark_data: dict, output_path: Path) -> None:
    """Generate and save the Accuracy vs. Achieved Sparsity Pareto curve."""
    summary = benchmark_data["summary"]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    styles = {
        "magnitude": {"color": "#1f77b4", "marker": "o", "label": "Magnitude Pruning (|W|)"},
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

    # Random chance baseline (3 classes on spirals = 33.3%)
    ax.axhline(1.0 / 3.0, color="gray", linestyle="--", linewidth=1.2, label="Chance Accuracy (33.3%)")

    ax.set_title("AQUA Self-Pruning Pareto Frontier: Accuracy vs. Achieved Sparsity", fontsize=12, pad=12)
    ax.set_xlabel("Achieved Sparsity (Fraction of Connections Pruned)", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.02, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Pareto plot to: {output_path}")


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-sparsity pruning benchmark sweep.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of fixed seeds to evaluate (e.g. 3).")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per training run.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64], help="Hidden layer sizes.")
    parser.add_argument("--sparsities", type=float, nargs="+", default=list(DEFAULT_SPARSITIES),
                        help="Target sparsities to sweep, e.g. 0.0 0.50 0.75 0.90 0.95")
    parser.add_argument("--output-dir", type=Path, default=Path("results/verified"),
                        help="Directory to save JSON results and plots.")
    parser.add_argument("--reference-results", type=Path, default=None,
                        help="Require exact historical accuracy/loss/count reproduction.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seeds < 1 or args.epochs < 1 or args.reps < 1 or args.warmup < 0:
        raise ValueError("Seeds, epochs, reps must be positive; warmup nonnegative.")
    if not args.sparsities or any(not 0 <= s < 1 for s in args.sparsities):
        raise ValueError("Need at least one sparsity, each in [0, 1).")
    if len({f"{s:.2f}" for s in args.sparsities}) != len(args.sparsities):
        raise ValueError("Sparsities must have distinct two-decimal summary keys.")
    seed_list = [42 + i for i in range(args.seeds)]
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run the multi-sparsity sweep across methods and seeds
    results = run_benchmark_sweep(
        sparsities=args.sparsities,
        seeds=seed_list,
        epochs=args.epochs,
        hidden_sizes=args.hidden,
        artifact_dir=output_dir / "models",
    )

    if args.reference_results is not None:
        reference = json.loads(args.reference_results.read_text(encoding="utf-8"))
        if results["raw_runs"] != reference["raw_runs"] or results["summary"] != reference["summary"]:
            raise AssertionError("Training results differ from the supplied historical reference.")
        results["regression_check"] = {"reference": args.reference_results.as_posix(),
                                      "raw_runs_exact_match": True, "summary_exact_match": True}

    # Measure the exact trained saliency artifact represented in the Pareto rows.
    from train.snapshot import load_snapshot, snapshot_test_data
    from train.train import evaluate
    target = 0.90 if 0.90 in args.sparsities else max(args.sparsities)
    artifact = next(a for a in results["model_artifacts"] if a["method"] == "saliency"
                    and a["target_sparsity"] == target and a["seed"] == seed_list[0])
    model, metadata = load_snapshot(output_dir / artifact["path"])
    test_x, test_y = snapshot_test_data(metadata["result"])
    test_loss, test_accuracy = evaluate(model, test_x, test_y)
    final = metadata["result"]["final"]
    if test_loss != final["test_loss"] or test_accuracy != final["test_accuracy"]:
        raise AssertionError("Snapshot test metrics differ from the recorded trained model.")
    print("\nMeasuring the saved trained model on its held-out inputs...")
    cost_metrics = measure_forward_costs(model, inputs=test_x, iterations=args.reps,
                                        warmup_reps=args.warmup)
    cost_metrics["source_model"] = artifact
    cost_metrics["test_accuracy"] = test_accuracy
    cost_metrics["test_loss"] = test_loss
    cost_metrics["input_source"] = "recorded held-out test split"
    results["cost_benchmark"] = cost_metrics

    # Claims use observed values and the actual custom architecture/budget.
    claim = build_falsifiable_claim(results, target)
    results["falsifiable_claim"] = claim
    print(f"\n{claim}\n")

    # 4. Save raw JSON data and Pareto plot
    json_path = output_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved raw experiment results to: {json_path}")

    plot_path = output_dir / "pareto_plot.png"
    generate_pareto_plot(results, plot_path)
    from experiments.evidence_report import write_report
    write_report(results, output_dir / "EVIDENCE.md")

    # Print summary table
    print("\n============================== BENCHMARK SUMMARY ==============================")
    print(f"{'Method':<12} | {'Target':<7} | {'Achieved':<9} | {'Active/Total':<14} | {'Test Acc (Mean +/- Std)':<22}")
    print("-" * 75)
    for method in ("magnitude", "saliency"):
        for target in args.sparsities:
            info = results["summary"][method][f"{target:.2f}"]
            achieved = info["achieved_sparsity_mean"]
            active = int(round(info["active_connections_mean"]))
            total = int(round(active / (1.0 - achieved))) if achieved < 1.0 else 0
            acc_str = f"{info['test_accuracy_mean']*100:.1f}% +/- {info['test_accuracy_std']*100:.1f}%"
            print(f"{method:<12} | {target:<7.2f} | {achieved:<9.4f} | {active:>4}/{total:<8} | {acc_str:<22}")
    print("===============================================================================\n")

    return 0


def build_falsifiable_claim(results: dict, target: float) -> str:
    """Describe observed means without assuming an accuracy threshold or speedup."""
    key = f"{target:.2f}"
    mag = results["summary"]["magnitude"][key]
    sal = results["summary"]["saliency"][key]
    cost = results["cost_benchmark"]
    return (
        f"At {sal['achieved_sparsity_mean']*100:.4f}% achieved weight sparsity "
        f"({cost['sparse_macs_per_sample']} active out of {cost['dense_macs_per_sample']} weights), "
        f"mean test accuracy across {sal['num_seeds']} paired seeds is "
        f"{mag['test_accuracy_mean']*100:.4f}% for magnitude and "
        f"{sal['test_accuracy_mean']*100:.4f}% for saliency. "
        f"For the identified trained saliency snapshot, sparse coordinate inference uses "
        f"{cost['macs_reduction_fraction']*100:.4f}% fewer weight MACs and its median latency "
        f"is {cost['slowdown_factor']:.3f} times the dense NumPy inference latency "
        f"at batch {cost['batch_size']} on the recorded environment. "
        "This timing ratio is measured on one model, not averaged across seeds."
    )


if __name__ == "__main__":
    sys.exit(main())
