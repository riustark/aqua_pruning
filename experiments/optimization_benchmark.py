"""Phase 4: Honest CPU Latency & Memory Benchmark.

Compares all 4 inference backends across batch sizes [1, 16, 128] and
sparsities [0.0, 0.50, 0.75, 0.90, 0.95] with warm-up, repeated measurements,
median/p95 latency, conversion costs, and memory footprints.

Outputs default to results/verified/optimization/; the CLI can override this.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.optimized_inference import OptimizedInferenceEngine
from experiments.benchmark import measure_forward_costs
from train.snapshot import load_snapshot, snapshot_test_data
from train.train import evaluate

BATCH_SIZES = (1, 16, 128)
SPARSITY_LEVELS = (0.0, 0.50, 0.75, 0.90, 0.95)
DEFAULT_WARMUP_REPS = 50
DEFAULT_BENCHMARK_REPS = 200


def measure_backend_latency(
    engine: OptimizedInferenceEngine,
    backend: str,
    inputs: np.ndarray,
    warmup_reps: int = DEFAULT_WARMUP_REPS,
    benchmark_reps: int = DEFAULT_BENCHMARK_REPS,
) -> dict:
    """Measure median, p95, min, and max latency in microseconds over repeated runs."""
    # Warmup
    for _ in range(warmup_reps):
        _ = engine.predict_logits(inputs, backend=backend)

    # Timed runs
    durations_ns: list[int] = []
    for _ in range(benchmark_reps):
        t0 = time.perf_counter_ns()
        _ = engine.predict_logits(inputs, backend=backend)
        durations_ns.append(time.perf_counter_ns() - t0)

    durations_us = np.array(durations_ns, dtype=np.float64) / 1000.0
    return {
        "median_us": float(np.median(durations_us)),
        "p95_us": float(np.percentile(durations_us, 95)),
        "mean_us": float(np.mean(durations_us)),
        "std_us": float(np.std(durations_us)),
        "min_us": float(np.min(durations_us)),
        "max_us": float(np.max(durations_us)),
    }


def run_cpu_benchmark(
    sparsities: Sequence[float] = SPARSITY_LEVELS,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    seed: int = 42,
    warmup_reps: int = DEFAULT_WARMUP_REPS,
    benchmark_reps: int = DEFAULT_BENCHMARK_REPS,
    benchmark_results: Path = Path("results/verified/benchmark_results.json"),
    method: str = "saliency",
) -> dict:
    """Time trained snapshots already measured by the accuracy sweep."""
    results: list[dict] = []
    benchmark_results = Path(benchmark_results)
    sweep = json.loads(benchmark_results.read_text(encoding="utf-8"))

    print(f"Running CPU inference benchmark across sparsities {list(sparsities)} and batches {list(batch_sizes)}...")

    for sparsity in sparsities:
        artifact = next((a for a in sweep["model_artifacts"] if a["method"] == method
                         and a["target_sparsity"] == sparsity and a["seed"] == seed), None)
        if artifact is None:
            raise ValueError(f"No trained artifact for {method}, sparsity={sparsity}, seed={seed}.")
        model, metadata = load_snapshot(benchmark_results.parent / artifact["path"])
        if metadata["model_sha256"] != artifact["model_sha256"]:
            raise ValueError("Sweep and snapshot model fingerprints differ.")
        test_x, test_y = snapshot_test_data(metadata["result"])
        test_loss, test_accuracy = evaluate(model, test_x, test_y)
        final = metadata["result"]["final"]
        if test_loss != final["test_loss"] or test_accuracy != final["test_accuracy"]:
            raise AssertionError("Loaded model no longer reproduces recorded test metrics.")
        achieved_sparsity, active_conn, total_prunable = model.sparsity()

        # Theoretical MACs per sample
        dense_macs = total_prunable
        sparse_macs = active_conn

        for batch_size in batch_sizes:
            if batch_size <= 0:
                raise ValueError("Batch sizes must be positive.")
            inputs = np.tile(test_x, ((batch_size + len(test_x) - 1) // len(test_x), 1))[:batch_size]
            cost = measure_forward_costs(model, inputs=inputs, iterations=benchmark_reps,
                                         warmup_reps=warmup_reps)
            t_dense, t_cached, t_coord, t_csc = (cost["timings"][name] for name in
                ("dense", "dense_cached", "sparse_coord", "sparse_csc_vectorized"))

            results.append({
                "sparsity": float(sparsity),
                "achieved_sparsity": float(achieved_sparsity),
                "active_connections": int(active_conn),
                "total_prunable": int(total_prunable),
                "batch_size": int(batch_size),
                "dense_macs_per_sample": int(dense_macs),
                "sparse_macs_per_sample": int(sparse_macs),
                "dense": t_dense,
                "dense_cached": t_cached,
                "sparse_coord": t_coord,
                "sparse_csc_vectorized": t_csc,
                "source_model": artifact,
                "model_sha256": cost["model_sha256"],
                "test_accuracy": test_accuracy, "test_loss": test_loss,
                "max_abs_logit_error": cost["max_abs_logit_error"],
                "class_predictions_match": cost["class_predictions_match"],
                "speedup_csc_vs_coord": float(t_coord["median_us"] / max(t_csc["median_us"], 1e-6)),
                "speedup_cached_vs_dense": float(t_dense["median_us"] / max(t_cached["median_us"], 1e-6)),
                "relative_to_dense_blas": float(t_csc["median_us"] / max(t_dense["median_us"], 1e-6)),
                "array_payload_bytes": cost["array_payload_bytes"],
                "memory_scope": cost["memory_scope"],
                "conversion_times_ms": cost["conversion_times_ms"],
            })

    return {
        "metadata": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "warmup_reps": warmup_reps,
            "benchmark_reps": benchmark_reps,
            "seed": seed,
            "method": method,
            "benchmark_results": benchmark_results.as_posix(),
            "input_source": "held-out split, repeated deterministically when batch exceeds test size",
            "timing_order": "rotating backends per repetition",
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "default"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU inference latency benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/verified/optimization"))
    parser.add_argument("--benchmark-results", type=Path, default=Path("results/verified/benchmark_results.json"))
    parser.add_argument("--method", choices=("magnitude", "saliency"), default="saliency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_REPS)
    parser.add_argument("--reps", type=int, default=DEFAULT_BENCHMARK_REPS)
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = run_cpu_benchmark(warmup_reps=args.warmup, benchmark_reps=args.reps,
                             benchmark_results=args.benchmark_results, method=args.method, seed=args.seed)

    json_path = output_dir / "inference_benchmark_results.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to: {json_path}")

    # Print summary table
    print("\n=============================== INFERENCE LATENCY SUMMARY ===============================")
    print(f"{'Sparsity':<9} | {'Batch':<6} | {'Dense (us)':<12} | {'Cached (us)':<12} | {'Coord (us)':<12} | {'CSC Vec (us)':<12} | {'CSC vs Coord':<12}")
    print("-" * 88)
    for row in data["results"]:
        sp = f"{row['achieved_sparsity']*100:.1f}%"
        bs = row["batch_size"]
        d_us = f"{row['dense']['median_us']:.1f}"
        c_us = f"{row['dense_cached']['median_us']:.1f}"
        coord_us = f"{row['sparse_coord']['median_us']:.1f}"
        csc_us = f"{row['sparse_csc_vectorized']['median_us']:.1f}"
        speedup = f"{row['speedup_csc_vs_coord']:.2f}x faster" if row['speedup_csc_vs_coord'] >= 1.0 else f"{1.0/row['speedup_csc_vs_coord']:.2f}x slower"
        print(f"{sp:<9} | {bs:<6} | {d_us:<12} | {c_us:<12} | {coord_us:<12} | {csc_us:<12} | {speedup:<12}")
    print("=========================================================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
