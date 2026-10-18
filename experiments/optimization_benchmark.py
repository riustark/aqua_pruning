"""Phase 4: Honest CPU Latency & Memory Benchmark.

Compares all 4 inference backends across batch sizes [1, 16, 128] and
sparsities [0.0, 0.50, 0.75, 0.90, 0.95] with warm-up, repeated measurements,
median/p95 latency, conversion costs, and memory footprints.

Outputs are written exclusively to results/optimization/.
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
from nn.layers import MLP
from nn.optimizers import Adam
from prune.importance import compute_layer_scores
from prune.scheduler import prune_to_sparsity

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
) -> dict:
    """Run full matrix of batch sizes, sparsities, and backends."""
    results: list[dict] = []
    rng = np.random.default_rng(seed)

    print(f"Running CPU inference benchmark across sparsities {list(sparsities)} and batches {list(batch_sizes)}...")

    for sparsity in sparsities:
        # Build model and prune to target
        model = MLP([2, 64, 64, 3], np.random.default_rng(seed))
        if sparsity > 0.0:
            scores = compute_layer_scores(model, "magnitude")
            prune_to_sparsity(model, Adam(model.parameters()), scores, sparsity)

        # Build optimized inference engine (measures conversion time)
        engine = OptimizedInferenceEngine(model)
        footprints = engine.memory_footprints()
        achieved_sparsity, active_conn, total_prunable = model.sparsity()

        # Theoretical MACs per sample
        dense_macs = total_prunable
        sparse_macs = active_conn

        for batch_size in batch_sizes:
            inputs = rng.normal(size=(batch_size, 2))

            # Verify equivalence before timing
            ref = engine.predict_logits(inputs, "dense")
            d_cached = engine.predict_logits(inputs, "dense_cached")
            s_coord = engine.predict_logits(inputs, "sparse_coord")
            s_csc = engine.predict_logits(inputs, "sparse_csc_vectorized")

            np.testing.assert_allclose(ref, d_cached, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(ref, s_coord, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(ref, s_csc, rtol=1e-10, atol=1e-10)

            # Measure latency for each backend
            t_dense = measure_backend_latency(engine, "dense", inputs, warmup_reps, benchmark_reps)
            t_cached = measure_backend_latency(engine, "dense_cached", inputs, warmup_reps, benchmark_reps)
            t_coord = measure_backend_latency(engine, "sparse_coord", inputs, warmup_reps, benchmark_reps)
            t_csc = measure_backend_latency(engine, "sparse_csc_vectorized", inputs, warmup_reps, benchmark_reps)

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
                "speedup_csc_vs_coord": float(t_coord["median_us"] / max(t_csc["median_us"], 1e-6)),
                "speedup_cached_vs_dense": float(t_dense["median_us"] / max(t_cached["median_us"], 1e-6)),
                "relative_to_dense_blass": float(t_csc["median_us"] / max(t_dense["median_us"], 1e-6)),
                "footprints": footprints,
                "conversion_times_ms": {
                    "dense_cached": engine.cached_build_time_ms,
                    "sparse_coord": engine.coord_build_time_ms,
                    "sparse_csc": engine.csc_build_time_ms,
                },
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
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "default"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU inference latency benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/optimization"))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_REPS)
    parser.add_argument("--reps", type=int, default=DEFAULT_BENCHMARK_REPS)
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = run_cpu_benchmark(warmup_reps=args.warmup, benchmark_reps=args.reps)

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
