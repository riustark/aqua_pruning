"""Generate the benchmark narrative directly from recorded numerical evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_report(data: dict, path: Path) -> None:
    """Use observed values throughout; standard-run error bars are population SD."""
    meta, cost = data["metadata"], data["cost_benchmark"]
    lines = ["# Trained model benchmark evidence", "", data["falsifiable_claim"], "",
             f"Seeds: {meta['seeds']}; epochs: {meta['epochs']}; hidden sizes: {meta['hidden_sizes']}.",
             "Each nonzero method/budget pair is trained independently. Dense baselines are shared.",
             "", "## Accuracy and sparsity", "",
             "Mean ± population standard deviation across seeds; these are not confidence intervals.",
             "Biases are excluded from the prunable weight budget. Integer budgets use rounding.", "",
             "| Method | Target | Achieved | Active weights | Test accuracy |",
             "|---|---:|---:|---:|---:|"]
    for method, rows in data["summary"].items():
        for row in rows.values():
            lines.append(f"| {method} | {row['target_sparsity']:.2f} | "
                         f"{row['achieved_sparsity_mean']*100:.4f}% | "
                         f"{row['active_connections_mean']:.0f} | "
                         f"{row['test_accuracy_mean']*100:.4f}% ± "
                         f"{row['test_accuracy_std']*100:.4f}% |")
    lines += ["", "## Paired method comparison", "",
              "Differences are saliency minus magnitude on the same seed, in percentage points.",
              "These descriptive results do not establish general superiority or equivalence.", "",
              "| Target | Paired differences | Mean difference | Sample SD |",
              "|---|---|---:|---:|"]
    for target in meta["sparsities"]:
        by_method = {method: {r["seed"]: r["test_accuracy"] for r in data["raw_runs"]
                             if r["method"] == method and r["target_sparsity"] == target}
                     for method in ("magnitude", "saliency")}
        diffs = np.array([100 * (by_method["saliency"][seed] - by_method["magnitude"][seed])
                          for seed in meta["seeds"]])
        sd = f"{diffs.std(ddof=1):.4f}" if len(diffs) > 1 else "not estimable (one seed)"
        lines.append(f"| {target:.2f} | {', '.join(f'{d:+.4f}' for d in diffs)} | "
                     f"{diffs.mean():+.4f} | {sd} |")
    source = cost["source_model"]
    lines += ["", "## Cost of the identified trained model", "",
              f"Snapshot: `{source['path']}`; seed {source['seed']}; method {source['method']}.",
              f"Model SHA-256: `{cost['model_sha256']}`.",
              f"Held-out accuracy: {cost['test_accuracy']*100:.4f}%; loss: {cost['test_loss']:.8f}.",
              f"Batch size: {cost['batch_size']}; warmups: {cost['warmup_reps']}; repetitions: {cost['benchmark_reps']}.",
              "All paths matched training-forward logits within rtol=atol=1e-10 and produced identical class predictions.",
              "Inference paths use the same weights and masks. Conversion is excluded from latency and recorded separately.",
              "Dense timing uses NumPy inference without constructing an autodiff graph.", "",
              "| Backend | Median µs | p95 µs | Maximum logit difference |",
              "|---|---:|---:|---:|"]
    for backend, timing in cost["timings"].items():
        lines.append(f"| {backend} | {timing['median_us']:.3f} | {timing['p95_us']:.3f} | "
                     f"{cost['max_abs_logit_error'][backend]:.3g} |")
    fastest = min(cost["timings"], key=lambda name: cost["timings"][name]["median_us"])
    lines += ["", f"Fastest observed backend in this measurement: `{fastest}`.",
              "Wall-clock measurements vary with machine load and environment; exact timing reproduction is not promised.",
              f"Sparse weight MACs per sample: {cost['sparse_macs_per_sample']} versus "
              f"{cost['dense_macs_per_sample']} for dense execution. Bias and activation operations are excluded.",
              "MAC reduction is not a claim of wall-clock speedup or reduced dense training cost.", "",
              "## Representation array payload", "",
              cost["memory_scope"] + ". The benchmark engine retains all representations together.", "",
              "| Representation | Array bytes |", "|---|---:|"]
    for name, count in cost["array_payload_bytes"].items():
        lines.append(f"| {name} | {count} |")
    lines += ["", "## Reproduction and regression", "",
              "Run from the repository root:", "", "```powershell",
              "python -m pytest -q",
              "python -m experiments.benchmark --seeds 3 --epochs 100 --output-dir results/verified --reference-results results/benchmark_results.json",
              "```", "",
              "The full command reproduces the default sweep; custom runs are identified by this report's configuration.",
              "Saved snapshots contain final inference weights, masks, configuration, and complete training histories.",
              "They do not contain resumable optimizer state. Training and pruning formulas are unchanged."]
    if "regression_check" in data:
        lines.append("Historical comparison: all raw accuracy/loss/count rows and summaries matched exactly.")
    lines += ["", "Environment and NumPy build/thread information are recorded in the JSON cost_benchmark.environment field.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    write_report(data, args.results.parent / "EVIDENCE.md")


if __name__ == "__main__":
    main()
