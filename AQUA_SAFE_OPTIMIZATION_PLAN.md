# AQUA Self-Pruning Neural Network — Safe Validation & Optimization Plan

## Role and objective
Act as a senior Python/NumPy ML engineer. Improve the **existing AQUA project** without breaking or silently changing its working implementation, public APIs, original experimental results, or current file structure.

**Primary goal:** verify prediction equivalence and experimental fairness; then experiment with an optional faster inference implementation while retaining the strong accuracy-versus-sparsity result. Do **not** claim a speedup unless reproducible wall-clock measurements demonstrate one.

**Known reference results (from existing runs, not guaranteed in every environment):** 115 tests passed; topology `[2, 64, 64, 3]` with 4,416 prunable weights; seeds 42/43/44; 0/50/75/90/95% sparsity; about 98–99% test accuracy; 90% sparsity leaves 442 active weights. Existing timing sample: dense NumPy approximately 0.29 ms/batch, sparse coordinate forward approximately 1.23 ms/batch. Treat these as baseline evidence to reproduce, not hardcoded expectations.

## Non-negotiable safety rules

1. **Never overwrite or delete the current working version or saved results.** Before writing code, inspect `git status`, commit the existing state if needed (ask permission before committing uncommitted user changes), and create a new branch named `experiment/safe-inference-optimization`. If git is unavailable, make a timestamped full-project backup outside the project folder before editing.
2. Prefer adding an isolated experimental module and tests over editing core files. Leave `engine/tensor.py`, `engine/operations.py`, `nn/layers.py`, `nn/optimizers.py`, `prune/importance.py`, `prune/scheduler.py`, `train/dataset.py`, and `train/train.py` **unchanged** unless a demonstrated bug requires a minimal, reviewed fix. Do not change their public signatures or training behavior.
3. Do not replace existing `experiments/benchmark.py` or `results/benchmark_results.json` / `results/pareto_plot.png`. Use **new, clearly named output files** under `results/optimization/`; keep generated artifacts ignored or separately tracked as appropriate.
4. Do not alter the 3-arm spiral baseline data generator, split logic, seeds, training schedule, accuracy calculation, or pruning criteria to make results look better. Harder data is a **separate experiment**, never the replacement baseline.
5. Do not add PyTorch, TensorFlow, JAX, CUDA, or new runtime dependencies. Use installed NumPy, Matplotlib, pytest and Python standard library. If a proposed technique needs a new dependency, document it and stop for approval.
6. Changes must be incremental: implement, test, and report each phase before starting the next. **Stop if any existing test fails or baseline reproducibility is lost.** Do not auto-fix by weakening tests or altering thresholds.
7. Preserve honest performance claims: parameter sparsity and theoretical MAC reduction are not the same as lower memory footprint or wall-clock acceleration. Dense arrays with zeros still occupy dense storage; separate index structures may create overhead.

## Phase 0 — Snapshot and reproduce original behavior

- Inspect the repository, existing docs, code contracts, and result JSON schema. Show an inventory of files you expect to add or touch before editing.
- Record Python, NumPy, BLAS details, CPU model if available, OS, and whether the project is installed in a venv.
- Run the original commands from the repository root:

```powershell
python -m pytest -q
python -m train.train --mode baseline --epochs 100 --seed 42
python -m train.train --mode prune --method saliency --sparsity 0.90 --epochs 100 --seed 42
```

- Compare with saved historical values without overwriting them. Record differences and explain possible nondeterminism.
- **Gate:** 115 existing tests pass (unless the checked-out repository legitimately has a different documented count), baseline training converges, and 90% pruning yields the correct integer weight count. If not, stop and diagnose.

## Phase 1 — Numerical equivalence (mandatory)

Create a separate test file such as `tests/test_inference_equivalence.py`. If necessary, add an isolated helper in `experiments/inference_validation.py`; do not modify the existing inference path.

Validate that existing dense masked inference and existing sparse-aware inference compute effectively identical outputs:

- At sparsities 0%, 50%, 75%, 90%, and 95%, for seeds 42, 43, 44, compare logits and predicted classes on exactly the same batch of held-out data.
- Include different batch sizes such as 1, 16, 128; validate both ReLU and tanh where supported.
- Check maximum absolute / relative output error via `numpy.testing.assert_allclose` with reasoned tolerances (start `rtol=1e-10`, `atol=1e-10` for float64; justify any relaxation).
- Compare classification predictions and accuracy; fail for any unexpected class mismatch. If logits are extremely close around ties, diagnose rather than silently ignore.
- Test edge cases: all-active masks, nearly all-inactive masks, a fully masked layer where API allows, nonzero biases, and uneven final minibatch.
- Ensure evaluation is pure: no gradient or parameter updates; no input mutation; repeated forward calls yield repeatable results.
- **Gate:** matching logits within justified floating-point tolerances and identical classifications for every tested configuration.

## Phase 2 — Fair-comparison audit (mandatory)

Add `tests/test_benchmark_fairness.py` or a small validation script. Do not modify existing benchmark results.

- Confirm magnitude and saliency runs at the same sparsity target use the **same data generation seed, train/test indices, initial weights, training hyperparameters, epoch count, pruning-start/end schedule, and evaluation split**.
- Confirm neither method uses test labels for computing importance scores, weight updates, tuning, or model selection.
- Check the active count equals the prescribed rounded integer budget across all sparsities, excluding biases; verify no regrowth and pruned Adam states remain zero.
- Make sure the 0% dense baseline isn't improperly counted as two independent training runs when summarizing uncertainty; explain why there are 27 distinct runs instead of 30 if the baseline is shared.
- Inspect reported per-seed test losses; if some magnitude and saliency values are mysteriously identical across every sparsity, verify that results are genuinely independently computed and not mistakenly reused.
- **Gate:** emit a concise fairness audit table with PASS/FAIL and evidence; stop and report any leakage, budget mismatch, or reused-result bug.

## Phase 3 — Optional inference optimization, isolated from core

Only after Phases 1 and 2 pass, create a standalone experimental implementation (suggested: `experiments/optimized_inference.py`) with a stable interface such as `predict_logits(model, inputs, backend="dense")`. Do not change the default training or `MLP.__call__` behavior.

Compare the following backend options where practical:

1. **Dense masked NumPy (reference):** use contiguous weight arrays with NumPy matrix multiplication; this is the correctness/performance baseline.
2. **Current sparse-coordinate path:** retain as the comparison baseline, without rewriting existing benchmark code.
3. **Optimized optional path:** try vectorized per-output sparse gathers/dot products, precomputed compressed index representations, or other **NumPy-only** approaches that avoid Python-level per-connection loops. Cache immutable inference-only representations; rebuild if weights or masks change. Measure extraction/conversion time separately. Do not assume the optimized sparse path will outperform dense BLAS for such a small model.

Critical correctness checks:

- Honor masks and biases exactly, using the correct activation between hidden layers and none on output logits.
- Match reference logits to tolerances from Phase 1, with identical predictions.
- Never mutate original weights, masks, gradients, Adam state, or the training graph.
- Clearly distinguish dense storage, number of nonzeros, index storage, and actual estimated serialized size.
- If the optimized path is slower, retain it only as documented experiment, **do not replace dense default inference**.

## Phase 4 — Honest CPU measurement

Write a separate benchmark script, suggested `experiments/optimization_benchmark.py`, that outputs new data only to `results/optimization/`.

- Measure identical input arrays, batch sizes 1, 16, 128, and multiple sparsity targets (especially 0%, 90%, 95%).
- Warm up each implementation and use `time.perf_counter_ns()` for repeated measurements. Prevent data generation/model conversion inside the timed loop unless separately labeled as end-to-end serving cost.
- Prefer median and p95 latency across multiple independently timed groups, not a single favorable number. Record iterations, run count, thread settings (BLAS threads if measurable), process conditions, versions, and CPU details.
- Make outputs numerically equivalent **before** timing and report prediction mismatches prominently.
- Report separately: dense masked forward latency, current sparse coordinate latency, new optional backend latency, conversion/build cost, theoretical MAC counts, and actual representation size where measured.
- Include a summary table with absolute times and ratios; no arbitrary '3x faster' or '90% faster' claim.
- **Success for optimization:** an optional new backend measurably improves on the existing sparse-coordinate path at relevant batch sizes **without loss of numerical correctness**. Beating dense BLAS is a stretch goal, not a required claim. If no improvement, document why and keep dense as recommended deployment path.

## Phase 5 — Robustness experiment (separate from existing case study)

Only once previous phases pass, add a **non-destructive** robustness evaluation script, e.g. `experiments/robustness.py`. Keep original `train/dataset.py` and reference benchmark untouched.

- Use higher spiral noise / harder geometry through isolated data-creation code or local parameterization without modifying the existing reference function's output for the baseline arguments.
- Compare identical conditions for magnitude and saliency across at least 5 seeds if runtime permits; include dense baseline, 50%, 75%, 90%, 95% sparsity.
- Report mean ± standard deviation, per-seed accuracy, loss, and 95% confidence intervals if appropriate; label small-sample uncertainty.
- Explain whether the two methods differ under challenging conditions. Do not conclude one is superior based on tiny differences with overlapping uncertainty.
- Report any degradation honestly: the goal is to find where pruning stops working, not to maintain perfect-looking accuracy.

## Required final deliverables

- A brief `results/optimization/VALIDATION_REPORT.md` showing original baseline reproduction, numerical equivalence, fairness, inference results, robustness findings, and any unresolved issues.
- New, isolated scripts/tests and new JSON/PNG evidence under `results/optimization/` (where applicable), **without overwriting the existing `results/` baseline files**.
- A concise changelog of each added/edited file with justification, plus the precise commands to reproduce every result.
- Clear statements: (a) how much accuracy is retained, (b) sparsity and active connection counts, (c) which backend is actually fastest on this CPU, (d) whether pruning reduced measured wall-clock time, and (e) whether magnitude vs. saliency is meaningfully different.

## Final acceptance checklist

- [ ] Original 115 tests still pass (or any legitimate pre-existing count discrepancy is explained).
- [ ] Original baseline and pruning CLI outputs are reproducible within documented tolerances.
- [ ] Dense/sparse/optimized inference logits and predictions are equivalent.
- [ ] Both methods use fair budgets, seeds, data splits, and training settings; no test leakage.
- [ ] No original source/result files accidentally overwritten; changes isolated on a branch/backup.
- [ ] New equivalence/fairness tests pass; failure cases are not hidden.
- [ ] New wall-clock benchmarks include warmup, repeat statistics, hardware/environment metadata.
- [ ] No unsupported claims of real-world inference speedup or memory savings.
- [ ] Robustness comparison has honestly reported variability and any accuracy degradation.

## Operating instructions for Antigravity

Start with **Phase 0**. Before making any edit, show the exact files you will add and current Git state. Complete phases in order; after each phase print the command executed, its real output, failures, and the next proposed change. Do not start a later phase until the previous gate is satisfied. Avoid generating oversized monolithic modules; keep each new helper narrowly scoped and readable. Prefer code clarity over optimization tricks that are hard to validate.
