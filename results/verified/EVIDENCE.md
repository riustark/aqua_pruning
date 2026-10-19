# Trained model benchmark evidence

At 89.9909% achieved weight sparsity (442 active out of 4416 weights), mean test accuracy across 3 paired seeds is 98.6111% for magnitude and 98.6111% for saliency. For the identified trained saliency snapshot, sparse coordinate inference uses 89.9909% fewer weight MACs and its median latency is 16.321 times the dense NumPy inference latency at batch 120 on the recorded environment. This timing ratio is measured on one model, not averaged across seeds.

Seeds: [42, 43, 44]; epochs: 100; hidden sizes: [64, 64].
Each nonzero method/budget pair is trained independently. Dense baselines are shared.

## Accuracy and sparsity

Mean ± population standard deviation across seeds; these are not confidence intervals.
Biases are excluded from the prunable weight budget. Integer budgets use rounding.

| Method | Target | Achieved | Active weights | Test accuracy |
|---|---:|---:|---:|---:|
| magnitude | 0.00 | 0.0000% | 4416 | 98.6111% ± 1.4164% |
| magnitude | 0.50 | 50.0000% | 2208 | 98.8889% ± 1.0393% |
| magnitude | 0.75 | 75.0000% | 1104 | 98.3333% ± 1.8002% |
| magnitude | 0.90 | 89.9909% | 442 | 98.6111% ± 1.4164% |
| magnitude | 0.95 | 94.9955% | 221 | 98.8889% ± 1.0393% |
| saliency | 0.00 | 0.0000% | 4416 | 98.6111% ± 1.4164% |
| saliency | 0.50 | 50.0000% | 2208 | 98.6111% ± 0.7857% |
| saliency | 0.75 | 75.0000% | 1104 | 99.1667% ± 1.1785% |
| saliency | 0.90 | 89.9909% | 442 | 98.6111% ± 1.4164% |
| saliency | 0.95 | 94.9955% | 221 | 98.3333% ± 1.3608% |

## Paired method comparison

Differences are saliency minus magnitude on the same seed, in percentage points.
These descriptive results do not establish general superiority or equivalence.

| Target | Paired differences | Mean difference | Sample SD |
|---|---|---:|---:|
| 0.00 | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 |
| 0.50 | +0.0000, +0.0000, -0.8333 | -0.2778 | 0.4811 |
| 0.75 | +0.8333, +1.6667, +0.0000 | +0.8333 | 0.8333 |
| 0.90 | +0.0000, +0.0000, +0.0000 | +0.0000 | 0.0000 |
| 0.95 | -0.8333, -0.8333, +0.0000 | -0.5556 | 0.4811 |

## Cost of the identified trained model

Snapshot: `models/saliency_s0p9_seed42.npz`; seed 42; method saliency.
Model SHA-256: `392738f8554ff6d08225c31225e045c5fd3e6f3f949cdb40e3a40074aefeaa75`.
Held-out accuracy: 99.1667%; loss: 0.03797718.
Batch size: 120; warmups: 20; repetitions: 200.
All paths matched training-forward logits within rtol=atol=1e-10 and produced identical class predictions.
Inference paths use the same weights and masks. Conversion is excluded from latency and recorded separately.
Dense timing uses NumPy inference without constructing an autodiff graph.

| Backend | Median µs | p95 µs | Maximum logit difference |
|---|---:|---:|---:|
| dense | 181.800 | 329.030 | 0 |
| dense_cached | 133.700 | 239.375 | 0 |
| sparse_coord | 2967.100 | 4037.530 | 1.42e-14 |
| sparse_csc_vectorized | 1507.600 | 2291.175 | 1.42e-14 |

Fastest observed backend in this measurement: `dense_cached`.
Wall-clock measurements vary with machine load and environment; exact timing reproduction is not promised.
Sparse weight MACs per sample: 442 versus 4416 for dense execution. Bias and activation operations are excluded.
MAC reduction is not a claim of wall-clock speedup or reduced dense training cost.

## Representation array payload

NumPy array payload per representation; excludes Python overhead and total engine memory. The benchmark engine retains all representations together.

| Representation | Array bytes |
|---|---:|
| dense_cached_bytes | 36376 |
| sparse_csc_bytes | 6352 |
| sparse_coord_bytes | 8120 |

## Reproduction and regression

Run from the repository root:

```powershell
python -m pytest -q
python -m experiments.benchmark --seeds 3 --epochs 100 --output-dir results/verified --reference-results results/benchmark_results.json
```

The full command reproduces the default sweep; custom runs are identified by this report's configuration.
Saved snapshots contain final inference weights, masks, configuration, and complete training histories.
They do not contain resumable optimizer state. Training and pruning formulas are unchanged.
Historical comparison: all raw accuracy/loss/count rows and summaries matched exactly.

Environment and NumPy build/thread information are recorded in the JSON cost_benchmark.environment field.
