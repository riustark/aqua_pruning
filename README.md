# AQUA Self-Pruning Neural Network

A complete, self-pruning neural network written entirely from scratch in **Python 3.10+ and NumPy**, without PyTorch, TensorFlow, JAX, or any autograd/ML libraries.

Developed for the Tredence AQUA technical challenge, this codebase demonstrates:
1. An exact, reverse-topological autodiff computational engine.
2. Training of an MLP with a custom Adam optimizer on non-linear synthetic spiral classification.
3. Mathematically principled network self-pruning using **Magnitude** and **First-Order Taylor Saliency** on a cubic schedule.
4. Hard budget enforcement reaching the exact integer target connection counts without weight revival.
5. Systems-level cost analysis comparing theoretical Multiply-Accumulate (MAC) savings against empirical BLAS latency.

---

## Quickstart & Setup

### Environment Setup (Windows PowerShell)

```powershell
# From the repository root:
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

*Note on Python Version:* Developed and verified on Python 3.14.2 (compatible with Python 3.10+).

---

## Stable Command Contracts

### 1. Automated Correctness Checks
Run the comprehensive finite-difference gradient and masking test suites:

```powershell
# Check analytical gradients against central differences (82 tests)
python -m pytest tests/test_gradients.py -v

# Check masking, gradient policy, Adam states, and learning integration (33 tests)
python -m pytest tests/test_masking.py -v

# Run the complete test suite (115 tests)
python -m pytest -q
```

### 2. Training and Self-Pruning

#### Baseline Mode (Dense Training)
```powershell
python -m train.train --mode baseline --epochs 100 --seed 42
```
*Expected Output:* Test accuracy converges from ~33% random chance to >99%.

#### Self-Pruning Mode (Taylor Saliency at 90% Target Sparsity)
```powershell
python -m train.train --mode prune --method saliency --sparsity 0.90 --epochs 100 --seed 42
```
*Expected Output:* Prunes 3,974 of 4,416 weights between epochs 10 and 60, achieving exact 89.99% sparsity (442 active connections) while maintaining >99% test accuracy.

#### Self-Pruning Mode (Magnitude Baseline at 90% Target Sparsity)
```powershell
python -m train.train --mode prune --method magnitude --sparsity 0.90 --epochs 100 --seed 42
```

### 3. Reproducible Multi-Sparsity Benchmark Sweep
Runs all 5 target sparsities (0%, 50%, 75%, 90%, 95%) across 3 fixed seeds for both methods, produces a Pareto plot, and benchmarks dense vs. sparse forward paths:

```powershell
python -m experiments.benchmark --seeds 3 --epochs 100
```

---

## Experimental Results & Evidence

### Pareto Frontier Summary (3 Fixed Seeds: 42, 43, 44)

| Method | Target Sparsity | Achieved Sparsity | Active / Total Weights | Test Accuracy (Mean ± Std) |
|---|---|---|---|---|
| **Magnitude** | 0% (Dense) | 0.00% | 4,416 / 4,416 | 98.6% ± 1.4% |
| **Magnitude** | 50% | 50.00% | 2,208 / 4,416 | 98.9% ± 1.0% |
| **Magnitude** | 75% | 75.00% | 1,104 / 4,416 | 98.3% ± 1.8% |
| **Magnitude** | 90% | 89.99% | 442 / 4,416 | 98.6% ± 1.4% |
| **Magnitude** | 95% | 95.00% | 221 / 4,416 | 98.9% ± 1.0% |
| **Saliency** | 0% (Dense) | 0.00% | 4,416 / 4,416 | 98.6% ± 1.4% |
| **Saliency** | 50% | 50.00% | 2,208 / 4,416 | 98.6% ± 0.8% |
| **Saliency** | 75% | 75.00% | 1,104 / 4,416 | 99.2% ± 1.2% |
| **Saliency** | 90% | 89.99% | 442 / 4,416 | 98.6% ± 1.4% |
| **Saliency** | 95% | 95.00% | 221 / 4,416 | 98.3% ± 1.4% |

*(Chance accuracy on 3-class spirals is 33.3%)*

### Pareto Plot
The Accuracy vs. Sparsity curve is saved to `results/pareto_plot.png`.

---

## Systems Insight: Theoretical Work vs. Empirical Wall-Clock Latency

A dense NumPy forward pass multiplying by zeros is **not** a speedup. We benchmarked a genuine sparse-aware forward path that skips zero connections against dense NumPy BLAS on a 90% pruned model:

* **Theoretical Arithmetic Work (MACs):** Reduced from 4,416 to 442 operations per sample (**90.0% reduction**).
* **Measured Dense BLAS Latency:** ~0.41 ms per batch.
* **Measured Sparse-Aware Forward Latency:** ~1.23 ms per batch (**3.0x slowdown**).

### Why does sparse forward run slower despite doing 90% fewer operations?
Dense matrix multiplication (`@`) utilizes OpenBLAS / Intel MKL with vectorized CPU SIMD registers (AVX-2/AVX-512) and sequential memory cache lines. In unstructured sparsity, skipping zero weights requires scattered coordinate lookups. In Python/NumPy, this coordinate overhead exceeds the savings from eliminated floating-point operations. Achieving actual wall-clock speedups in production requires structured/block sparsity or specialized sparse accelerators (e.g., NVIDIA Sparse Tensor Cores).

---

## Non-Negotiable Source Structure

```text
aqua-self-pruning/
├── engine/
│   ├── tensor.py          # Autodiff Tensor, graph tracking, reverse topological backward
│   └── operations.py      # Differentiable ops (+, -, *, /, @, sum, mean, relu, tanh, stable softmax CE)
├── nn/
│   ├── layers.py          # Linear layer with boolean masks, MLP composition, parameter access
│   └── optimizers.py      # From-scratch Adam with bias correction and pruning-state reset
├── prune/
│   ├── importance.py      # Magnitude and first-order Taylor saliency score calculation
│   └── scheduler.py       # Cubic polynomial sparsity schedule and exact integer selection
├── train/
│   ├── dataset.py         # Seeded synthetic interleaved spirals and deterministic mini-batch iterator
│   └── train.py           # Training coordinator, baseline and prune CLI modes
├── experiments/
│   └── benchmark.py       # Multi-sparsity sweep, Pareto plot, and sparse cost benchmark
├── tests/
│   ├── test_gradients.py  # Finite-difference gradient checks for all operations and broadcasting
│   └── test_masking.py    # Mask forward behavior, gradient policy, Adam state reset, and integration
├── results/               # Committed raw experiment JSON and Pareto plot
├── README.md              # Beginner-readable guide and reproduction contracts
├── DESIGN.md              # Mathematical derivations, pruning defense notes, and serving trade-offs
└── pyproject.toml         # Packaging metadata, dependency specifications, and pytest options
```
