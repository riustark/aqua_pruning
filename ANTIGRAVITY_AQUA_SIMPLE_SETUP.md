# AQUA Self-Pruning Neural Network — Antigravity Build Specification

> **Goal:** Build the Tredence AQUA technical challenge in **Python 3.10+ and NumPy**, from scratch, using the **exact small architecture below**. Favor correctness, understandable and reusable code, reproducible evidence, and the ability to explain every line in a technical interview.
>
> **Important:** No prompt can guarantee zero bugs. Implement in small stages, run automated tests and numerical checks, inspect failures, and fix them before proceeding. Never claim a test passed unless it was actually executed.

## 1. Non-negotiable file structure

```text
aqua-self-pruning/
├── engine/
│   ├── tensor.py
│   └── operations.py
├── nn/
│   ├── layers.py
│   └── optimizers.py
├── prune/
│   ├── importance.py
│   └── scheduler.py
├── train/
│   ├── dataset.py
│   └── train.py
├── experiments/
│   └── benchmark.py
├── tests/
│   ├── test_gradients.py
│   └── test_masking.py
├── results/
├── README.md
├── DESIGN.md
└── pyproject.toml
```

- Keep this exact source-file layout. Do **not** add other Python modules, CLI scripts, config files, or unnecessary architecture layers.
- `results/` is for experiment output: CSV/JSON data and plots; outputs are not additional source modules.
- Use Python namespace-package imports from the repository root (`from engine.tensor import Tensor`, etc.) without creating extra `__init__.py` files. Invoke commands from the project root with `python -m ...`.
- Use `pyproject.toml` for dependencies and tool settings only. Use `argparse` in existing entry-point modules for options.
- No unsupported frameworks in the implementation: **No PyTorch, TensorFlow, JAX, Keras, HuggingFace, or autodiff/pruning/neural-network libraries.** NumPy is allowed; Matplotlib only for plots; pytest for testing. Synthetic data avoids dataset dependencies. Do not use scikit-learn for training or metrics.

## 2. File-by-file responsibility

| File | Single main responsibility | Expected public pieces |
|---|---|---|
| `engine/tensor.py` | Core `Tensor` object, graph tracking, gradient storage, reverse topological `backward`, broadcasting gradient reduction | `Tensor`, `unbroadcast` helper |
| `engine/operations.py` | Differentiable math operations and stable classification loss | add/subtract/multiply/divide/matmul, sum/mean, ReLU, tanh or sigmoid, stable softmax cross-entropy (operations can be methods on `Tensor`, but keep shared operation logic here) |
| `nn/layers.py` | Trainable `Linear` and MLP composition, parameter access, forward pass | `Linear`, `MLP`, `parameters()` |
| `nn/optimizers.py` | Implement Adam updates and optimizer-state management | `Adam`, `step()`, `zero_grad()`, pruning-state reset or masked-state enforcement |
| `prune/importance.py` | Calculate importance scores for pruning, independently of the schedule | magnitude baseline; gradient-informed saliency |
| `prune/scheduler.py` | Target sparsity over time; select/prune connections by masks; enforce hard budget | `target_sparsity_at_step`, `prune_to_sparsity` |
| `train/dataset.py` | Deterministic synthetic classification data and mini-batches | spiral generator, train/test split, batch iterator |
| `train/train.py` | Coordinate forward, loss, backward, optimizer, scheduled pruning, metrics, and CLI | baseline and prune modes |
| `experiments/benchmark.py` | Reproducible multi-sparsity experiments, baselines, plots, and actual cost evidence | CLI for sweep, raw results, Pareto plot, sparse-aware cost measurement |
| `tests/test_gradients.py` | Finite-difference gradient checking of **all** implemented operations, including broadcasting | pytest suite |
| `tests/test_masking.py` | Tests for pruning, mask forward behavior, masked gradients, Adam states, numerical stability, and learning integration | pytest suite |
| `README.md` | Setup, commands, results, caveats, how to reproduce | beginner-readable guide |
| `DESIGN.md` | Derivations, gradient/masking policy, pruning rationale, trade-offs, bottlenecks, serving at scale | defense notes |
| `pyproject.toml` | Version constraints, dependencies, basic tool settings | NumPy, Matplotlib, pytest, packaging optional |

Keep helpers close to where they are used. Avoid inheritance hierarchies, registries, factories, decorators, complex configuration layers, and premature optimization.

## 3. Code quality and syntax standards

1. Python 3.10+; 4 spaces; PEP 8 names (`snake_case` functions, `PascalCase` classes); descriptive variables; no ambiguous single-letter identifiers except familiar mathematical loop indices.
2. Use type hints for public functions and methods. Write concise docstrings specifying arguments, returns, array shapes when important, and exceptions/assumptions.
3. Separate operations with mathematical effects from orchestration and I/O. Do not print from mathematical core modules; use the training and benchmark entry points for output.
4. Prefer plain NumPy arrays with floating types for tensor values and masks with a boolean dtype. Make shape, dtype, and numeric expectations explicit. Use explicit `np.random.default_rng(seed)` rather than hidden global randomness.
5. Never silently ignore incompatible shapes or bad inputs; raise clear errors. Guard division/log computations and use numerically stable softmax cross-entropy (subtract row max; log-sum-exp or equivalent).
6. Avoid modifying tensors in place while a graph that depends on them is awaiting backpropagation. Rebuild the graph per training step, and clear per-step gradient buffers before backward.
7. Each math operation must document its backward rule and handle broadcasting correctly. Never write gradients as `np.sum` over arbitrary axes without verifying output shape matches the parent.
8. Store trainable parameters in one consistent representation, and define one clear convention for *raw weight*, *effective masked weight*, and *gradient*.
9. No copied benchmark results or fabricated timings; fixed seeds, recorded settings, machine/environment details where relevant, and committed raw experiment outputs.
10. Do not mark any code complete until it imports, tests pass, and at least one documented command executes successfully.

## 4. Mathematical and architectural contract

### Autodiff
- `Tensor` wraps `numpy.ndarray`, owns `grad`, tracks parent tensors and a backward closure/function for each differentiable operation.
- Backpropagate from a scalar loss in **reverse topological order** so shared nodes accumulate gradient contributions correctly.
- Implement add, subtract, multiply, divide, matmul, sum, mean, ReLU, and one differentiable nonlinear activation (tanh recommended). Also stable softmax with cross-entropy.
- Handle scalar and array inputs; reduce broadcasted gradients back to the exact original parent shape. Example: `(N, D) + (D,)` produces a bias gradient of shape `(D,)`.
- Finite-difference gradient check with central differences: `(f(x + eps) - f(x - eps)) / (2 * eps)`; use small arrays, stable eps/tolerances, and avoid ReLU kink points for numeric checking.
- Test repeated use of a parent (e.g., `x * x`), branching and accumulation, and at least one matrix operation.

### Network and optimizer
- `Linear` applies `input @ weight + bias`; MLP composes at least two linear layers and nonlinear activation.
- Use a justified initialization (e.g., He for ReLU, Xavier for tanh), explain output shapes.
- Implement Adam from scratch with first/second moments, bias correction, epsilon and parameter updates; reset per-step gradients.
- Train with mini-batches on a small synthetic non-linear classification task (spirals is suitable), using a deterministic data split and recorded losses/accuracies.
- Check losses and gradients for finite values (`np.isfinite`) and fail loudly if instability occurs.

### Mandatory pruning correctness (pay particular attention)
- Maintain an **explicit boolean mask** for each prunable weight tensor. The effective forward weight is `W_eff = W_raw * mask` (or a mathematically equivalent masking operation). Pruned connections must contribute **exactly zero** to forward outputs.
- Define and document the chosen masked-weight gradient policy. With an explicit fixed mask and `W_eff = W_raw * mask`, ordinary chain rule gives `dL/dW_raw = mask * dL/dW_eff`; hence masked raw weights receive zero gradient. This is correct for **fixed-mask optimization**; a distinct dense/surrogate gradient would be needed to score dormant connections for regrowth. Do not conflate these gradients.
- If pruning is permanent, make it explicit and do not claim regrowth. For permanently masked entries, force the raw weight, gradient, and optimizer's Adam moments to zero (or otherwise prove equivalent behavior) after pruning and across subsequent steps so old moments cannot reactivate weights.
- If you opt to implement regrowth (optional), define how dormant connections are scored without relying on zero masked raw gradients, and specify/reset/reinitialize Adam moments upon revival. Add an explicit regression test. Simpler default: **no regrowth**.
- For pruning decisions, compare scores only among currently active prunable connections. Keep biases unpruned unless explicitly justified. State exactly which weights contribute to the sparsity denominator.
- Magnitude-only score `abs(W)` is the required baseline. For gradient-informed pruning, use a principled approximation such as first-order Taylor saliency `abs(W * g)` using the gradient of the *effective active weight* and explain the approximation `ΔL ≈ -gW` and its limitations (first-order, noise, per-step variation).
- Define a gradual schedule (e.g., cubic sparsity ramp) that goes from initial to target sparsity. For each requested target, reach the exact feasible integer connection count, clearly documenting rounding.
- Test for no pruned weight revival across repeated Adam steps, mask dtype and shape, consistent count/sparsity, correct masked loss/gradient behavior, and state-reset behavior. If you do not support revival, say so rather than faking a revival test.

### Evidence and honest performance
- Sweep target sparsities **0%, 50%, 75%, 90%, 95%** (adjust only if infeasible, disclose changes); run both magnitude and gradient-informed pruning at matching budgets.
- Save raw per-seed results and plots under `results/`; report final accuracy, achieved sparsity, number of active connections, and variance across **at least 3 fixed seeds** if runtime permits. Explain limitations when fewer seeds are run.
- Show accuracy vs achieved sparsity for both methods. Make one falsifiable quantitative claim supported by saved observations.
- A dense NumPy forward pass multiplying by zero is **not** a speedup. Count the theoretical dense and active multiply-add work and/or implement a **sparse-aware forward path** in `experiments/benchmark.py` that genuinely skips inactive connections and time it fairly. Explain that sparse Python loops can be slower than optimized dense BLAS despite fewer operations. Clearly distinguish theoretical operations from measured wall-clock latency.
- Report honest outcomes even if gradient-informed pruning loses to magnitude pruning or high sparsity hurts accuracy.

## 5. Exact development stages — do not skip

### Milestone 1: Autodiff engine
Implement `engine/tensor.py`, `engine/operations.py`, and `tests/test_gradients.py` only. Verify imports, scalar and broadcast cases, accumulation, all required operations, and numeric gradient checks. Run tests and show real terminal output. Explain Tensor, graph, and gradient in beginner terms. **Stop and wait for approval**.

### Milestone 2: Neural network learning
Implement `nn/layers.py`, `nn/optimizers.py`, `train/dataset.py`, baseline training inside `train/train.py`, and appropriate tests in the existing `tests/test_masking.py` if needed. Show training loss decreases, accuracy is above chance, finite values, and fixed-seed reproducibility. **Stop and wait for approval**.

### Milestone 3: Correct self-pruning
Implement `prune/importance.py`, `prune/scheduler.py`, pruning support in layers/optimizer/training and mask tests in `tests/test_masking.py`. Demonstrate exact mask zeros, correct chain-rule gradient policy, no accidental Adam reactivation, and achieved target sparsity. **Stop and wait for approval**.

### Milestone 4: Evidence and comparison
Implement `experiments/benchmark.py` using existing train/evaluation functions. Compare pruning methods, sweep budgets, save plot and raw results, demonstrate actual work saved without false speed claims. **Stop and wait for approval**.

### Milestone 5: Documentation and clean-room verification
Finalize `README.md`, `DESIGN.md`, and `pyproject.toml`; validate the entire documented workflow from repository root; show tests and generated artifacts. Make incremental Git commits with meaningful messages. Explain how to defend the masking rule, Adam-state reset, saliency approximation, results, and serving trade-offs.

## 6. Commands the finished project must support

Use these as required stable CLI contracts; implement the flags inside the indicated existing modules.

```powershell
# Windows PowerShell, run from project root
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .

# Automated correctness checks
python -m pytest tests/test_gradients.py -v
python -m pytest tests/test_masking.py -v
python -m pytest -q

# Ordinary training and self-pruning (implement in train/train.py)
python -m train.train --mode baseline --epochs 100 --seed 42
python -m train.train --mode prune --method saliency --sparsity 0.90 --epochs 100 --seed 42

# Reproducible Pareto experiments (implement in experiments/benchmark.py)
python -m experiments.benchmark --seeds 3 --epochs 100
```

If local Python is a newer compatible version instead of 3.10, use `py -3.14` or `py -m venv .venv`, and record the tested version. Do not claim compatibility without testing. `pyproject.toml` must contain valid editable-install metadata so `pip install -e .` actually succeeds with namespace packages; alternatively document a simpler install command, but never leave a broken advertised command.

## 7. Acceptance checklist

- [ ] Exact agreed source-file structure preserved; no extra Python modules.
- [ ] No prohibited ML/autodiff frameworks.
- [ ] `Tensor.backward` has correct topological ordering and gradient accumulation.
- [ ] Every implemented operation passes finite-difference checks, including broadcasting.
- [ ] Stable loss and deterministic baseline learning demonstrated.
- [ ] Adam implementation verified and masked optimizer states cannot undo pruning.
- [ ] Gradient policy for fixed masked weights is derived and documented.
- [ ] Saliency and magnitude methods share an identical target budget.
- [ ] Sparsity targets achieved and measured accurately.
- [ ] Raw results, Pareto plot, and honest compute-cost evidence are committed.
- [ ] README contains verified reproduction commands and DESIGN explains trade-offs.
- [ ] All tests are executed, failures addressed, no fabricated numbers, no unverified claims.

## 8. Instructions for Antigravity agent

1. First read this complete Markdown file and briefly restate the architecture and Milestone 1 plan.
2. **Do not generate the full project in one pass.** Work on Milestone 1 only until I approve the next milestone.
3. Keep code straightforward enough for an intermediate Python learner to read. After each file, explain: **what it does, its inputs/outputs, which file calls it, and why it exists**.
4. Write runnable code, not pseudocode, and write tests before declaring completion. Run tests in the project environment, disclose actual results, and correct errors; aim for bug-free behavior but never assert a guarantee.
5. Avoid new files and dependencies except the exact original structure and necessary test/experiment outputs. Preserve public APIs and reuse existing functions rather than duplicating calculations.
6. Never silently weaken challenge criteria or invent experimental outcomes to meet targets.
7. Before moving to the next milestone, provide a short summary of implemented behavior, actual test commands and outcomes, limitations, and the next file to implement. **Wait for my explicit approval.**
