# AQUA Self-Pruning Architecture & Technical Defense Notes

This document provides the mathematical derivations, architectural decisions, and systems trade-offs underlying the AQUA self-pruning implementation. Use these notes to defend the implementation in technical review.

---

## 1. Autodiff Engine Foundations

### 1.1 Reverse-Mode Automatic Differentiation & Topological Ordering
Backpropagation evaluates gradients of a scalar loss $L$ with respect to all node tensors in a Directed Acyclic Graph (DAG):

1. **Topological Sort:** The graph is traversed from loss to leaf nodes using an explicit stack-based depth-first search (`_topological_order()`). An explicit stack prevents recursion overflow on deep neural networks.
2. **Reverse Traversal & Accumulation:** Nodes are visited in reverse topological order. When an intermediate node $y$ is consumed by multiple downstream operations ($z_1 = f(y)$, $z_2 = g(y)$), reverse topological ordering ensures that all downstream gradients are accumulated into $y.\text{grad}$ before $y$ backpropagates into its parents:
   $$\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z_1} \frac{\partial z_1}{\partial y} + \frac{\partial L}{\partial z_2} \frac{\partial z_2}{\partial y}$$
3. **Graph Lifecycle:** To avoid memory leaks and stale references, graphs are rebuilt dynamically per mini-batch forward pass. `optimizer.zero_grad()` clears accumulated buffers before each backward pass.

### 1.2 Broadcasting Reduction (`unbroadcast`)
NumPy broadcasting stretches dimensions of size 1 or prepends leading singleton axes (e.g., adding bias $(D,)$ to activation $(N, D)$).

Because the forward value was copied across the broadcasted axes, the chain rule dictates that the gradient with respect to the pre-broadcast input is the **sum** of gradients across those copied axes:
$$\frac{\partial L}{\partial b_j} = \sum_{i=1}^N \frac{\partial L}{\partial Y_{ij}}$$

`engine.tensor.unbroadcast(grad, target_shape)` implements this in two steps:
1. Sums away all prepended axes where `grad.ndim > len(target_shape)`.
2. Sums across all axes where `target_shape[axis] == 1` while preserving `keepdims=True`.

### 1.3 Numerically Stable Softmax Cross-Entropy
Given raw logits $z \in \mathbb{R}^{N \times C}$ and ground-truth integer labels $y \in \{0, \dots, C-1\}^N$:

#### Numerical Overflow Guard:
Directly computing $\exp(z_{ic})$ overflows standard 64-bit floating point if $z_{ic} > 709$. We subtract the row maximum $m_i = \max_c z_{ic}$:
$$p_{ic} = \frac{\exp(z_{ic} - m_i)}{\sum_{k=1}^C \exp(z_{ik} - m_i)}$$
Because $z_{ic} - m_i \le 0$, every exponent is $\le 1.0$, preventing overflow.

#### Loss Formulation:
$$L = -\frac{1}{N} \sum_{i=1}^N \log p_{i, y_i} = -\frac{1}{N} \sum_{i=1}^N \left( (z_{i, y_i} - m_i) - \log \sum_{k=1}^C \exp(z_{ik} - m_i) \right)$$

#### Gradient Derivation:
$$\frac{\partial L}{\partial z_{ic}} = \frac{1}{N} (p_{ic} - \mathbb{I}(c = y_i))$$
This combined loss-and-softmax backward rule avoids materializing large intermediate Jacobian matrices.

---

## 2. Masked-Weight Gradient Policy & Defense

### 2.1 The Forward Formulation
For each prunable linear layer, we maintain a raw trainable parameter tensor $W_{\text{raw}} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ and an explicit boolean mask $M \in \{0, 1\}^{D_{\text{in}} \times D_{\text{out}}}$.

The effective forward weight is:
$$W_{\text{eff}} = W_{\text{raw}} \odot M$$
where $\odot$ denotes element-wise Hadamard multiplication. The layer forward pass is:
$$Y = X W_{\text{eff}} + b = X (W_{\text{raw}} \odot M) + b$$

### 2.2 Backward Pass Chain Rule
Applying the multivariable chain rule:
$$\frac{\partial L}{\partial W_{\text{raw}, ij}} = \frac{\partial L}{\partial W_{\text{eff}, ij}} \cdot \frac{\partial W_{\text{eff}, ij}}{\partial W_{\text{raw}, ij}} = \frac{\partial L}{\partial W_{\text{eff}, ij}} \cdot M_{ij}$$

In tensor notation:
$$\nabla_{W_{\text{raw}}} L = M \odot \nabla_{W_{\text{eff}}} L$$

### 2.3 Defense: Why Masked Weights Receive Zero Gradient
* **Mathematical Exactness:** When connection $(i, j)$ is pruned ($M_{ij} = 0$), $W_{\text{eff}, ij} \equiv 0$ regardless of $W_{\text{raw}, ij}$. The partial derivative $\frac{\partial W_{\text{eff}, ij}}{\partial W_{\text{raw}, ij}} = 0$. Hence, $\frac{\partial L}{\partial W_{\text{raw}, ij}} = 0$.
* **Fixed-Mask Optimization:** For any post-pruning fine-tuning phase, the surviving subnetwork must be optimized over the subspace defined by the active connections. Propagating non-zero gradients to dormant weights would violate the chain rule for the masked function.
* **Why Regrowth Would Require a Separate Surrogate:** To support regrowth without violating the chain rule, one cannot use $\nabla_{W_{\text{raw}}} L$ (which is zero). One would need a distinct dense surrogate loss gradient or an auxiliary shadow network. By adopting the standard permanent pruning paradigm, our gradient policy is strictly exact and mathematically consistent.

---

## 3. Adam-State Reset & No-Reactivation Guarantee

### 3.1 The Reactivation Vulnerability in Adaptive Optimizers
Adam maintains per-parameter running first ($m$) and second ($v$) moment vectors:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$W_t = W_{t-1} - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**The Failure Mode:**
Suppose connection $(i, j)$ is pruned at step $T$, setting $M_{ij} = 0$ and $W_{T, ij} = 0$. At step $T+1$, the gradient is zero ($g_{T+1, ij} = 0$).
However, if the optimizer state is untouched, historical momentum $m_{T, ij} \ne 0$ remains.
At step $T+1$:
$$m_{T+1, ij} = \beta_1 m_{T, ij} \ne 0$$
$$\Delta W_{T+1, ij} = -\eta \frac{\hat{m}_{T+1, ij}}{\sqrt{\hat{v}_{T+1, ij}} + \epsilon} \ne 0$$
The pruned weight immediately **reactivates** to a non-zero value, corrupting the sparsity budget.

### 3.2 The Solution & Mathematical Proof of Non-Revival
Whenever `apply_mask(new_mask)` or `prune_to_sparsity(...)` executes:
1. Raw weight values are zeroed: $W_{\text{raw}, ij} = 0$ where $M_{ij} = 0$.
2. Optimizer states are cleared via `optimizer.reset_pruned_state()`:
   $$m_{ij} \leftarrow 0, \quad v_{ij} \leftarrow 0 \quad \forall (i, j) \text{ where } M_{ij} = 0$$

#### Proof:
For all subsequent steps $t > T$:
1. $g_{t, ij} = 0$ (by chain rule through $W_{\text{eff}} = W_{\text{raw}} \cdot M$).
2. $m_{t, ij} = \beta_1 \cdot 0 + (1 - \beta_1) \cdot 0 = 0$.
3. $v_{t, ij} = \beta_2 \cdot 0 + (1 - \beta_2) \cdot 0^2 = 0$.
4. $\hat{m}_{t, ij} = 0, \quad \hat{v}_{t, ij} = 0$.
5. Update: $\Delta W_{t, ij} = -\eta \frac{0}{\sqrt{0} + \epsilon} = 0$.
6. $W_{t, ij} = W_{t-1, ij} - 0 = 0$.

Thus, $W_{ij}$ is mathematically trapped at $0.0$. Accidental reactivation is impossible.

---

## 4. Importance Scoring & First-Order Taylor Saliency

### 4.1 Derivation of Taylor Saliency
Let $L(W)$ denote the objective loss. Suppose we prune connection $(i, j)$, shifting the weight by $\Delta W_{ij} = -W_{ij}$.

Expanding $L(W + \Delta W)$ around the current weight vector $W$:
$$L(W + \Delta W) = L(W) + \nabla_W L^T \Delta W + \frac{1}{2} \Delta W^T \nabla^2_W L \Delta W + \mathcal{O}(\|\Delta W\|^3)$$

The first-order perturbation in loss resulting from zeroing connection $(i, j)$ is:
$$\Delta L_{ij} \approx \frac{\partial L}{\partial W_{ij}} (-W_{ij}) = -g_{ij} W_{ij}$$

The estimated magnitude of disruption to the loss is:
$$S_{ij} = |\Delta L_{ij}| \approx |g_{ij} \cdot W_{ij}|$$

### 4.2 Limitations & Critical Assumptions
1. **Curvature Omission (Zero Hessian):** Ignores the second-order term $\frac{1}{2} W_{ij}^2 H_{ij, ij}$. If a parameter has zero gradient at a local minimum ($g_{ij} \approx 0$), first-order saliency gives a score of 0 even if the loss surface has steep curvature.
2. **Stochastic Gradient Variance:** Single mini-batch gradients are noisy estimators of the true loss gradient. In our implementation, saliency is evaluated on the full training set prior to pruning steps to stabilize the ranking.
3. **Multi-Parameter Co-Dependency:** Taylor series assumes perturbing **one** connection at a time. Pruning thousands of connections simultaneously compounds errors because off-diagonal Hessian terms ($H_{ij, kl}$) are non-zero.

---

## 5. Cubic Sparsity Schedule & Budget Enforcement

### 5.1 Polynomial Ramp vs. One-Shot Pruning
Pruning a network in a single step (one-shot) inflicts catastrophic damage on internal feature representations. We employ an established polynomial schedule (Zhu & Gupta, 2017):
$$s(t) = s_f \left(1 - \left(1 - \frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}}\right)^3\right)$$

* **Warm-Up Phase ($t < t_{\text{start}}$):** Dense baseline training establishes stable coordinate frames.
* **Cubic Ramp ($t_{\text{start}} \le t \le t_{\text{end}}$):** Pruning starts gently, accelerates in the middle, and decelerates near the target.
* **Fine-Tuning Phase ($t > t_{\text{end}}$):** The mask is frozen, allowing remaining connections to re-converge.

### 5.2 Exact Integer Budget Formulation
* **Denominator:** Strictly the total number of prunable weight parameters $\sum_l \text{in}_l \times \text{out}_l$. Biases are excluded.
* **Connection Count:** The integer budget is enforced by `target_pruned = int(round(total * target_sparsity))`.
* **Global Ranking:** Scores are compared across all active connections in all layers, allowing the network to naturally allocate higher sparsity to less critical layers.

---

## 6. Serving at Scale: Hardware & Latency Bottlenecks

### 6.1 The BLAS vs. Sparse Matrix Paradox
Our benchmark revealed a critical systems finding:
* **Theoretical MAC reduction:** 90.0% reduction (4,416 $\to$ 442 operations).
* **Empirical Wall-Clock Latency:** 3.0x **slowdown** (0.41ms $\to$ 1.23ms).

```
Dense Matrix Multiplication (BLAS):
[  Contiguous Memory Access  ] -> [ SIMD Registers (AVX2/AVX-512) ] -> High Throughput

Unstructured Sparse Forward (Coordinate / CSR):
[ Scattered Index Lookups   ] -> [ Cache Misses / Branching       ] -> Memory Stall
```

### 6.2 Industrial Deployment Recommendations
To translate theoretical connection pruning into wall-clock inference speedups in production:
1. **2:4 Semi-Structured Sparsity:** Modern hardware (NVIDIA Ampere/Hopper Tensor Cores) accelerates 2:4 sparsity (exactly 2 values out of every 4 are zero) with specialized hardware instructions, providing genuine 2x speedups.
2. **Block / Channel Pruning:** Pruning entire columns or hidden units allows removing rows/columns from the dense weight matrices entirely, reducing matrix dimensions while remaining inside fast BLAS routines.
3. **Sparse-Trained Compilers:** Deployment via TVM, TensorRT, or ONNX Runtime with vendor-optimized sparse GEMM kernels.
