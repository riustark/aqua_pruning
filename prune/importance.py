"""Importance scoring methods for pruning prunable weight connections.

Two scoring methods are supported:

1. **Magnitude baseline (``"magnitude"``):**
   ``score = |W_raw|``
   Assumes connections with small absolute weight contribute least to network output.
   Computationally cheap and data-free.

2. **First-order Taylor saliency (``"saliency"``):**
   ``score = |g * W_raw|``
   where ``g = dL / d(W_raw)`` is computed over training data.

### Mathematical Derivation & Limitations of Saliency:

Consider the loss function :math:`L(W)`. If connection :math:`(i, j)` is pruned,
the parameter changes by :math:`\\Delta W_{ij} = -W_{ij}`.
A first-order Taylor expansion around the current weights gives:

.. math::
    \\Delta L = L(W + \\Delta W) - L(W) \\approx \\nabla_W L \\cdot \\Delta W
    = \\sum_{ij} g_{ij} (-W_{ij}) = -\\sum_{ij} g_{ij} W_{ij}

For a single connection, the estimated absolute impact on the loss is:

.. math::
    |\\Delta L_{ij}| \\approx |g_{ij} \\cdot W_{ij}|

Connections with low saliency scores are predicted to have minimal impact on loss.

**Limitations:**
1. **First-order approximation:** Ignores second-order curvature (:math:`\\frac{1}{2} \\Delta W^T H \\Delta W`),
   which can be significant for non-linear deep networks.
2. **Gradient noise:** Stochastic mini-batch gradients are noisy. Evaluating saliency
   over full batches or multiple steps mitigates this variance.
3. **Non-interaction assumption:** Pruning multiple connections simultaneously ignores
   cross-coupling effects between weights.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from engine import operations as ops
from engine.tensor import Tensor
from nn.layers import Linear, MLP

METHODS = ("magnitude", "saliency")


def compute_magnitude_scores(layer: Linear) -> np.ndarray:
    """Calculate magnitude-based importance scores for a Linear layer.

    Pruned entries (where ``layer.mask == False``) have score 0.0.

    Args:
        layer: Linear layer.

    Returns:
        Array of absolute weights with shape ``(in_features, out_features)``.
    """
    scores = np.abs(layer.weight.data)
    # Ensure pruned connections have exact zero score
    return np.where(layer.mask, scores, 0.0)


def compute_saliency_scores(
    model: MLP, features: np.ndarray, labels: np.ndarray
) -> list[np.ndarray]:
    """Calculate gradient-informed Taylor saliency scores across all MLP layers.

    Evaluates loss on ``(features, labels)``, runs backpropagation, and computes
    ``|g * W|`` for each prunable layer.

    Args:
        model: MLP network.
        features: Input features of shape ``(N, D)``.
        labels: Class labels of shape ``(N,)``.

    Returns:
        List of score arrays matching each Linear layer in the model.
    """
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Cannot compute saliency on empty data.")
    if len(features) != len(labels):
        raise ValueError(
            f"Features length ({len(features)}) != labels length ({len(labels)})."
        )

    # Clear gradients on all model parameters before computing fresh saliency gradients
    for param in model.parameters():
        param.zero_grad()

    # Forward pass and loss
    logits = model(Tensor(features))
    loss = ops.softmax_cross_entropy(logits, labels)
    loss.backward()

    scores: list[np.ndarray] = []
    for layer in model.layers:
        # Effective gradient dL/dW_raw equals mask * dL/dW_eff
        raw_grad = layer.weight.grad
        layer_saliency = np.abs(layer.weight.data * raw_grad)
        # Ensure pruned connections have exact zero score
        layer_saliency = np.where(layer.mask, layer_saliency, 0.0)
        scores.append(layer_saliency)

    return scores


def compute_layer_scores(
    model: MLP,
    method: str,
    features: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Compute importance scores for all prunable layers in the model.

    Args:
        model: MLP network.
        method: ``"magnitude"`` or ``"saliency"``.
        features: Required when ``method == "saliency"``.
        labels: Required when ``method == "saliency"``.

    Returns:
        List of 2D score arrays matching ``model.layers``.

    Raises:
        ValueError: If ``method`` is unknown or data is missing for saliency.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown pruning method '{method}', expected one of {METHODS}.")

    if method == "magnitude":
        return [compute_magnitude_scores(layer) for layer in model.layers]

    if method == "saliency":
        if features is None or labels is None:
            raise ValueError("Saliency pruning requires features and labels to compute gradients.")
        return compute_saliency_scores(model, features, labels)

    raise ValueError(f"Unhandled method '{method}'.")
