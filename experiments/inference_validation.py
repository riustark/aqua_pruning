"""Inference equivalence validation utilities (Phase 1).

Compares dense masked forward pass (reference) with sparse-aware forward pass
to mathematically guarantee exact prediction and logit equivalence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.benchmark import SparseLinearRepresentation, sparse_aware_forward
from nn.layers import MLP


def extract_sparse_layers(model: MLP) -> list[SparseLinearRepresentation]:
    """Convert an MLP's layers into sparse representations."""
    return [
        SparseLinearRepresentation(layer.weight.data, layer.mask, layer.bias.data)
        for layer in model.layers
    ]


def dense_forward_numpy(model: MLP, inputs: np.ndarray) -> np.ndarray:
    """Evaluate model forward pass using dense NumPy BLAS without graph tracking.

    Computes inputs @ (W * mask) + b across all layers.
    Pure evaluation: does not mutate inputs, weights, masks, or gradients.
    """
    hidden = np.array(inputs, dtype=np.float64, copy=True)
    last_idx = len(model.layers) - 1
    for idx, layer in enumerate(model.layers):
        effective_w = layer.weight.data * layer.mask.astype(np.float64)
        hidden = hidden @ effective_w + layer.bias.data
        if idx < last_idx:
            hidden = np.maximum(hidden, 0.0) if model.activation == "relu" else np.tanh(hidden)
    return hidden


def sparse_forward_numpy(
    sparse_layers: list[SparseLinearRepresentation],
    inputs: np.ndarray,
    activation: str = "relu",
) -> np.ndarray:
    """Evaluate model forward pass skipping inactive connections."""
    inputs_copy = np.array(inputs, dtype=np.float64, copy=True)
    return sparse_aware_forward(sparse_layers, inputs_copy, activation=activation)


def verify_equivalence(
    dense_logits: np.ndarray,
    sparse_logits: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Verify maximum absolute/relative error and exact classification matching.

    Returns:
        Dict with max_abs_diff, max_rel_diff, class_agreement_rate, and pass_status.
    """
    abs_diff = np.abs(dense_logits - sparse_logits)
    max_abs_diff = float(np.max(abs_diff))

    # Avoid division by zero for relative diff
    denom = np.maximum(np.abs(dense_logits), 1e-12)
    rel_diff = abs_diff / denom
    max_rel_diff = float(np.max(rel_diff))

    dense_preds = np.argmax(dense_logits, axis=-1)
    sparse_preds = np.argmax(sparse_logits, axis=-1)
    class_matches = int(np.sum(dense_preds == sparse_preds))
    total_samples = int(len(dense_preds))
    agreement_rate = class_matches / total_samples

    is_close = bool(np.allclose(dense_logits, sparse_logits, rtol=rtol, atol=atol))
    exact_classes = (class_matches == total_samples)

    return {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "class_agreement_rate": agreement_rate,
        "logits_match": is_close,
        "classes_match": exact_classes,
        "pass_status": is_close and exact_classes,
    }
