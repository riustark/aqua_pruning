"""Optimized inference implementations and model representations (Phase 3).

Provides multiple inference backends:
1. 'dense': Reference dense masked BLAS (computes inputs @ (W * mask) + b).
2. 'dense_cached': Precomputed effective weight W_eff = W * mask, evaluated via BLAS.
3. 'sparse_coord': Existing coordinate representation looping over active connections.
4. 'sparse_csc_vectorized': Column-grouped compressed representation performing
   vectorized matrix-vector dot products per output node (O(out_features) loop
   instead of O(connections) loop).

All backends are strictly read-only, non-mutating, and mathematically equivalent.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from experiments.benchmark import SparseLinearRepresentation, sparse_aware_forward
from nn.layers import MLP

BackendType = Literal["dense", "dense_cached", "sparse_coord", "sparse_csc_vectorized"]


class CachedDenseLayer:
    """Precomputed effective weight representation for fast dense BLAS evaluation."""

    def __init__(self, weight: np.ndarray, mask: np.ndarray, bias: np.ndarray) -> None:
        self.w_eff = np.ascontiguousarray(weight * mask.astype(np.float64))
        self.bias = np.ascontiguousarray(bias)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs @ self.w_eff + self.bias

    @property
    def memory_bytes(self) -> int:
        return self.w_eff.nbytes + self.bias.nbytes


class CSCVectorizedLayer:
    """Compressed Sparse Column representation with per-output-node vectorized gather.

    Groups active connections by output neuron j:
        output[:, j] = bias[j] + inputs[:, active_rows[j]] @ active_weights[j]
    Replaces thousands of Python per-connection loop iterations with exactly
    out_features vectorized BLAS operations.
    """

    def __init__(self, weight: np.ndarray, mask: np.ndarray, bias: np.ndarray) -> None:
        self.out_features = weight.shape[1]
        self.bias = bias.copy()

        # Build per-column index arrays
        self.col_rows: list[np.ndarray] = []
        self.col_weights: list[np.ndarray] = []
        self.num_active = int(mask.sum())

        for j in range(self.out_features):
            active_row_idx = np.where(mask[:, j])[0].astype(np.int32)
            self.col_rows.append(active_row_idx)
            self.col_weights.append(weight[active_row_idx, j].astype(np.float64))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        num_samples = inputs.shape[0]
        output = np.empty((num_samples, self.out_features), dtype=np.float64)

        for j in range(self.out_features):
            rows = self.col_rows[j]
            weights = self.col_weights[j]
            if len(rows) == 0:
                output[:, j] = self.bias[j]
            else:
                # Vectorized matrix-vector multiplication for all batch samples
                output[:, j] = self.bias[j] + inputs[:, rows] @ weights

        return output

    @property
    def memory_bytes(self) -> int:
        bytes_weights = sum(w.nbytes for w in self.col_weights)
        bytes_indices = sum(r.nbytes for r in self.col_rows)
        return bytes_weights + bytes_indices + self.bias.nbytes


class OptimizedInferenceEngine:
    """Standalone read-only inference engine hosting multiple backends.

    Does not modify or wrap the original training graph.
    """

    def __init__(self, model: MLP) -> None:
        self.activation = model.activation
        self.model = model

        # Conversion / extraction timings
        t0 = time.perf_counter_ns()
        self.dense_cached_layers = [
            CachedDenseLayer(l.weight.data, l.mask, l.bias.data) for l in model.layers
        ]
        self.cached_build_time_ms = (time.perf_counter_ns() - t0) / 1e6

        t1 = time.perf_counter_ns()
        self.sparse_coord_layers = [
            SparseLinearRepresentation(l.weight.data, l.mask, l.bias.data) for l in model.layers
        ]
        self.coord_build_time_ms = (time.perf_counter_ns() - t1) / 1e6

        t2 = time.perf_counter_ns()
        self.sparse_csc_layers = [
            CSCVectorizedLayer(l.weight.data, l.mask, l.bias.data) for l in model.layers
        ]
        self.csc_build_time_ms = (time.perf_counter_ns() - t2) / 1e6

    def predict_logits(self, inputs: np.ndarray, backend: BackendType = "dense") -> np.ndarray:
        """Evaluate inference using the selected backend.

        Args:
            inputs: Array of shape (N, in_features).
            backend: 'dense', 'dense_cached', 'sparse_coord', or 'sparse_csc_vectorized'.

        Returns:
            Logits array of shape (N, num_classes).
        """
        arr = np.asarray(inputs, dtype=np.float64)

        if backend == "dense":
            hidden = arr
            last_idx = len(self.model.layers) - 1
            for idx, layer in enumerate(self.model.layers):
                effective_w = layer.weight.data * layer.mask.astype(np.float64)
                hidden = hidden @ effective_w + layer.bias.data
                if idx < last_idx:
                    hidden = np.maximum(hidden, 0.0) if self.activation == "relu" else np.tanh(hidden)
            return hidden

        elif backend == "dense_cached":
            hidden = arr
            last_idx = len(self.dense_cached_layers) - 1
            for idx, layer in enumerate(self.dense_cached_layers):
                hidden = layer.forward(hidden)
                if idx < last_idx:
                    hidden = np.maximum(hidden, 0.0) if self.activation == "relu" else np.tanh(hidden)
            return hidden

        elif backend == "sparse_coord":
            return sparse_aware_forward(self.sparse_coord_layers, arr, activation=self.activation)

        elif backend == "sparse_csc_vectorized":
            hidden = arr
            last_idx = len(self.sparse_csc_layers) - 1
            for idx, layer in enumerate(self.sparse_csc_layers):
                hidden = layer.forward(hidden)
                if idx < last_idx:
                    hidden = np.maximum(hidden, 0.0) if self.activation == "relu" else np.tanh(hidden)
            return hidden

        raise ValueError(f"Unknown backend '{backend}'.")

    def memory_footprints(self) -> dict[str, int]:
        """Return NumPy array payload bytes per representation, excluding Python
        overhead, temporaries, and the total memory of this multi-backend engine.
        """
        dense_bytes = sum(l.memory_bytes for l in self.dense_cached_layers)
        csc_bytes = sum(l.memory_bytes for l in self.sparse_csc_layers)
        coord_bytes = sum(
            l.active_weights.nbytes + l.active_rows.nbytes + l.active_cols.nbytes + l.bias.nbytes
            for l in self.sparse_coord_layers
        )
        return {
            "dense_cached_bytes": dense_bytes,
            "sparse_csc_bytes": csc_bytes,
            "sparse_coord_bytes": coord_bytes,
        }
