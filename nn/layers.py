"""Trainable layers: ``Linear`` and a small multi-layer perceptron ``MLP``.

Parameter convention:

* ``weight`` is the **raw weight** Tensor, shape ``(in_features, out_features)``,
  the object the optimizer updates.
* ``mask`` is an explicit boolean NumPy array with the same shape;
  ``True`` = active connection, ``False`` = pruned connection.
* ``effective_weight()`` returns ``W_eff = W_raw * mask`` (with mask cast to float64),
  the weight actually used in the forward pass. Pruned connections contribute
  exactly zero to forward outputs.
* ``weight.grad`` always means ``dL/d(raw weight)``. By the chain rule through
  ``W_eff = W_raw * mask``, ``dL/dW_raw = mask * dL/dW_eff``. Masked raw weights
  receive zero gradient (correct for fixed-mask optimization; regrowth is not
  supported).
* Biases are never pruned and are excluded from the sparsity denominator.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from engine import operations as ops
from engine.tensor import Tensor

ACTIVATIONS = ("relu", "tanh")


class Linear:
    """Fully connected layer computing ``output = inputs @ effective_weight + bias``.

    Shapes: inputs ``(N, in_features)`` -> output ``(N, out_features)``.

    Initialization:
        * ``"he"`` (for ReLU): ``weight ~ Normal(0, sqrt(2 / in_features))``.
        * ``"xavier"`` (for tanh): ``weight ~ Normal(0, sqrt(2 / (in + out)))``.
        * Biases start at zero.
        * Mask starts all True (100% active, 0% pruned).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        init: str = "he",
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"Layer sizes must be positive, got {in_features} -> {out_features}."
            )
        if init == "he":
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            raise ValueError(f"Unknown init '{init}', expected 'he' or 'xavier'.")

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            rng.normal(0.0, std, size=(in_features, out_features)), requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self.mask: np.ndarray = np.ones((in_features, out_features), dtype=bool)

    def effective_weight(self) -> Tensor:
        """Effective weight ``W_eff = W_raw * mask`` used in the forward pass.

        Because ``mask`` enters as a constant float array, the backward pass
        multiplies the incoming gradient by ``mask``, guaranteeing zero gradient
        at all pruned positions: ``dL/dW_raw = mask * dL/dW_eff``.
        """
        return self.weight * self.mask.astype(np.float64)

    def apply_mask(self, new_mask: np.ndarray) -> None:
        """Install an updated boolean mask and zero raw weights at pruned positions.

        Pruning is permanent: ``new_mask`` can only turn active connections off.
        Attempting to turn a previously pruned connection back on raises an error.

        Args:
            new_mask: Boolean array of shape ``(in_features, out_features)``.

        Raises:
            TypeError: If ``new_mask`` is not a boolean NumPy array.
            ValueError: If shape differs or any pruned connection would revive.
        """
        if not isinstance(new_mask, np.ndarray) or new_mask.dtype != np.bool_:
            raise TypeError("mask must be a NumPy array with dtype bool.")
        if new_mask.shape != self.mask.shape:
            raise ValueError(f"mask shape {new_mask.shape} != weight shape {self.mask.shape}.")
        if np.any(new_mask & ~self.mask):
            raise ValueError("Regrowth is not supported: a pruned connection cannot be revived.")

        self.mask = new_mask.copy()
        # Zero raw weight values and gradients at permanently pruned positions.
        self.weight.data = np.where(self.mask, self.weight.data, 0.0)
        self.weight.grad = np.where(self.mask, self.weight.grad, 0.0)

    def num_active(self) -> int:
        """Number of active (unpruned) weight connections."""
        return int(self.mask.sum())

    def __call__(self, inputs: Tensor) -> Tensor:
        """Apply the layer to a batch of shape ``(N, in_features)``.

        Raises:
            ValueError: If ``inputs`` is not 2-D with ``in_features`` columns.
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(
                f"Linear expects input shape (N, {self.in_features}), got {inputs.shape}."
            )
        return inputs @ self.effective_weight() + self.bias

    def parameters(self) -> list[Tensor]:
        """Trainable tensors in a fixed order: ``[weight, bias]``."""
        return [self.weight, self.bias]


class MLP:
    """Stack of ``Linear`` layers with a non-linearity between them.

    ``layer_sizes=[2, 64, 64, 3]`` builds ``2->64 -> act -> 64->64 -> act -> 64->3``.
    The last layer has **no** activation: it outputs raw class scores (logits)
    of shape ``(N, num_classes)``, which go into softmax cross-entropy.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        rng: np.random.Generator,
        activation: str = "relu",
    ) -> None:
        sizes = [int(size) for size in layer_sizes]
        if len(sizes) < 3:
            raise ValueError(
                "MLP needs at least [input, hidden, output] sizes (two Linear layers)."
            )
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}', expected one of {ACTIVATIONS}.")

        init = "he" if activation == "relu" else "xavier"
        self.activation = activation
        self.layer_sizes = sizes
        self.layers = [
            Linear(sizes[i], sizes[i + 1], rng, init=init) for i in range(len(sizes) - 1)
        ]

    def __call__(self, inputs: Tensor | np.ndarray) -> Tensor:
        """Forward pass: ``(N, layer_sizes[0])`` -> logits ``(N, layer_sizes[-1])``."""
        hidden = ops.ensure_tensor(inputs)
        last_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < last_index:
                hidden = hidden.relu() if self.activation == "relu" else hidden.tanh()
        return hidden

    def parameters(self) -> list[Tensor]:
        """All trainable tensors, layer by layer: ``[W1, b1, W2, b2, ...]``."""
        return [param for layer in self.layers for param in layer.parameters()]

    def sparsity(self) -> tuple[float, int, int]:
        """Calculate network connection sparsity across all prunable weights.

        Biases are excluded from the prunable connection count.

        Returns:
            ``(sparsity, active_connections, total_prunable)``
            where ``sparsity = 1.0 - active_connections / total_prunable``.
        """
        total_prunable = sum(layer.mask.size for layer in self.layers)
        active_connections = sum(layer.num_active() for layer in self.layers)
        achieved_sparsity = 1.0 - active_connections / total_prunable
        return achieved_sparsity, active_connections, total_prunable
