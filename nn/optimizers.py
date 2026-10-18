"""Adam optimizer implemented from scratch with NumPy, including pruning state reset."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from engine.tensor import Tensor


class Adam:
    """Adam: gradient descent with per-parameter adaptive step sizes.

    For each parameter ``p`` with gradient ``g`` at step ``t`` (starting at 1)::

        m = beta1 * m + (1 - beta1) * g          # running mean of gradients
        v = beta2 * v + (1 - beta2) * g**2       # running mean of squared grads
        m_hat = m / (1 - beta1**t)               # bias correction (m, v start at 0)
        v_hat = v / (1 - beta2**t)
        p = p - learning_rate * m_hat / (sqrt(v_hat) + epsilon)

    The moments ``m`` and ``v`` are stored per parameter, in the same order as
    ``parameters``.

    To support pruning without weight revival:
        When a parameter connection is masked (pruned), its raw value, gradient,
        and optimizer moments (``m`` and ``v``) are zeroed via ``reset_pruned_state``.
        This guarantees that old momentum cannot reactivate pruned connections.
    """

    def __init__(
        self,
        parameters: Sequence[Tensor],
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        params = list(parameters)
        if not params:
            raise ValueError("Adam needs at least one parameter.")
        if any(not param.requires_grad for param in params):
            raise ValueError("All optimized parameters must have requires_grad=True.")
        if len({id(param) for param in params}) != len(params):
            raise ValueError("The same parameter was passed to Adam more than once.")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {beta1}, {beta2}.")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")

        self.parameters = params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_count = 0
        self.first_moments = [np.zeros_like(param.data) for param in params]
        self.second_moments = [np.zeros_like(param.data) for param in params]
        self.param_masks: dict[int, np.ndarray] = {}

    def register_mask(self, param: Tensor, mask: np.ndarray) -> None:
        """Register a boolean mask for a parameter to enforce zero states for pruned entries."""
        if mask.shape != param.shape:
            raise ValueError(f"Mask shape {mask.shape} != param shape {param.shape}.")
        if mask.dtype != np.bool_:
            raise TypeError("Mask must have boolean dtype.")
        self.param_masks[id(param)] = mask.copy()
        self.reset_pruned_state({id(param): mask})

    def reset_pruned_state(self, param_masks: dict[int, np.ndarray] | None = None) -> None:
        """Force raw values, gradients, and Adam moments to zero at pruned positions.

        Args:
            param_masks: Mapping from id(param) to boolean mask (True = active).
                If None, uses all masks currently registered in ``self.param_masks``.
        """
        masks = self.param_masks if param_masks is None else param_masks
        for index, param in enumerate(self.parameters):
            mask = masks.get(id(param))
            if mask is not None:
                self.first_moments[index] = np.where(mask, self.first_moments[index], 0.0)
                self.second_moments[index] = np.where(mask, self.second_moments[index], 0.0)
                param.data = np.where(mask, param.data, 0.0)
                param.grad = np.where(mask, param.grad, 0.0)
                # Keep registered copy updated
                self.param_masks[id(param)] = mask.copy()

    def zero_grad(self) -> None:
        """Reset every parameter's gradient to zero (call before ``backward``)."""
        for param in self.parameters:
            param.zero_grad()

    def step(self) -> None:
        """Apply one Adam update to every parameter using its current ``grad``.

        Parameter arrays are replaced (not modified in place), so any old graph
        that still references the previous values is left untouched.

        Raises:
            FloatingPointError: If a gradient or updated parameter is not finite.
        """
        self.step_count += 1
        correction1 = 1.0 - self.beta1 ** self.step_count
        correction2 = 1.0 - self.beta2 ** self.step_count

        for index, param in enumerate(self.parameters):
            grad = param.grad
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError(f"Non-finite gradient for parameter {index}.")

            first = self.beta1 * self.first_moments[index] + (1.0 - self.beta1) * grad
            second = self.beta2 * self.second_moments[index] + (1.0 - self.beta2) * grad ** 2
            self.first_moments[index] = first
            self.second_moments[index] = second

            update = self.learning_rate * (first / correction1) / (
                np.sqrt(second / correction2) + self.epsilon
            )
            new_value = param.data - update
            if not np.all(np.isfinite(new_value)):
                raise FloatingPointError(f"Non-finite value after update of parameter {index}.")

            # If parameter has a registered mask, enforce exact zero on pruned connections
            mask = self.param_masks.get(id(param))
            if mask is not None:
                new_value = np.where(mask, new_value, 0.0)
                self.first_moments[index] = np.where(mask, self.first_moments[index], 0.0)
                self.second_moments[index] = np.where(mask, self.second_moments[index], 0.0)

            param.data = new_value
