"""Differentiable operations used by ``Tensor``.

Every function here does two things:

1. **Forward:** compute the output value with NumPy.
2. **Backward rule:** define a closure that, given ``dL/d(output)``, returns
   ``dL/d(input)`` for each input, *already reduced to the input's exact
   shape* (see ``unbroadcast``).

Notation in docstrings: ``g`` is the upstream gradient ``dL/d(out)``.
Python numbers are accepted wherever a ``Tensor`` is expected and are treated
as constants (no gradient).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from engine.tensor import BackwardFn, Tensor, unbroadcast

Operand = Tensor | float | int | np.ndarray
Axis = int | tuple[int, ...] | None


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def ensure_tensor(value: Operand) -> Tensor:
    """Return ``value`` as a ``Tensor``; non-tensors become constants."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=False)


def _make_result(
    data: np.ndarray,
    parents: tuple[Tensor, ...],
    backward_fn: BackwardFn,
    op_name: str,
) -> Tensor:
    """Wrap ``data`` in a Tensor, linking it into the graph only if needed.

    If no parent requires a gradient, the result is a plain constant and the
    graph is not stored, which saves memory.
    """
    track = any(parent.requires_grad for parent in parents)
    return Tensor(
        data,
        requires_grad=track,
        parents=parents if track else (),
        backward_fn=backward_fn if track else None,
        op_name=op_name,
    )


def _check_broadcastable(left: Tensor, right: Tensor, op_name: str) -> None:
    """Raise a clear error if two shapes cannot be broadcast together."""
    try:
        np.broadcast_shapes(left.shape, right.shape)
    except ValueError as error:
        raise ValueError(
            f"{op_name}: shapes {left.shape} and {right.shape} are not broadcastable."
        ) from error


def _normalize_axes(axis: Axis, ndim: int) -> tuple[int, ...]:
    """Turn ``axis`` (None, int, or tuple, possibly negative) into sorted positive axes."""
    if axis is None:
        return tuple(range(ndim))
    raw_axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = []
    for single_axis in raw_axes:
        if not isinstance(single_axis, (int, np.integer)):
            raise TypeError(f"Axis must be an int, got {type(single_axis).__name__}.")
        if not -ndim <= single_axis < ndim:
            raise ValueError(f"Axis {single_axis} is out of range for {ndim}-D tensor.")
        axes.append(int(single_axis) % ndim)
    if len(set(axes)) != len(axes):
        raise ValueError(f"Repeated axis in {axis}.")
    return tuple(sorted(axes))


# ----------------------------------------------------------------------
# Element-wise binary operations (support NumPy broadcasting)
# ----------------------------------------------------------------------
def add(left: Operand, right: Operand) -> Tensor:
    """``out = left + right``.

    Backward: ``dL/dleft = g``, ``dL/dright = g`` (each unbroadcast).
    """
    left, right = ensure_tensor(left), ensure_tensor(right)
    _check_broadcastable(left, right, "add")

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        return unbroadcast(grad, left.shape), unbroadcast(grad, right.shape)

    return _make_result(left.data + right.data, (left, right), backward_fn, "add")


def subtract(left: Operand, right: Operand) -> Tensor:
    """``out = left - right``.

    Backward: ``dL/dleft = g``, ``dL/dright = -g`` (each unbroadcast).
    """
    left, right = ensure_tensor(left), ensure_tensor(right)
    _check_broadcastable(left, right, "subtract")

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        return unbroadcast(grad, left.shape), unbroadcast(-grad, right.shape)

    return _make_result(left.data - right.data, (left, right), backward_fn, "subtract")


def multiply(left: Operand, right: Operand) -> Tensor:
    """``out = left * right`` (element-wise).

    Backward: ``dL/dleft = g * right``, ``dL/dright = g * left``
    (each unbroadcast).
    """
    left, right = ensure_tensor(left), ensure_tensor(right)
    _check_broadcastable(left, right, "multiply")

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            unbroadcast(grad * right.data, left.shape),
            unbroadcast(grad * left.data, right.shape),
        )

    return _make_result(left.data * right.data, (left, right), backward_fn, "multiply")


def divide(left: Operand, right: Operand) -> Tensor:
    """``out = left / right`` (element-wise).

    Backward: ``dL/dleft = g / right``,
    ``dL/dright = -g * left / right**2`` (each unbroadcast).

    Raises:
        ZeroDivisionError: If any element of ``right`` is exactly zero.
    """
    left, right = ensure_tensor(left), ensure_tensor(right)
    _check_broadcastable(left, right, "divide")
    if np.any(right.data == 0.0):
        raise ZeroDivisionError("divide: denominator contains zero.")

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            unbroadcast(grad / right.data, left.shape),
            unbroadcast(-grad * left.data / (right.data ** 2), right.shape),
        )

    return _make_result(left.data / right.data, (left, right), backward_fn, "divide")


# ----------------------------------------------------------------------
# Matrix multiplication
# ----------------------------------------------------------------------
def matmul(left: Operand, right: Operand) -> Tensor:
    """``out = left @ right`` for 2-D inputs ``(N, K) @ (K, M) -> (N, M)``.

    Backward: ``dL/dleft = g @ right.T`` with shape ``(N, K)``,
    ``dL/dright = left.T @ g`` with shape ``(K, M)``.

    Raises:
        ValueError: If inputs are not 2-D or inner dimensions differ.
    """
    left, right = ensure_tensor(left), ensure_tensor(right)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"matmul expects 2-D inputs, got {left.shape} and {right.shape}.")
    if left.shape[1] != right.shape[0]:
        raise ValueError(f"matmul: inner dimensions differ, {left.shape} @ {right.shape}.")

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        return grad @ right.data.T, left.data.T @ grad

    return _make_result(left.data @ right.data, (left, right), backward_fn, "matmul")


# ----------------------------------------------------------------------
# Reductions
# ----------------------------------------------------------------------
def _expand_reduced_grad(
    grad: np.ndarray, input_shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool
) -> np.ndarray:
    """Spread a reduced gradient back over the input shape (copy along ``axes``)."""
    if not keepdims:
        grad = np.expand_dims(grad, axes)
    return np.broadcast_to(grad, input_shape).copy()


def tensor_sum(tensor: Operand, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Sum over ``axis`` (all elements when ``None``).

    Backward: every input element contributed with weight 1, so
    ``dL/dinput`` is ``g`` copied back along the summed axes.
    """
    tensor = ensure_tensor(tensor)
    axes = _normalize_axes(axis, tensor.ndim)

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        return (_expand_reduced_grad(grad, tensor.shape, axes, keepdims),)

    data = np.sum(tensor.data, axis=axes, keepdims=keepdims)
    return _make_result(data, (tensor,), backward_fn, "sum")


def tensor_mean(tensor: Operand, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Mean over ``axis`` (all elements when ``None``).

    Backward: each of the ``count`` averaged elements has weight
    ``1 / count``, so ``dL/dinput = g / count`` copied along the axes.

    Raises:
        ValueError: If the reduced region is empty.
    """
    tensor = ensure_tensor(tensor)
    axes = _normalize_axes(axis, tensor.ndim)
    count = int(np.prod([tensor.shape[a] for a in axes])) if axes else 1
    if count == 0:
        raise ValueError("mean of an empty region is undefined.")

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        return (_expand_reduced_grad(grad, tensor.shape, axes, keepdims) / count,)

    data = np.sum(tensor.data, axis=axes, keepdims=keepdims) / count
    return _make_result(data, (tensor,), backward_fn, "mean")


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------
def relu(tensor: Operand) -> Tensor:
    """``out = max(0, x)`` element-wise.

    Backward: ``dL/dx = g * (x > 0)``. At exactly ``x == 0`` (the kink) we
    use gradient 0, a standard sub-gradient choice.
    """
    tensor = ensure_tensor(tensor)
    positive = tensor.data > 0.0

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        return (grad * positive,)

    return _make_result(np.where(positive, tensor.data, 0.0), (tensor,), backward_fn, "relu")


def tanh(tensor: Operand) -> Tensor:
    """``out = tanh(x)`` element-wise.

    Backward: ``dL/dx = g * (1 - out**2)``, reusing the forward output.
    """
    tensor = ensure_tensor(tensor)
    output = np.tanh(tensor.data)

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        return (grad * (1.0 - output ** 2),)

    return _make_result(output, (tensor,), backward_fn, "tanh")


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def softmax_cross_entropy(logits: Tensor, labels: Sequence[int] | np.ndarray) -> Tensor:
    """Mean softmax cross-entropy loss, computed in a numerically stable way.

    Forward, per row ``i`` with true class ``y_i``::

        shifted   = z_i - max(z_i)                  # avoids exp overflow
        log_p_i   = shifted - log(sum(exp(shifted)))
        loss      = -mean_i(log_p_i[y_i])

    Subtracting the row maximum does not change softmax but keeps every
    ``exp`` argument <= 0, so it never overflows.

    Backward: ``dL/dz = g * (softmax(z) - one_hot(y)) / N``.

    Args:
        logits: Tensor of shape ``(N, C)`` with raw class scores.
        labels: Integer class indices of shape ``(N,)`` in ``[0, C)``.

    Returns:
        Scalar Tensor (shape ``()``) holding the mean loss.

    Raises:
        TypeError: If labels are not integers.
        ValueError: If shapes are wrong or labels are out of range.
    """
    logits = ensure_tensor(logits)
    label_array = np.asarray(labels)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, C), got shape {logits.shape}.")
    num_samples, num_classes = logits.shape
    if num_samples == 0 or num_classes == 0:
        raise ValueError("logits must have at least one sample and one class.")
    if not np.issubdtype(label_array.dtype, np.integer):
        raise TypeError(f"labels must be integers, got dtype {label_array.dtype}.")
    if label_array.shape != (num_samples,):
        raise ValueError(f"labels must have shape ({num_samples},), got {label_array.shape}.")
    if np.any(label_array < 0) or np.any(label_array >= num_classes):
        raise ValueError(f"labels must lie in [0, {num_classes}).")

    rows = np.arange(num_samples)
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp
    loss_value = -log_probs[rows, label_array].mean()

    def backward_fn(grad: np.ndarray) -> tuple[np.ndarray, ...]:
        grad_logits = np.exp(log_probs)  # softmax probabilities
        grad_logits[rows, label_array] -= 1.0
        return (grad * grad_logits / num_samples,)

    return _make_result(np.asarray(loss_value), (logits,), backward_fn, "softmax_cross_entropy")
