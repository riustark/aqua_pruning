"""Core autodiff object: the ``Tensor``.

A ``Tensor`` wraps a NumPy array (``data``), stores its gradient (``grad``),
and remembers how it was produced (its ``parents`` and a ``backward_fn``).
Calling ``backward()`` on a scalar result walks this graph in reverse
topological order and accumulates ``d(result)/d(tensor)`` into every tensor
that requires gradients.

The actual math of each operation (forward value and backward rule) lives in
``engine/operations.py``. The operator methods on ``Tensor`` (``+``, ``@``,
``.relu()``...) are thin wrappers that delegate to those functions.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# A backward function receives the gradient flowing into an operation's output
# and returns one gradient array per parent, each with that parent's shape.
BackwardFn = Callable[[np.ndarray], tuple[np.ndarray, ...]]

DTYPE = np.float64


def unbroadcast(grad: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Reduce a broadcasted gradient back to the shape of the original input.

    NumPy broadcasting virtually copies an input along new or size-1 axes.
    Each copy receives its own gradient, so the true gradient of the original
    input is the *sum* over those copied axes.

    Example: ``x`` of shape ``(N, D)`` plus ``b`` of shape ``(D,)`` gives an
    output gradient of shape ``(N, D)``; the gradient for ``b`` is that array
    summed over axis 0, giving shape ``(D,)``.

    Args:
        grad: Gradient with the (broadcasted) output shape.
        target_shape: Shape of the input that was broadcast.

    Returns:
        Gradient array with exactly ``target_shape``.

    Raises:
        ValueError: If ``grad`` cannot be produced by broadcasting
            ``target_shape``.
    """
    target_shape = tuple(target_shape)
    if grad.shape == target_shape:
        return grad

    extra_dims = grad.ndim - len(target_shape)
    if extra_dims < 0:
        raise ValueError(
            f"Cannot unbroadcast gradient of shape {grad.shape} "
            f"to larger rank shape {target_shape}."
        )

    # 1) Sum away leading axes that broadcasting added in front.
    reduced = grad.sum(axis=tuple(range(extra_dims))) if extra_dims else grad

    # 2) Sum (keeping the axis) wherever the input had size 1 but was stretched.
    stretched_axes = tuple(
        axis
        for axis, size in enumerate(target_shape)
        if size == 1 and reduced.shape[axis] != 1
    )
    if stretched_axes:
        reduced = reduced.sum(axis=stretched_axes, keepdims=True)

    if reduced.shape != target_shape:
        raise ValueError(
            f"Gradient shape {grad.shape} is not a broadcast of {target_shape}."
        )
    return reduced


class Tensor:
    """A NumPy array that records operations so gradients can be computed.

    Attributes:
        data: Float64 NumPy array holding the value.
        grad: Float64 array of the same shape holding the accumulated gradient
            ``d(loss)/d(self)``. Starts at zeros.
        requires_grad: If True, ``backward`` accumulates a gradient here and
            operations using this tensor are recorded in the graph.
        parents: Tensors this one was computed from (empty for leaf tensors).
        backward_fn: Function mapping this tensor's gradient to its parents'
            gradients, or ``None`` for leaves / untracked results.
        op_name: Short label of the producing operation (for debugging).
    """

    def __init__(
        self,
        data: np.ndarray | float | int | Sequence,
        requires_grad: bool = False,
        parents: tuple["Tensor", ...] = (),
        backward_fn: BackwardFn | None = None,
        op_name: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            raise TypeError("Pass tensor.data, not a Tensor, to build a new Tensor.")
        array = np.array(data, dtype=DTYPE)
        if not np.all(np.isfinite(array)):
            raise ValueError("Tensor data must contain only finite values.")
        self.data: np.ndarray = array
        self.grad: np.ndarray = np.zeros_like(array)
        self.requires_grad: bool = bool(requires_grad)
        self.parents: tuple[Tensor, ...] = tuple(parents)
        self.backward_fn: BackwardFn | None = backward_fn
        self.op_name: str = op_name

    # ------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying array."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.data.size

    def item(self) -> float:
        """Return the value of a one-element tensor as a Python float."""
        if self.data.size != 1:
            raise ValueError(f"item() needs a single element, got shape {self.shape}.")
        return float(self.data.reshape(()))

    def zero_grad(self) -> None:
        """Reset the accumulated gradient to zeros (call before each step)."""
        self.grad = np.zeros_like(self.data)

    def __repr__(self) -> str:
        label = f", op='{self.op_name}'" if self.op_name else ""
        return (
            f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}"
            f"{label})\n{self.data}"
        )

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------
    def _topological_order(self) -> list["Tensor"]:
        """Return graph nodes so every node appears after all of its parents.

        Uses an explicit stack (not recursion) so deep graphs cannot hit
        Python's recursion limit. Only nodes that require gradients are kept.
        """
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node.parents:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Compute gradients of this tensor w.r.t. every tracked ancestor.

        Gradients are *added* to each tensor's ``.grad`` (so call
        ``zero_grad()`` between training steps). Within one call, every node's
        gradient is fully summed from all its consumers before being passed on
        to its own parents, which is why reverse topological order is used.

        Args:
            grad: Upstream gradient with this tensor's shape. May be omitted
                only when this tensor is a scalar (one element); then 1.0 is
                used, i.e. ``d(self)/d(self) = 1``.

        Raises:
            RuntimeError: If this tensor does not require gradients.
            ValueError: If ``grad`` is missing for a non-scalar or has the
                wrong shape, or an operation returns a wrongly shaped gradient.
            FloatingPointError: If any gradient becomes NaN or infinite.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad.")

        if grad is None:
            if self.data.size != 1:
                raise ValueError(
                    f"backward() without an explicit grad needs a scalar, got shape {self.shape}."
                )
            seed = np.ones_like(self.data)
        else:
            seed = np.array(grad, dtype=DTYPE)
            if seed.shape != self.shape:
                raise ValueError(f"grad shape {seed.shape} does not match tensor shape {self.shape}.")

        # Gradients for this backward pass only, keyed by node identity.
        pending: dict[int, np.ndarray] = {id(self): seed}

        for node in reversed(self._topological_order()):
            node_grad = pending.pop(id(node), None)
            if node_grad is None:
                continue  # Node does not influence the output.
            if not np.all(np.isfinite(node_grad)):
                raise FloatingPointError(
                    f"Non-finite gradient encountered at op '{node.op_name or 'leaf'}'."
                )
            node.grad = node.grad + node_grad

            if node.backward_fn is None:
                continue
            parent_grads = node.backward_fn(node_grad)
            if len(parent_grads) != len(node.parents):
                raise ValueError(
                    f"Op '{node.op_name}' returned {len(parent_grads)} grads "
                    f"for {len(node.parents)} parents."
                )
            for parent, parent_grad in zip(node.parents, parent_grads):
                if not parent.requires_grad:
                    continue
                if parent_grad.shape != parent.shape:
                    raise ValueError(
                        f"Op '{node.op_name}' produced grad shape {parent_grad.shape} "
                        f"for parent of shape {parent.shape}."
                    )
                key = id(parent)
                pending[key] = pending[key] + parent_grad if key in pending else parent_grad

    # ------------------------------------------------------------------
    # Operator wrappers (math lives in engine/operations.py)
    # ------------------------------------------------------------------
    # Imports are local to avoid a circular import: operations imports Tensor.
    def __add__(self, other: "Tensor | float") -> "Tensor":
        from engine import operations
        return operations.add(self, other)

    def __radd__(self, other: "Tensor | float") -> "Tensor":
        from engine import operations
        return operations.add(other, self)

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        from engine import operations
        return operations.subtract(self, other)

    def __rsub__(self, other: "Tensor | float") -> "Tensor":
        from engine import operations
        return operations.subtract(other, self)

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        from engine import operations
        return operations.multiply(self, other)

    def __rmul__(self, other: "Tensor | float") -> "Tensor":
        from engine import operations
        return operations.multiply(other, self)

    def __truediv__(self, other: "Tensor | float") -> "Tensor":
        from engine import operations
        return operations.divide(self, other)

    def __rtruediv__(self, other: "Tensor | float") -> "Tensor":
        from engine import operations
        return operations.divide(other, self)

    def __neg__(self) -> "Tensor":
        from engine import operations
        return operations.multiply(self, -1.0)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        from engine import operations
        return operations.matmul(self, other)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        """Sum of elements over ``axis`` (all elements when ``None``)."""
        from engine import operations
        return operations.tensor_sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        """Mean of elements over ``axis`` (all elements when ``None``)."""
        from engine import operations
        return operations.tensor_mean(self, axis=axis, keepdims=keepdims)

    def relu(self) -> "Tensor":
        """Element-wise ``max(0, x)``."""
        from engine import operations
        return operations.relu(self)

    def tanh(self) -> "Tensor":
        """Element-wise hyperbolic tangent."""
        from engine import operations
        return operations.tanh(self)
