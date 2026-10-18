"""Finite-difference gradient checks for the autodiff engine.

Run from the project root:

    python -m pytest tests/test_gradients.py -v

Strategy: for an operation ``out = f(inputs)``, pick a random upstream
gradient ``upstream`` and define the scalar ``L = sum(out * upstream)``. The
analytic gradient comes from ``out.backward(upstream)``; the numeric one from
central differences ``(L(x + eps) - L(x - eps)) / (2 * eps)``. Using a random
upstream (not all ones) catches transposition and broadcasting bugs.
"""

from __future__ import annotations

import sys
from typing import Callable

import numpy as np
import pytest

from engine import operations as ops
from engine.tensor import Tensor, unbroadcast

EPS = 1e-6
RTOL = 1e-5
ATOL = 1e-7


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def numeric_gradient(
    forward: Callable[[list[np.ndarray]], np.ndarray],
    arrays: list[np.ndarray],
    index: int,
    upstream: np.ndarray,
) -> np.ndarray:
    """Central-difference gradient of ``sum(forward(arrays) * upstream)`` w.r.t. ``arrays[index]``."""
    target = arrays[index]
    grad = np.zeros_like(target)
    for position in np.ndindex(target.shape):
        original = target[position]
        target[position] = original + EPS
        loss_plus = np.sum(forward(arrays) * upstream)
        target[position] = original - EPS
        loss_minus = np.sum(forward(arrays) * upstream)
        target[position] = original
        grad[position] = (loss_plus - loss_minus) / (2.0 * EPS)
    return grad


def check_gradients(
    build: Callable[[list[Tensor]], Tensor],
    arrays: list[np.ndarray],
    seed: int = 0,
) -> None:
    """Assert analytic gradients of ``build`` match numeric ones for every input."""
    rng = np.random.default_rng(seed)
    arrays = [np.array(a, dtype=np.float64) for a in arrays]

    tensors = [Tensor(a, requires_grad=True) for a in arrays]
    output = build(tensors)
    upstream = rng.normal(size=output.shape)
    output.backward(upstream)

    def forward(raw: list[np.ndarray]) -> np.ndarray:
        return build([Tensor(a) for a in raw]).data

    for index, tensor in enumerate(tensors):
        expected = numeric_gradient(forward, arrays, index, upstream)
        assert tensor.grad.shape == tensor.shape, "gradient shape must match input shape"
        np.testing.assert_allclose(tensor.grad, expected, rtol=RTOL, atol=ATOL)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


# ----------------------------------------------------------------------
# Tensor basics
# ----------------------------------------------------------------------
def test_tensor_construction_defaults() -> None:
    tensor = Tensor([[1, 2], [3, 4]], requires_grad=True)
    assert tensor.data.dtype == np.float64
    assert tensor.shape == (2, 2)
    assert np.array_equal(tensor.grad, np.zeros((2, 2)))
    assert tensor.parents == () and tensor.backward_fn is None


def test_tensor_rejects_non_finite_and_nested_tensor() -> None:
    with pytest.raises(ValueError):
        Tensor([1.0, np.nan])
    with pytest.raises(TypeError):
        Tensor(Tensor(1.0))


def test_item_and_zero_grad() -> None:
    tensor = Tensor(3.0, requires_grad=True)
    (tensor * tensor).backward()
    assert tensor.item() == 3.0
    assert tensor.grad == pytest.approx(6.0)
    tensor.zero_grad()
    assert tensor.grad == 0.0
    with pytest.raises(ValueError):
        Tensor([1.0, 2.0]).item()


def test_constants_do_not_build_graph() -> None:
    result = Tensor([1.0, 2.0]) * 3.0 + 1.0
    assert not result.requires_grad
    assert result.parents == () and result.backward_fn is None
    with pytest.raises(RuntimeError):
        result.sum().backward()


# ----------------------------------------------------------------------
# unbroadcast helper
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "target_shape, expected",
    [
        ((3,), np.full(3, 2.0)),
        ((1, 3), np.full((1, 3), 2.0)),
        ((2, 1), np.full((2, 1), 3.0)),
        ((), np.array(6.0)),
        ((2, 3), np.ones((2, 3))),
    ],
)
def test_unbroadcast_shapes(target_shape: tuple[int, ...], expected: np.ndarray) -> None:
    result = unbroadcast(np.ones((2, 3)), target_shape)
    assert result.shape == target_shape
    np.testing.assert_allclose(result, expected)


def test_unbroadcast_rejects_incompatible() -> None:
    with pytest.raises(ValueError):
        unbroadcast(np.ones((2, 3)), (4,))
    with pytest.raises(ValueError):
        unbroadcast(np.ones(3), (2, 3))


# ----------------------------------------------------------------------
# Finite-difference checks: binary element-wise ops (+ broadcasting)
# ----------------------------------------------------------------------
BINARY_SHAPES = [
    ((3, 4), (3, 4)),  # same shape
    ((3, 4), (4,)),  # bias-style broadcast (N, D) + (D,)
    ((3, 4), (3, 1)),  # column broadcast
    ((3, 4), ()),  # scalar tensor
    ((1, 4), (3, 1)),  # both sides broadcast
]


@pytest.mark.parametrize("left_shape, right_shape", BINARY_SHAPES)
@pytest.mark.parametrize(
    "operation",
    [ops.add, ops.subtract, ops.multiply],
    ids=["add", "subtract", "multiply"],
)
def test_binary_gradients(operation, left_shape, right_shape, rng) -> None:
    left = rng.normal(size=left_shape)
    right = rng.normal(size=right_shape)
    check_gradients(lambda t: operation(t[0], t[1]), [left, right])


@pytest.mark.parametrize("left_shape, right_shape", BINARY_SHAPES)
def test_divide_gradients(left_shape, right_shape, rng) -> None:
    left = rng.normal(size=left_shape)
    # Keep denominators away from zero so finite differences are well-behaved.
    right = rng.uniform(0.5, 2.0, size=right_shape) * rng.choice([-1.0, 1.0], size=right_shape)
    check_gradients(lambda t: ops.divide(t[0], t[1]), [left, right])


def test_bias_gradient_has_parameter_shape(rng) -> None:
    inputs = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
    bias = Tensor(np.zeros(3), requires_grad=True)
    (inputs + bias).sum().backward()
    assert bias.grad.shape == (3,)
    np.testing.assert_allclose(bias.grad, np.full(3, 5.0))


def test_python_scalar_operators(rng) -> None:
    data = rng.uniform(0.5, 2.0, size=(2, 3))
    check_gradients(lambda t: 2.0 + t[0], [data])
    check_gradients(lambda t: 2.0 - t[0], [data])
    check_gradients(lambda t: 3.0 * t[0], [data])
    check_gradients(lambda t: 1.0 / t[0], [data])
    check_gradients(lambda t: -t[0], [data])
    check_gradients(lambda t: t[0] / 4.0 - 1.0, [data])


# ----------------------------------------------------------------------
# Finite-difference checks: matmul, reductions, activations
# ----------------------------------------------------------------------
@pytest.mark.parametrize("shapes", [((3, 4), (4, 2)), ((1, 5), (5, 1)), ((4, 1), (1, 3))])
def test_matmul_gradients(shapes, rng) -> None:
    left = rng.normal(size=shapes[0])
    right = rng.normal(size=shapes[1])
    check_gradients(lambda t: t[0] @ t[1], [left, right])


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum_gradients(axis, keepdims, rng) -> None:
    data = rng.normal(size=(3, 4))
    check_gradients(lambda t: t[0].sum(axis=axis, keepdims=keepdims), [data])


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_mean_gradients(axis, keepdims, rng) -> None:
    data = rng.normal(size=(3, 4))
    check_gradients(lambda t: t[0].mean(axis=axis, keepdims=keepdims), [data])


def test_reductions_on_scalar_and_3d(rng) -> None:
    check_gradients(lambda t: t[0].sum(), [np.array(1.5)])
    check_gradients(lambda t: t[0].mean(axis=(0, 2)), [rng.normal(size=(2, 3, 4))])


def test_relu_gradients_away_from_kink(rng) -> None:
    # Values at least 0.1 away from 0 so central differences never cross the kink.
    magnitude = rng.uniform(0.1, 2.0, size=(4, 5))
    data = magnitude * rng.choice([-1.0, 1.0], size=(4, 5))
    check_gradients(lambda t: t[0].relu(), [data])


def test_relu_values_and_kink_convention() -> None:
    tensor = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    output = tensor.relu()
    np.testing.assert_array_equal(output.data, [0.0, 0.0, 2.0])
    output.sum().backward()
    np.testing.assert_array_equal(tensor.grad, [0.0, 0.0, 1.0])


def test_tanh_gradients(rng) -> None:
    check_gradients(lambda t: t[0].tanh(), [rng.normal(size=(3, 4))])


# ----------------------------------------------------------------------
# Softmax cross-entropy
# ----------------------------------------------------------------------
def test_softmax_cross_entropy_gradients(rng) -> None:
    labels = np.array([0, 2, 1, 2])
    logits = rng.normal(size=(4, 3))
    check_gradients(lambda t: ops.softmax_cross_entropy(t[0], labels), [logits])


def test_softmax_cross_entropy_matches_naive_formula(rng) -> None:
    logits = rng.normal(size=(5, 4))
    labels = np.array([0, 1, 2, 3, 0])
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    expected = -np.mean(np.log(probabilities[np.arange(5), labels]))
    assert ops.softmax_cross_entropy(Tensor(logits), labels).item() == pytest.approx(expected)


def test_softmax_cross_entropy_gradient_formula(rng) -> None:
    logits = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    labels = np.array([3, 0, 1])
    ops.softmax_cross_entropy(logits, labels).backward()
    probabilities = np.exp(logits.data) / np.exp(logits.data).sum(axis=1, keepdims=True)
    one_hot = np.eye(4)[labels]
    np.testing.assert_allclose(logits.grad, (probabilities - one_hot) / 3)


def test_softmax_cross_entropy_is_stable_for_huge_logits() -> None:
    logits = Tensor([[1000.0, 0.0, -1000.0], [-1000.0, 1000.0, 0.0]], requires_grad=True)
    loss = ops.softmax_cross_entropy(logits, np.array([0, 2]))
    loss.backward()
    assert np.isfinite(loss.item())
    # Row 0 is perfectly correct (loss ~0); row 1 is wrong by 1000 (loss ~1000).
    assert loss.item() == pytest.approx(500.0)
    assert np.all(np.isfinite(logits.grad))


@pytest.mark.parametrize(
    "labels, error",
    [
        (np.array([0, 3]), ValueError),  # out of range
        (np.array([-1, 0]), ValueError),  # negative
        (np.array([0]), ValueError),  # wrong length
        (np.array([0.0, 1.0]), TypeError),  # float labels
    ],
)
def test_softmax_cross_entropy_validates_labels(labels, error) -> None:
    with pytest.raises(error):
        ops.softmax_cross_entropy(Tensor(np.zeros((2, 3))), labels)


def test_softmax_cross_entropy_requires_2d_logits() -> None:
    with pytest.raises(ValueError):
        ops.softmax_cross_entropy(Tensor(np.zeros(3)), np.array([0]))


# ----------------------------------------------------------------------
# Graph behaviour: reuse, branching, accumulation, ordering
# ----------------------------------------------------------------------
def test_repeated_parent_x_times_x() -> None:
    tensor = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    (tensor * tensor).sum().backward()
    np.testing.assert_allclose(tensor.grad, 2.0 * tensor.data)


def test_branching_diamond_graph() -> None:
    # a -> b = 2a, c = a^2 -> d = b * c = 2a^3, so dd/da = 6a^2.
    a = Tensor(1.5, requires_grad=True)
    b = a * 2.0
    c = a * a
    d = b * c
    d.backward()
    assert a.grad == pytest.approx(6.0 * 1.5 ** 2)


def test_shared_intermediate_node_is_processed_once(rng) -> None:
    # hidden is used by two branches; topological order must sum both first.
    data = rng.normal(size=(2, 3))
    check_gradients(
        lambda t: (lambda hidden: hidden * hidden + hidden.tanh())(t[0] * 3.0 + 1.0),
        [data],
    )


def test_gradients_accumulate_across_backward_calls() -> None:
    tensor = Tensor([1.0, 2.0], requires_grad=True)
    (tensor * 3.0).sum().backward()
    (tensor * 3.0).sum().backward()
    np.testing.assert_allclose(tensor.grad, [6.0, 6.0])


def test_unused_input_gets_zero_gradient() -> None:
    used = Tensor(2.0, requires_grad=True)
    unused = Tensor(5.0, requires_grad=True)
    (used * used).backward()
    assert unused.grad == 0.0


def test_mixed_requires_grad_only_tracks_trainable(rng) -> None:
    trainable = Tensor(rng.normal(size=(2, 2)), requires_grad=True)
    frozen = Tensor(rng.normal(size=(2, 2)))
    (trainable * frozen).sum().backward()
    np.testing.assert_allclose(trainable.grad, frozen.data)
    np.testing.assert_array_equal(frozen.grad, np.zeros((2, 2)))


def test_deep_chain_does_not_hit_recursion_limit() -> None:
    tensor = Tensor(0.0, requires_grad=True)
    result = tensor
    for _ in range(sys.getrecursionlimit() + 500):
        result = result + 1.0
    result.backward()
    assert tensor.grad == pytest.approx(1.0)


def test_two_layer_network_end_to_end(rng) -> None:
    """Composite check resembling an MLP: matmul + bias + tanh/relu + matmul + loss."""
    inputs = rng.normal(size=(4, 3))
    labels = np.array([0, 1, 1, 0])
    weight_1 = rng.normal(size=(3, 5))
    bias_1 = rng.normal(size=(5,))
    weight_2 = rng.normal(size=(5, 2))
    bias_2 = rng.normal(size=(2,))

    def network(t: list[Tensor]) -> Tensor:
        hidden = (Tensor(inputs) @ t[0] + t[1]).tanh()
        logits = hidden @ t[2] + t[3]
        return ops.softmax_cross_entropy(logits, labels)

    check_gradients(network, [weight_1, bias_1, weight_2, bias_2])


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------
def test_incompatible_broadcast_raises() -> None:
    with pytest.raises(ValueError, match="not broadcastable"):
        Tensor(np.ones((2, 3))) + Tensor(np.ones(4))


def test_matmul_shape_errors() -> None:
    with pytest.raises(ValueError, match="inner dimensions"):
        Tensor(np.ones((2, 3))) @ Tensor(np.ones((2, 3)))
    with pytest.raises(ValueError, match="2-D"):
        Tensor(np.ones(3)) @ Tensor(np.ones((3, 1)))


def test_divide_by_zero_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        Tensor([1.0, 2.0]) / Tensor([1.0, 0.0])


def test_backward_requires_scalar_or_matching_grad() -> None:
    tensor = Tensor([1.0, 2.0], requires_grad=True)
    doubled = tensor * 2.0
    with pytest.raises(ValueError):
        doubled.backward()
    with pytest.raises(ValueError):
        doubled.backward(np.ones(3))


def test_invalid_axis_raises() -> None:
    with pytest.raises(ValueError):
        Tensor(np.ones((2, 3))).sum(axis=2)
    with pytest.raises(ValueError):
        Tensor(np.ones((2, 3))).mean(axis=(0, 0))


def test_non_finite_gradient_fails_loudly() -> None:
    tensor = Tensor([1.0], requires_grad=True)
    output = tensor * 2.0
    with pytest.raises(FloatingPointError):
        output.backward(np.array([np.inf]))
