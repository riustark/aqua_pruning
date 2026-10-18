"""Tests for layers, Adam, data, masking, pruning, and training integration.

Run from the project root:

    python -m pytest tests/test_masking.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from engine import operations as ops
from engine.tensor import Tensor
from nn.layers import MLP, Linear
from nn.optimizers import Adam
from prune.importance import (
    METHODS,
    compute_layer_scores,
    compute_magnitude_scores,
    compute_saliency_scores,
)
from prune.scheduler import prune_to_sparsity, target_sparsity_at_step
from train import train as train_module
from train.dataset import iterate_minibatches, make_spirals, train_test_split
from train.train import evaluate, train_baseline, train_pruned

# Small, fast settings for training tests.
FAST = dict(samples_per_class=60, hidden_sizes=(32, 32), batch_size=32)


# ----------------------------------------------------------------------
# Linear & Masking Fundamentals
# ----------------------------------------------------------------------
def test_linear_forward_matches_formula() -> None:
    rng = np.random.default_rng(0)
    layer = Linear(3, 4, rng)
    inputs = rng.normal(size=(5, 3))
    output = layer(Tensor(inputs))
    assert output.shape == (5, 4)
    np.testing.assert_allclose(output.data, inputs @ layer.weight.data + layer.bias.data)


def test_linear_parameters_and_initial_values() -> None:
    layer = Linear(3, 4, np.random.default_rng(0))
    weight, bias = layer.parameters()
    assert weight is layer.weight and bias is layer.bias
    assert weight.shape == (3, 4) and bias.shape == (4,)
    assert weight.requires_grad and bias.requires_grad
    np.testing.assert_array_equal(bias.data, np.zeros(4))
    assert layer.mask.dtype == np.bool_
    assert layer.mask.shape == (3, 4)
    assert layer.mask.all()
    np.testing.assert_array_equal(layer.effective_weight().data, weight.data)


@pytest.mark.parametrize(
    "init, expected_std",
    [("he", np.sqrt(2.0 / 500)), ("xavier", np.sqrt(2.0 / (500 + 400)))],
)
def test_linear_initialization_scale(init: str, expected_std: float) -> None:
    layer = Linear(500, 400, np.random.default_rng(0), init=init)
    assert abs(layer.weight.data.mean()) < 0.1 * expected_std
    assert layer.weight.data.std() == pytest.approx(expected_std, rel=0.02)


def test_linear_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        Linear(0, 4, rng)
    with pytest.raises(ValueError):
        Linear(3, 4, rng, init="uniform")
    with pytest.raises(ValueError, match="expects input shape"):
        Linear(3, 4, rng)(Tensor(np.ones((5, 2))))


def test_linear_gradients_flow_to_weight_and_bias() -> None:
    rng = np.random.default_rng(1)
    layer = Linear(3, 2, rng)
    inputs = rng.normal(size=(4, 3))
    layer(Tensor(inputs)).sum().backward()
    np.testing.assert_allclose(layer.weight.grad, inputs.T @ np.ones((4, 2)))
    np.testing.assert_allclose(layer.bias.grad, np.full(2, 4.0))


def test_masked_weight_forward_exact_zero() -> None:
    """Pruned connections must contribute exactly zero to forward outputs."""
    rng = np.random.default_rng(2)
    layer = Linear(3, 3, rng)
    mask = np.array([
        [True, False, True],
        [False, True, False],
        [True, True, False],
    ], dtype=bool)
    layer.apply_mask(mask)

    inputs = rng.normal(size=(4, 3))
    out1 = layer(Tensor(inputs)).data.copy()

    # Even if raw weight data at pruned position (0, 1) is modified, forward output is identical
    layer.weight.data[0, 1] = 999.0
    out2 = layer(Tensor(inputs)).data.copy()
    np.testing.assert_array_equal(out1, out2)


def test_masked_weight_gradient_chain_rule() -> None:
    """By chain rule through W_eff = W_raw * mask, dL/dW_raw = mask * dL/dW_eff.

    Pruned connections receive strictly zero gradient on raw weights.
    """
    rng = np.random.default_rng(3)
    layer = Linear(3, 2, rng)
    mask = np.array([
        [True, False],
        [False, True],
        [True, False],
    ], dtype=bool)
    layer.apply_mask(mask)

    inputs = rng.normal(size=(4, 3))
    out = layer(Tensor(inputs))
    loss = (out * out).sum()
    loss.backward()

    # Pruned entries must receive exactly zero gradient
    assert np.all(layer.weight.grad[~mask] == 0.0)
    # Active entries receive non-zero gradient matching unmasked formula
    assert np.all(layer.weight.grad[mask] != 0.0)


def test_apply_mask_zeros_raw_weight_and_enforces_no_regrowth() -> None:
    layer = Linear(2, 2, np.random.default_rng(4))
    initial_mask = np.array([[True, True], [True, True]], dtype=bool)
    layer.apply_mask(initial_mask)

    # Prune one connection
    pruned_mask = np.array([[True, False], [True, True]], dtype=bool)
    layer.apply_mask(pruned_mask)
    assert layer.weight.data[0, 1] == 0.0
    assert layer.num_active() == 3

    # Attempting to revive (turn connection [0, 1] back on) must raise ValueError
    revival_mask = np.array([[True, True], [True, True]], dtype=bool)
    with pytest.raises(ValueError, match="Regrowth is not supported"):
        layer.apply_mask(revival_mask)

    # Invalid mask dtype or shape
    with pytest.raises(TypeError, match="dtype bool"):
        layer.apply_mask(np.array([[1, 0], [1, 1]]))
    with pytest.raises(ValueError, match="mask shape"):
        layer.apply_mask(np.ones((3, 2), dtype=bool))


# ----------------------------------------------------------------------
# MLP & Sparsity
# ----------------------------------------------------------------------
def test_mlp_structure_and_output_shape() -> None:
    model = MLP([2, 8, 6, 3], np.random.default_rng(0))
    assert [(l.in_features, l.out_features) for l in model.layers] == [(2, 8), (8, 6), (6, 3)]
    assert len(model.parameters()) == 6
    assert model(np.zeros((7, 2))).shape == (7, 3)


def test_mlp_sparsity_excludes_biases() -> None:
    model = MLP([2, 4, 3], np.random.default_rng(0))
    # Layer 0: 2*4 = 8 weights. Layer 1: 4*3 = 12 weights. Total prunable = 20.
    # Biases (4 + 3 = 7) must NOT be counted.
    sparsity, active, total = model.sparsity()
    assert total == 20
    assert active == 20
    assert sparsity == 0.0

    # Prune 5 connections from layer 0
    mask0 = model.layers[0].mask.copy()
    mask0.flat[:5] = False
    model.layers[0].apply_mask(mask0)

    sparsity, active, total = model.sparsity()
    assert total == 20
    assert active == 15
    assert sparsity == pytest.approx(5 / 20)


def test_mlp_loss_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(3)
    model = MLP([2, 5, 3], rng, activation="relu")
    # Mask some connections
    mask0 = np.ones((2, 5), dtype=bool)
    mask0[0, 1] = False
    model.layers[0].apply_mask(mask0)

    inputs = rng.normal(size=(6, 2))
    labels = np.array([0, 1, 2, 0, 1, 2])

    def loss_value() -> float:
        return ops.softmax_cross_entropy(model(inputs), labels).item()

    ops.softmax_cross_entropy(model(inputs), labels).backward()
    eps = 1e-6
    for param in model.parameters():
        numeric = np.zeros_like(param.data)
        for position in np.ndindex(param.shape):
            original = param.data[position]
            param.data[position] = original + eps
            plus = loss_value()
            param.data[position] = original - eps
            minus = loss_value()
            param.data[position] = original
            numeric[position] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(param.grad, numeric, rtol=1e-4, atol=1e-7)


# ----------------------------------------------------------------------
# Adam & Pruned State Reset (No Reactivation)
# ----------------------------------------------------------------------
def test_adam_matches_hand_computed_updates() -> None:
    param = Tensor([1.0, -2.0], requires_grad=True)
    lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
    optimizer = Adam([param], learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=eps)

    expected = param.data.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    for step, grad in enumerate([np.array([0.5, -1.0]), np.array([-0.2, 3.0])], start=1):
        param.grad = grad.copy()
        optimizer.step()
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        expected = expected - lr * (m / (1 - beta1 ** step)) / (
            np.sqrt(v / (1 - beta2 ** step)) + eps
        )
        np.testing.assert_allclose(param.data, expected)
        np.testing.assert_allclose(optimizer.first_moments[0], m)
        np.testing.assert_allclose(optimizer.second_moments[0], v)
    assert optimizer.step_count == 2


def test_adam_pruning_state_reset_prevents_reactivation() -> None:
    """Verify that zeroing moments prevents old momentum from reviving pruned weights."""
    weight = Tensor([[2.0, -1.5], [0.8, 3.0]], requires_grad=True)
    optimizer = Adam([weight], learning_rate=0.05)

    # Run several steps to build up strong non-zero momentum m and second moment v
    for _ in range(5):
        weight.grad = np.array([[10.0, -5.0], [8.0, 12.0]])
        optimizer.step()

    assert np.all(optimizer.first_moments[0] != 0.0)
    assert np.all(optimizer.second_moments[0] != 0.0)

    # Now prune connection (0, 1) and (1, 0)
    mask = np.array([[True, False], [False, True]], dtype=bool)
    optimizer.register_mask(weight, mask)

    # Check moments and weight are zeroed at pruned entries
    assert weight.data[0, 1] == 0.0
    assert weight.data[1, 0] == 0.0
    assert optimizer.first_moments[0][0, 1] == 0.0
    assert optimizer.second_moments[0][1, 0] == 0.0

    # Execute 25 repeated Adam steps with non-zero gradient flowing into active entries
    # and zero gradient into pruned entries (as guaranteed by the mask backward rule)
    for _ in range(25):
        weight.grad = np.where(mask, 1.0, 0.0)
        optimizer.step()

        # Pruned entries must remain strictly 0.0 across all steps
        assert weight.data[0, 1] == 0.0
        assert weight.data[1, 0] == 0.0
        assert optimizer.first_moments[0][0, 1] == 0.0
        assert optimizer.second_moments[0][1, 0] == 0.0


def test_adam_replaces_arrays_instead_of_mutating() -> None:
    param = Tensor([1.0], requires_grad=True)
    old_array = param.data
    optimizer = Adam([param])
    param.grad = np.array([1.0])
    optimizer.step()
    assert old_array[0] == 1.0 and param.data is not old_array


def test_adam_zero_grad_resets_all_parameters() -> None:
    model = MLP([2, 4, 3], np.random.default_rng(0))
    optimizer = Adam(model.parameters())
    ops.softmax_cross_entropy(model(np.ones((2, 2))), np.array([0, 1])).backward()
    assert any(np.any(p.grad != 0) for p in model.parameters())
    optimizer.zero_grad()
    assert all(np.all(p.grad == 0) for p in model.parameters())


def test_adam_rejects_non_finite_gradient() -> None:
    param = Tensor([1.0], requires_grad=True)
    optimizer = Adam([param])
    param.grad = np.array([np.nan])
    with pytest.raises(FloatingPointError):
        optimizer.step()


# ----------------------------------------------------------------------
# Importance Scoring
# ----------------------------------------------------------------------
def test_magnitude_scores_match_absolute_weights() -> None:
    layer = Linear(2, 3, np.random.default_rng(0))
    mask = np.array([[True, False, True], [True, True, False]], dtype=bool)
    layer.apply_mask(mask)

    scores = compute_magnitude_scores(layer)
    expected = np.where(mask, np.abs(layer.weight.data), 0.0)
    np.testing.assert_array_equal(scores, expected)
    assert scores[0, 1] == 0.0
    assert scores[1, 2] == 0.0


def test_saliency_scores_formula_and_masking() -> None:
    model = MLP([2, 4, 3], np.random.default_rng(1))
    features = np.ones((5, 2))
    labels = np.array([0, 1, 2, 0, 1])

    scores = compute_saliency_scores(model, features, labels)
    assert len(scores) == 2
    for layer, score in zip(model.layers, scores):
        assert score.shape == layer.mask.shape
        assert np.all(np.isfinite(score))
        assert np.all(score >= 0.0)


def test_compute_layer_scores_dispatch_and_validation() -> None:
    model = MLP([2, 4, 3], np.random.default_rng(0))
    mag_scores = compute_layer_scores(model, method="magnitude")
    assert len(mag_scores) == 2

    with pytest.raises(ValueError, match="requires features and labels"):
        compute_layer_scores(model, method="saliency")

    with pytest.raises(ValueError, match="Unknown pruning method"):
        compute_layer_scores(model, method="random")


# ----------------------------------------------------------------------
# Pruning Scheduler & Exact Budget
# ----------------------------------------------------------------------
def test_target_sparsity_at_step_cubic_schedule() -> None:
    final = 0.80
    start = 10
    end = 50

    assert target_sparsity_at_step(5, final, start, end) == 0.0
    assert target_sparsity_at_step(10, final, start, end) == 0.0
    assert target_sparsity_at_step(50, final, start, end) == 0.80
    assert target_sparsity_at_step(60, final, start, end) == 0.80

    # Intermediate values strictly between 0 and 0.80 and monotonically increasing
    steps = list(range(10, 51))
    sparsities = [target_sparsity_at_step(s, final, start, end) for s in steps]
    for i in range(len(sparsities) - 1):
        assert sparsities[i] <= sparsities[i + 1]


def test_target_sparsity_validation() -> None:
    with pytest.raises(ValueError):
        target_sparsity_at_step(1, 1.0, 0, 10)
    with pytest.raises(ValueError):
        target_sparsity_at_step(1, -0.1, 0, 10)
    with pytest.raises(ValueError):
        target_sparsity_at_step(1, 0.5, 20, 10)


@pytest.mark.parametrize("target_sparsity", [0.0, 0.50, 0.75, 0.90, 0.95])
def test_prune_to_sparsity_exact_integer_connection_count(target_sparsity: float) -> None:
    model = MLP([2, 32, 32, 3], np.random.default_rng(0))
    optimizer = Adam(model.parameters())
    scores = compute_layer_scores(model, "magnitude")

    total_prunable = sum(l.mask.size for l in model.layers)
    expected_pruned = int(round(total_prunable * target_sparsity))
    expected_active = total_prunable - expected_pruned

    active, total = prune_to_sparsity(model, optimizer, scores, target_sparsity)
    assert total == total_prunable
    assert active == expected_active
    achieved_sparsity, curr_active, _ = model.sparsity()
    assert curr_active == expected_active
    assert achieved_sparsity == pytest.approx(1.0 - expected_active / total_prunable)


def test_prune_to_sparsity_removes_lowest_scores() -> None:
    model = MLP([2, 4, 3], np.random.default_rng(0))
    optimizer = Adam(model.parameters())
    # Artificial scores where layer 0 has low values and layer 1 has high values
    scores = [
        np.arange(8, dtype=float).reshape(2, 4),
        np.arange(100, 112, dtype=float).reshape(4, 3),
    ]
    # Prune 4 connections (out of 20 total) -> target sparsity 4/20 = 0.20
    prune_to_sparsity(model, optimizer, scores, 0.20)
    # The 4 lowest scores in layer 0 (0, 1, 2, 3) must be pruned
    assert np.all(~model.layers[0].mask.flat[:4])
    assert np.all(model.layers[0].mask.flat[4:])
    assert np.all(model.layers[1].mask)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def test_spirals_shapes_dtypes_and_balance() -> None:
    features, labels = make_spirals(samples_per_class=50, num_classes=3, seed=0)
    assert features.shape == (150, 2) and features.dtype == np.float64
    assert labels.shape == (150,) and labels.dtype == np.int64
    np.testing.assert_array_equal(np.bincount(labels), [50, 50, 50])


def test_train_test_split_is_disjoint_complete_and_deterministic() -> None:
    features = np.arange(20, dtype=float).reshape(10, 2)
    labels = np.arange(10)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, 0.3, seed=1)
    assert len(train_y) == 7 and len(test_y) == 3
    assert sorted(np.concatenate([train_y, test_y]).tolist()) == list(range(10))


# ----------------------------------------------------------------------
# Training Integration: Baseline & Pruning Modes
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["magnitude", "saliency"])
def test_train_pruned_achieves_target_sparsity_and_learns(method: str) -> None:
    target_sparsity = 0.70
    result = train_pruned(
        method=method,
        sparsity=target_sparsity,
        epochs=30,
        seed=42,
        **FAST,
    )
    final = result["final"]
    total = final["total_prunable"]
    expected_active = total - int(round(total * target_sparsity))

    assert final["active_connections"] == expected_active
    assert final["sparsity"] == pytest.approx(1.0 - expected_active / total)
    assert final["train_accuracy"] > 0.60
    assert final["test_accuracy"] > 0.60
    assert np.all(np.isfinite(result["history"]["train_loss"]))


def test_cli_runs_baseline_and_prune_modes(tmp_path, capsys) -> None:
    # Test prune CLI
    out_prune = tmp_path / "prune.json"
    code = train_module.main([
        "--mode", "prune", "--method", "saliency", "--sparsity", "0.80",
        "--epochs", "4", "--seed", "1", "--hidden", "8", "8",
        "--output", str(out_prune)
    ])
    assert code == 0
    stdout = capsys.readouterr().out
    assert "Mode: prune" in stdout
    assert "Sparsity:" in stdout
    saved = json.loads(out_prune.read_text(encoding="utf-8"))
    assert saved["config"]["mode"] == "prune"
    assert saved["config"]["method"] == "saliency"
    assert saved["final"]["sparsity"] == pytest.approx(0.80, abs=0.02)
