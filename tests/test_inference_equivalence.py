"""Phase 1: Numerical Equivalence Test Suite.

Validates that dense masked inference and sparse-aware inference produce
effectively identical logits (within float64 precision rtol=1e-10, atol=1e-10)
and 100% identical class predictions across all sparsities, seeds, batch sizes,
activations, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.tensor import Tensor
from experiments.inference_validation import (
    dense_forward_numpy,
    extract_sparse_layers,
    sparse_forward_numpy,
    verify_equivalence,
)
from nn.layers import MLP, Linear
from nn.optimizers import Adam
from prune.importance import compute_layer_scores
from prune.scheduler import prune_to_sparsity
from train.dataset import make_spirals, train_test_split

SPARSITY_TARGETS = [0.0, 0.50, 0.75, 0.90, 0.95]
TEST_SEEDS = [42, 43, 44]
BATCH_SIZES = [1, 16, 128]


# ----------------------------------------------------------------------
# Helper to build a pruned model for a given seed and sparsity
# ----------------------------------------------------------------------
def create_pruned_model(
    seed: int,
    sparsity: float,
    activation: str = "relu",
    hidden_sizes: tuple[int, ...] = (64, 64),
) -> MLP:
    """Build an MLP and prune it to target sparsity."""
    rng = np.random.default_rng(seed)
    layer_sizes = [2, *hidden_sizes, 3]
    model = MLP(layer_sizes, rng, activation=activation)

    if sparsity > 0.0:
        optimizer = Adam(model.parameters())
        scores = compute_layer_scores(model, "magnitude")
        prune_to_sparsity(model, optimizer, scores, sparsity)

    return model


# ----------------------------------------------------------------------
# 1. Equivalence across Sparsity Targets, Seeds, and Batch Sizes
# ----------------------------------------------------------------------
@pytest.mark.parametrize("sparsity", SPARSITY_TARGETS)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_dense_sparse_equivalence_across_sparsities_and_seeds(sparsity: float, seed: int) -> None:
    model = create_pruned_model(seed, sparsity, activation="relu")
    sparse_layers = extract_sparse_layers(model)

    features, labels = make_spirals(samples_per_class=60, num_classes=3, seed=seed)
    _, test_x, _, test_y = train_test_split(features, labels, test_fraction=0.2, seed=seed)

    dense_logits = dense_forward_numpy(model, test_x)
    sparse_logits = sparse_forward_numpy(sparse_layers, test_x, activation="relu")

    check = verify_equivalence(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)
    assert check["pass_status"], f"Failed for sparsity {sparsity}, seed {seed}: {check}"
    assert check["classes_match"]
    np.testing.assert_allclose(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_dense_sparse_equivalence_batch_sizes(batch_size: int) -> None:
    model = create_pruned_model(seed=42, sparsity=0.90, activation="relu")
    sparse_layers = extract_sparse_layers(model)

    rng = np.random.default_rng(99)
    inputs = rng.normal(size=(batch_size, 2))

    dense_logits = dense_forward_numpy(model, inputs)
    sparse_logits = sparse_forward_numpy(sparse_layers, inputs, activation="relu")

    check = verify_equivalence(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)
    assert check["pass_status"]
    np.testing.assert_allclose(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------------
# 2. Activation Functions: ReLU and Tanh
# ----------------------------------------------------------------------
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_dense_sparse_equivalence_activations(activation: str) -> None:
    model = create_pruned_model(seed=43, sparsity=0.75, activation=activation)
    sparse_layers = extract_sparse_layers(model)

    rng = np.random.default_rng(101)
    inputs = rng.normal(size=(32, 2))

    dense_logits = dense_forward_numpy(model, inputs)
    sparse_logits = sparse_forward_numpy(sparse_layers, inputs, activation=activation)

    check = verify_equivalence(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)
    assert check["pass_status"]
    np.testing.assert_allclose(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------------
# 3. Edge Cases
# ----------------------------------------------------------------------
def test_edge_case_all_active_mask() -> None:
    """100% active mask (dense)."""
    model = create_pruned_model(seed=42, sparsity=0.0)
    sparse_layers = extract_sparse_layers(model)
    inputs = np.random.default_rng(1).normal(size=(25, 2))

    dense_logits = dense_forward_numpy(model, inputs)
    sparse_logits = sparse_forward_numpy(sparse_layers, inputs)
    np.testing.assert_allclose(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)


def test_edge_case_nearly_all_inactive_mask() -> None:
    """Only 1 active connection per layer."""
    model = MLP([2, 16, 16, 3], np.random.default_rng(5))
    for layer in model.layers:
        sparse_mask = np.zeros(layer.mask.shape, dtype=bool)
        sparse_mask[0, 0] = True
        layer.apply_mask(sparse_mask)

    sparse_layers = extract_sparse_layers(model)
    inputs = np.random.default_rng(2).normal(size=(20, 2))

    dense_logits = dense_forward_numpy(model, inputs)
    sparse_logits = sparse_forward_numpy(sparse_layers, inputs)
    np.testing.assert_allclose(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)


def test_edge_case_fully_masked_layer() -> None:
    """Entire layer pruned to 0 (only biases remain)."""
    model = MLP([2, 8, 3], np.random.default_rng(7))
    for layer in model.layers:
        layer.apply_mask(np.zeros(layer.mask.shape, dtype=bool))
        layer.bias.data = np.arange(layer.bias.data.size, dtype=float) + 1.0

    sparse_layers = extract_sparse_layers(model)
    inputs = np.random.default_rng(3).normal(size=(10, 2))

    dense_logits = dense_forward_numpy(model, inputs)
    sparse_logits = sparse_forward_numpy(sparse_layers, inputs)
    # Output should simply be the final layer bias for every sample
    np.testing.assert_allclose(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)
    for row in range(10):
        np.testing.assert_allclose(dense_logits[row], model.layers[-1].bias.data)


def test_edge_case_nonzero_biases() -> None:
    """Verify biases are properly integrated across layers."""
    model = create_pruned_model(seed=44, sparsity=0.80)
    for idx, layer in enumerate(model.layers):
        layer.bias.data = np.random.default_rng(idx).normal(size=layer.bias.data.shape)

    sparse_layers = extract_sparse_layers(model)
    inputs = np.random.default_rng(4).normal(size=(15, 2))

    dense_logits = dense_forward_numpy(model, inputs)
    sparse_logits = sparse_forward_numpy(sparse_layers, inputs)
    np.testing.assert_allclose(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)


def test_edge_case_uneven_minibatch() -> None:
    """Test non-power-of-2 / prime minibatch sizes (e.g. 13 and 37)."""
    model = create_pruned_model(seed=42, sparsity=0.90)
    sparse_layers = extract_sparse_layers(model)

    for prime_size in [13, 37]:
        inputs = np.random.default_rng(prime_size).normal(size=(prime_size, 2))
        dense_logits = dense_forward_numpy(model, inputs)
        sparse_logits = sparse_forward_numpy(sparse_layers, inputs)
        np.testing.assert_allclose(dense_logits, sparse_logits, rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------------
# 4. Evaluation Purity (Immutability & Repeatability)
# ----------------------------------------------------------------------
def test_evaluation_purity_and_immutability() -> None:
    """Verify that forward pass does not mutate inputs or model parameters."""
    model = create_pruned_model(seed=42, sparsity=0.90)
    sparse_layers = extract_sparse_layers(model)

    inputs = np.random.default_rng(55).normal(size=(20, 2))
    inputs_before = inputs.copy()
    weights_before = [l.weight.data.copy() for l in model.layers]
    masks_before = [l.mask.copy() for l in model.layers]
    biases_before = [l.bias.data.copy() for l in model.layers]

    # Run inference multiple times
    out1_dense = dense_forward_numpy(model, inputs)
    out1_sparse = sparse_forward_numpy(sparse_layers, inputs)
    out2_dense = dense_forward_numpy(model, inputs)
    out2_sparse = sparse_forward_numpy(sparse_layers, inputs)

    # Immutability
    np.testing.assert_array_equal(inputs, inputs_before)
    for l, w, m, b in zip(model.layers, weights_before, masks_before, biases_before):
        np.testing.assert_array_equal(l.weight.data, w)
        np.testing.assert_array_equal(l.mask, m)
        np.testing.assert_array_equal(l.bias.data, b)

    # Exact repeatability
    np.testing.assert_array_equal(out1_dense, out2_dense)
    np.testing.assert_array_equal(out1_sparse, out2_sparse)
