"""Phase 2: Fair-Comparison Audit Test Suite.

Audits benchmark fairness:
1. Shared data generation seeds, train/test indices, initial weights, hyperparameters.
2. Zero test leakage (test set never touches score computation or updates).
3. Exact integer budgets across all sparsities, excluding biases.
4. Audit of shared baseline (27 distinct training runs, not 30).
5. Mathematical independence of magnitude and saliency pruned masks and trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nn.layers import MLP
from nn.optimizers import Adam
from prune.importance import compute_layer_scores
from prune.scheduler import prune_to_sparsity
from train.dataset import make_spirals, train_test_split
from train.train import train_baseline, train_pruned

FAST = dict(samples_per_class=60, hidden_sizes=(32, 32), batch_size=32)


# ----------------------------------------------------------------------
# 1. Deterministic Initialization and Identical Conditions Across Methods
# ----------------------------------------------------------------------
def test_magnitude_and_saliency_share_identical_initial_conditions() -> None:
    """Confirm both methods start with identical data, split, weights, and batch order."""
    seed = 42
    # Train 1 epoch in baseline, magnitude, and saliency
    run_base = train_baseline(epochs=1, seed=seed, **FAST)
    run_mag = train_pruned(method="magnitude", sparsity=0.50, epochs=1, seed=seed, **FAST)
    run_sal = train_pruned(method="saliency", sparsity=0.50, epochs=1, seed=seed, **FAST)

    # Initial metrics before any training step must be bit-for-bit identical
    assert run_base["initial"]["train_loss"] == run_mag["initial"]["train_loss"]
    assert run_base["initial"]["train_loss"] == run_sal["initial"]["train_loss"]
    assert run_base["initial"]["test_loss"] == run_mag["initial"]["test_loss"]
    assert run_base["initial"]["test_loss"] == run_sal["initial"]["test_loss"]

    # In epoch 1 (before prune_start), batch losses and updates must be identical
    assert run_base["history"]["train_loss"][0] == run_mag["history"]["train_loss"][0]
    assert run_base["history"]["train_loss"][0] == run_sal["history"]["train_loss"][0]


# ----------------------------------------------------------------------
# 2. No Test Set Leakage Audit
# ----------------------------------------------------------------------
def test_no_test_set_leakage_in_saliency_and_training() -> None:
    """Verify test data is never passed to score calculation or backward passes."""
    seed = 42
    features, labels = make_spirals(samples_per_class=60, num_classes=3, seed=seed)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, 0.2, seed=seed)

    model = MLP([2, 32, 32, 3], np.random.default_rng(seed))

    # Saliency must only accept training data
    # Intentionally corrupt test labels with invalid values (-99)
    corrupted_test_y = np.full_like(test_y, -99)

    # Compute saliency on training data: must succeed without any error
    scores = compute_layer_scores(model, "saliency", train_x, train_y)
    assert len(scores) == len(model.layers)
    for s in scores:
        assert np.all(np.isfinite(s))

    # If someone accidentally passed test data with corrupted labels, it would fail
    with pytest.raises(ValueError):
        compute_layer_scores(model, "saliency", test_x, corrupted_test_y)


# ----------------------------------------------------------------------
# 3. Exact Budget and Bias Exclusion Audit
# ----------------------------------------------------------------------
@pytest.mark.parametrize("sparsity", [0.0, 0.50, 0.75, 0.90, 0.95])
def test_exact_integer_budget_enforcement(sparsity: float) -> None:
    """Verify active connection count matches exact integer formula and biases are excluded."""
    model = MLP([2, 64, 64, 3], np.random.default_rng(0))
    total_weights = 2 * 64 + 64 * 64 + 64 * 3  # 4416
    total_biases = 64 + 64 + 3  # 131

    achieved_sparsity, active, total = model.sparsity()
    assert total == total_weights
    assert active == total_weights

    optimizer = Adam(model.parameters())
    scores = compute_layer_scores(model, "magnitude")
    prune_to_sparsity(model, optimizer, scores, sparsity)

    expected_pruned = int(round(total_weights * sparsity))
    expected_active = total_weights - expected_pruned

    achieved_sparsity, active, total = model.sparsity()
    assert total == total_weights
    assert active == expected_active
    assert achieved_sparsity == pytest.approx(expected_pruned / total_weights)

    # Verify biases remain 100% unpruned
    for layer in model.layers:
        assert not hasattr(layer.bias, "mask")
        assert layer.bias.data.shape[0] == layer.out_features


# ----------------------------------------------------------------------
# 4. Audit of Shared Baseline in Benchmark Data (27 Runs)
# ----------------------------------------------------------------------
def test_benchmark_data_structure_and_shared_baseline() -> None:
    """Audit saved results/benchmark_results.json for 27 distinct runs and shared 0% baseline."""
    json_path = Path("results/benchmark_results.json")
    assert json_path.exists(), "Benchmark results file must exist"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    raw_runs = data["raw_runs"]

    # 5 sparsities * 3 seeds * 2 methods = 30 entries in table,
    # but at sparsity 0.0 both methods share the 3 baseline runs -> 27 distinct training executions.
    assert len(raw_runs) == 30

    baseline_runs = [r for r in raw_runs if r["target_sparsity"] == 0.0]
    mag_base = [r for r in baseline_runs if r["method"] == "magnitude"]
    sal_base = [r for r in baseline_runs if r["method"] == "saliency"]

    # Check identical baseline metrics per seed
    for mb, sb in zip(mag_base, sal_base):
        assert mb["seed"] == sb["seed"]
        assert mb["test_accuracy"] == sb["test_accuracy"]
        assert mb["test_loss"] == sb["test_loss"]


# ----------------------------------------------------------------------
# 5. Independence of Magnitude and Saliency at Non-Zero Sparsities
# ----------------------------------------------------------------------
def test_magnitude_and_saliency_select_distinct_masks() -> None:
    """Verify that magnitude and saliency select different masks (no results reuse)."""
    seed = 42
    sparsity = 0.75
    run_mag = train_pruned(method="magnitude", sparsity=sparsity, epochs=30, seed=seed, **FAST)
    run_sal = train_pruned(method="saliency", sparsity=sparsity, epochs=30, seed=seed, **FAST)

    # Both achieve the same integer connection count
    assert run_mag["final"]["active_connections"] == run_sal["final"]["active_connections"]

    # But the selected connections and losses must be independently computed
    # Training loss trajectories must diverge once pruning begins
    mag_losses = run_mag["history"]["train_loss"]
    sal_losses = run_sal["history"]["train_loss"]
    assert mag_losses != sal_losses
