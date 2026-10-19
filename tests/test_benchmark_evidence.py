"""Regression checks for trained-model provenance and read-only benchmarking."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from experiments.benchmark import main, measure_forward_costs
from experiments.optimization_benchmark import run_cpu_benchmark
from train.snapshot import load_snapshot, model_fingerprint, snapshot_test_data
import train.train as training


def capture_training(**kwargs):
    models, optimizers = [], []
    model_class, optimizer_class = training.MLP, training.Adam

    def make_model(*args, **options):
        model = model_class(*args, **options)
        models.append(model)
        return model

    def make_optimizer(*args, **options):
        optimizer = optimizer_class(*args, **options)
        optimizers.append(optimizer)
        return optimizer

    with patch.object(training, "MLP", make_model), patch.object(training, "Adam", make_optimizer):
        result = training.train_model(**kwargs)
    return result, models[0], optimizers[0]


def state_arrays(model, optimizer):
    return [array.copy() for layer in model.layers for array in
            (layer.weight.data, layer.weight.grad, layer.bias.data, layer.bias.grad, layer.mask)] + [
                array.copy() for array in optimizer.first_moments + optimizer.second_moments]


@pytest.mark.parametrize("mode,method", [("baseline", "magnitude"),
                                         ("prune", "magnitude"), ("prune", "saliency")])
def test_export_preserves_complete_training_state_and_predictions(tmp_path, mode, method):
    options = dict(mode=mode, method=method, sparsity=.9 if mode == "prune" else 0,
                   epochs=12, seed=42, hidden_sizes=(8, 8), samples_per_class=30)
    expected, original, original_adam = capture_training(**options)
    path = tmp_path / "trained.npz"
    actual, exported, exported_adam = capture_training(**options, snapshot_path=path)
    assert actual == expected  # Includes every loss, accuracy and sparsity epoch.
    assert original_adam.step_count == exported_adam.step_count
    for before, after in zip(state_arrays(original, original_adam), state_arrays(exported, exported_adam)):
        np.testing.assert_array_equal(before, after)
    loaded, metadata = load_snapshot(path)
    assert model_fingerprint(loaded) == model_fingerprint(original)
    assert metadata["result"] == actual
    x, y = snapshot_test_data(actual)
    np.testing.assert_array_equal(original(x).data, loaded(x).data)
    assert training.evaluate(loaded, x, y) == (actual["final"]["test_loss"], actual["final"]["test_accuracy"])


def test_cost_measurement_preserves_weights_masks_gradients_and_adam(tmp_path):
    result, model, optimizer = capture_training(mode="prune", method="saliency", sparsity=.75,
        epochs=8, seed=7, hidden_sizes=(8, 8), samples_per_class=20)
    before = state_arrays(model, optimizer)
    step = optimizer.step_count
    fingerprint = model_fingerprint(model)
    x, _ = snapshot_test_data(result)
    cost = measure_forward_costs(model, inputs=x, iterations=3, warmup_reps=1)
    assert cost["model_sha256"] == fingerprint
    assert cost["class_predictions_match"]
    assert cost["sparse_macs_per_sample"] == result["final"]["active_connections"]
    assert optimizer.step_count == step
    for old, new in zip(before, state_arrays(model, optimizer)):
        np.testing.assert_array_equal(old, new)


def test_snapshot_detects_parameter_tampering(tmp_path):
    path = tmp_path / "model.npz"
    training.train_baseline(epochs=1, hidden_sizes=(4,), snapshot_path=path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays["weight_0"][0, 0] += 1
    with path.open("wb") as output:
        np.savez_compressed(output, **arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        load_snapshot(path)


def test_custom_sweep_times_its_trained_artifact_and_reports_observed_values(tmp_path):
    assert main(["--seeds", "1", "--epochs", "2", "--hidden", "4", "--sparsities", "0", ".5",
                 "--output-dir", str(tmp_path), "--warmup", "0", "--reps", "2"]) == 0
    data = json.loads((tmp_path / "benchmark_results.json").read_text())
    cost = data["cost_benchmark"]
    source = cost["source_model"]
    loaded, metadata = load_snapshot(tmp_path / source["path"])
    assert cost["model_sha256"] == source["model_sha256"] == model_fingerprint(loaded)
    assert cost["test_accuracy"] == metadata["result"]["final"]["test_accuracy"]
    assert "442" not in data["falsifiable_claim"]
    assert "maintain test accuracy > 90%" not in data["falsifiable_claim"]
    assert "confidence intervals" in (tmp_path / "EVIDENCE.md").read_text(encoding="utf-8")
    matrix = run_cpu_benchmark(sparsities=[.5], batch_sizes=[1, 16], seed=42,
        warmup_reps=0, benchmark_reps=2, benchmark_results=tmp_path / "benchmark_results.json")
    assert all(row["model_sha256"] == source["model_sha256"] for row in matrix["results"])
    assert all(row["test_accuracy"] == cost["test_accuracy"] for row in matrix["results"])
