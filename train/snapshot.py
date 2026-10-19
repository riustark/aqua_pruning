"""Inference-only snapshots of trained weights and masks; never resume training.

The result JSON preserves the configuration and measured accuracy of the exact
model. No optimizer state is loaded or updated by these helpers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from nn.layers import MLP


def model_fingerprint(model: MLP) -> str:
    """Stable identity for architecture, activation, weights, biases, and masks."""
    digest = hashlib.sha256()
    digest.update(json.dumps([model.layer_sizes, model.activation]).encode())
    for layer in model.layers:
        for value in (layer.weight.data, layer.bias.data, layer.mask):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def save_snapshot(model: MLP, result: dict, path: Path) -> None:
    """Copy final model arrays to disk without changing the live training state."""
    arrays = {}
    for index, layer in enumerate(model.layers):
        arrays[f"weight_{index}"] = layer.weight.data
        arrays[f"bias_{index}"] = layer.bias.data
        arrays[f"mask_{index}"] = layer.mask
    metadata = {"format_version": 1, "model_sha256": model_fingerprint(model),
                "result": result}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez_compressed(output, metadata=np.array(json.dumps(metadata)), **arrays)


def load_snapshot(path: Path) -> tuple[MLP, dict]:
    """Load a separate inference model, preserving masks exactly, with no pickle."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
        if metadata["format_version"] != 1:
            raise ValueError("Unsupported model snapshot version.")
        config = metadata["result"]["config"]
        # A local generator avoids consuming any global/training RNG state.
        model = MLP(config["layer_sizes"], np.random.default_rng(0),
                    activation=config["activation"])
        for index, layer in enumerate(model.layers):
            weight, bias, mask = (data[f"{name}_{index}"] for name in ("weight", "bias", "mask"))
            if weight.shape != layer.weight.shape or bias.shape != layer.bias.shape:
                raise ValueError("Snapshot parameter shape does not match architecture.")
            if mask.shape != weight.shape or mask.dtype != np.bool_:
                raise ValueError("Snapshot mask must be boolean and match weight shape.")
            if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
                raise ValueError("Snapshot parameters must be finite.")
            if np.any(weight[~mask] != 0):
                raise ValueError("Snapshot contains nonzero permanently pruned weights.")
            layer.weight.data = weight.copy()
            layer.bias.data = bias.copy()
            layer.mask = mask.copy()
    if model_fingerprint(model) != metadata["model_sha256"]:
        raise ValueError("Snapshot model fingerprint mismatch.")
    return model, metadata


def snapshot_test_data(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the held-out split from the recorded standard training config."""
    from train.dataset import make_spirals, train_test_split

    config = result["config"]
    data_seq, split_seq, _, _ = np.random.SeedSequence(config["seed"]).spawn(4)
    features, labels = make_spirals(
        config["samples_per_class"], config["num_classes"], config["noise"],
        seed=int(data_seq.generate_state(1)[0]),
    )
    _, test_x, _, test_y = train_test_split(
        features, labels, config["test_fraction"], seed=int(split_seq.generate_state(1)[0])
    )
    return test_x, test_y
