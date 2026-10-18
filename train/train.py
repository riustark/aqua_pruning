"""Training entry point: coordinate forward, loss, backward, optimizer, scheduled pruning, and CLI.

Commands supported:

    python -m train.train --mode baseline --epochs 100 --seed 42
    python -m train.train --mode prune --method saliency --sparsity 0.90 --epochs 100 --seed 42

Prune mode uses gradual magnitude or saliency pruning on a cubic schedule:
no pruning before ``prune_start_fraction`` of the epochs, the target sparsity
is reached at ``prune_end_fraction``, and the remaining epochs fine-tune the
surviving connections with the mask fixed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from engine import operations as ops
from engine.tensor import Tensor
from nn.layers import MLP
from nn.optimizers import Adam
from prune.importance import METHODS, compute_layer_scores
from prune.scheduler import prune_to_sparsity, target_sparsity_at_step
from train.dataset import iterate_minibatches, make_spirals, train_test_split

DEFAULT_HIDDEN_SIZES = (64, 64)


def evaluate(model: MLP, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return ``(mean loss, accuracy)`` of ``model`` on a dataset (no update).

    Accuracy is the fraction of rows whose highest logit equals the true label.
    """
    logits = model(Tensor(features))
    loss = ops.softmax_cross_entropy(logits, labels).item()
    accuracy = float(np.mean(np.argmax(logits.data, axis=1) == labels))
    return loss, accuracy


def train_one_epoch(
    model: MLP,
    optimizer: Adam,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """Run one epoch over the shuffled training data; return the mean batch loss.

    Every batch builds a fresh graph: zero grads -> forward -> loss -> backward
    -> optimizer step.

    Raises:
        FloatingPointError: If any batch loss becomes NaN or infinite.
    """
    batch_losses = []
    for batch_features, batch_labels in iterate_minibatches(features, labels, batch_size, rng):
        optimizer.zero_grad()
        loss = ops.softmax_cross_entropy(model(Tensor(batch_features)), batch_labels)
        loss_value = loss.item()
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss: {loss_value}.")
        loss.backward()
        optimizer.step()
        batch_losses.append(loss_value)
    return float(np.mean(batch_losses))


def prune_epoch_range(
    epochs: int, start_fraction: float, end_fraction: float
) -> tuple[int, int]:
    """Convert fractions of training into 1-based epoch numbers ``(start, end)``.

    ``1 <= start <= end <= epochs`` always holds, so the final target is
    always reached.
    """
    if not (0.0 <= start_fraction <= end_fraction <= 1.0):
        raise ValueError(
            f"Need 0 <= start ({start_fraction}) <= end ({end_fraction}) <= 1."
        )
    start = min(max(1, int(round(start_fraction * epochs))), epochs)
    end = min(max(start, int(round(end_fraction * epochs))), epochs)
    return start, end


def train_model(
    mode: str = "baseline",
    method: str = "magnitude",
    sparsity: float = 0.0,
    epochs: int = 100,
    seed: int = 42,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    samples_per_class: int = 200,
    num_classes: int = 3,
    noise: float = 0.2,
    test_fraction: float = 0.2,
    prune_start_fraction: float = 0.1,
    prune_end_fraction: float = 0.6,
) -> dict:
    """Train an MLP on spirals (in either dense baseline or gradual prune mode).

    One ``seed`` controls everything: it is split by ``np.random.SeedSequence``
    into independent streams for data generation, the train/test split, weight
    initialization, and batch shuffling. A baseline and a pruned run with the
    same seed share identical data, initial weights, and batch order.

    In ``prune`` mode, after each epoch in ``[start, end]`` the target is
    ``target_sparsity_at_step(epoch, sparsity, start, end)``; scores are computed
    on the training set and lowest-scoring active connections are removed.

    Returns:
        Dict with keys ``config``, ``history``, ``initial``, and ``final`` metrics.
    """
    if mode not in ("baseline", "prune"):
        raise ValueError(f"mode must be 'baseline' or 'prune', got '{mode}'.")
    if mode == "prune":
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}', expected one of {METHODS}.")
        if not (0.0 <= sparsity < 1.0):
            raise ValueError(f"sparsity must be in [0, 1), got {sparsity}.")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")

    data_seq, split_seq, init_seq, shuffle_seq = np.random.SeedSequence(seed).spawn(4)
    features, labels = make_spirals(
        samples_per_class, num_classes, noise, seed=int(data_seq.generate_state(1)[0])
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_fraction, seed=int(split_seq.generate_state(1)[0])
    )

    layer_sizes = [features.shape[1], *hidden_sizes, num_classes]
    model = MLP(layer_sizes, np.random.default_rng(init_seq), activation="relu")
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    shuffle_rng = np.random.default_rng(shuffle_seq)
    prune_start, prune_end = prune_epoch_range(epochs, prune_start_fraction, prune_end_fraction)

    initial_train_loss, initial_train_acc = evaluate(model, train_x, train_y)
    initial_test_loss, initial_test_acc = evaluate(model, test_x, test_y)
    history: dict[str, list[float]] = {
        "train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": [],
        "sparsity": [],
    }

    for epoch in range(1, epochs + 1):
        history["train_loss"].append(
            train_one_epoch(model, optimizer, train_x, train_y, batch_size, shuffle_rng)
        )

        # Scheduled pruning step
        if mode == "prune" and prune_start <= epoch <= prune_end:
            step_target = target_sparsity_at_step(epoch, sparsity, prune_start, prune_end)
            scores = compute_layer_scores(model, method, train_x, train_y)
            prune_to_sparsity(model, optimizer, scores, step_target)

        _, train_acc = evaluate(model, train_x, train_y)
        test_loss, test_acc = evaluate(model, test_x, test_y)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        history["sparsity"].append(model.sparsity()[0])

    final_train_loss, final_train_acc = evaluate(model, train_x, train_y)
    achieved_sparsity, active_conn, total_prunable = model.sparsity()
    return {
        "config": {
            "mode": mode, "method": method if mode == "prune" else None,
            "target_sparsity": sparsity if mode == "prune" else 0.0,
            "epochs": epochs, "seed": seed,
            "layer_sizes": layer_sizes, "activation": "relu",
            "learning_rate": learning_rate, "batch_size": batch_size,
            "samples_per_class": samples_per_class, "num_classes": num_classes,
            "noise": noise, "test_fraction": test_fraction,
            "prune_start_epoch": prune_start if mode == "prune" else None,
            "prune_end_epoch": prune_end if mode == "prune" else None,
            "num_train": int(len(train_y)), "num_test": int(len(test_y)),
            "python": platform.python_version(), "numpy": np.__version__,
        },
        "history": history,
        "initial": {
            "train_loss": initial_train_loss, "train_accuracy": initial_train_acc,
            "test_loss": initial_test_loss, "test_accuracy": initial_test_acc,
        },
        "final": {
            "train_loss": final_train_loss, "train_accuracy": final_train_acc,
            "test_loss": history["test_loss"][-1], "test_accuracy": history["test_accuracy"][-1],
            "sparsity": achieved_sparsity,
            "active_connections": active_conn,
            "total_prunable": total_prunable,
        },
    }


def train_baseline(epochs: int = 100, seed: int = 42, **kwargs) -> dict:
    """Train a dense MLP without pruning. See ``train_model`` for options."""
    return train_model(mode="baseline", epochs=epochs, seed=seed, **kwargs)


def train_pruned(
    method: str, sparsity: float, epochs: int = 100, seed: int = 42, **kwargs
) -> dict:
    """Train with gradual ``method`` pruning to ``sparsity``. See ``train_model``."""
    return train_model(
        mode="prune", method=method, sparsity=sparsity, epochs=epochs, seed=seed, **kwargs
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Train an MLP on synthetic spirals.")
    parser.add_argument("--mode", choices=["baseline", "prune"], default="baseline",
                        help="Training mode: baseline (dense) or prune (self-pruning).")
    parser.add_argument("--method", choices=list(METHODS), default="magnitude",
                        help="Importance score method: magnitude or saliency (prune mode only).")
    parser.add_argument("--sparsity", type=float, default=0.90,
                        help="Target fraction of prunable connections to remove.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(DEFAULT_HIDDEN_SIZES),
                        help="Hidden layer sizes, e.g. --hidden 64 64")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON file for the full result (e.g. results/run.json).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints a progress summary and returns an exit code."""
    args = parse_args(argv)
    result = train_model(
        mode=args.mode,
        method=args.method,
        sparsity=args.sparsity if args.mode == "prune" else 0.0,
        epochs=args.epochs,
        seed=args.seed,
        hidden_sizes=args.hidden,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    config, history = result["config"], result["history"]
    initial, final = result["initial"], result["final"]

    mode_label = args.mode if args.mode == "baseline" else (
        f"prune ({args.method}, target sparsity {args.sparsity:.2f}, "
        f"epochs {config['prune_start_epoch']}-{config['prune_end_epoch']})"
    )
    print(f"Mode: {mode_label} | layers: {config['layer_sizes']} | seed: {args.seed}")
    print(f"Before training: train loss {initial['train_loss']:.4f}, "
          f"test acc {initial['test_accuracy']:.3f}")
    report_every = max(1, args.epochs // 10)
    for epoch in range(args.epochs):
        if (epoch + 1) % report_every == 0 or epoch == 0:
            print(f"epoch {epoch + 1:4d} | train loss {history['train_loss'][epoch]:.4f} | "
                  f"train acc {history['train_accuracy'][epoch]:.3f} | "
                  f"test loss {history['test_loss'][epoch]:.4f} | "
                  f"test acc {history['test_accuracy'][epoch]:.3f} | "
                  f"sparsity {history['sparsity'][epoch]:.4f}")
    print(f"Final: train acc {final['train_accuracy']:.3f}, test acc {final['test_accuracy']:.3f} "
          f"(chance = {1.0 / config['num_classes']:.3f})")
    print(f"Sparsity: {final['sparsity']:.4f} | active connections "
          f"{final['active_connections']}/{final['total_prunable']} (biases excluded)")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved result to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
