"""Deterministic synthetic classification data (interleaved spirals)."""

from __future__ import annotations

from typing import Iterator

import numpy as np


def make_spirals(
    samples_per_class: int = 200,
    num_classes: int = 3,
    noise: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``num_classes`` interleaved 2-D spiral arms.

    Each class ``c`` places points at radius ``r`` in ``[0, 1]`` and angle
    ``theta = 4c + 4r + noise``, so the arms wind around each other and no
    straight line can separate them (a linear model scores near chance).

    Args:
        samples_per_class: Points per class (> 0).
        num_classes: Number of spiral arms / classes (>= 2).
        noise: Standard deviation of Gaussian noise added to the angle (>= 0).
        seed: Seed for ``np.random.default_rng``.

    Returns:
        ``features`` float64 of shape ``(samples_per_class * num_classes, 2)``
        and ``labels`` int64 of shape ``(samples_per_class * num_classes,)``,
        ordered class by class (shuffle with ``train_test_split``).
    """
    if samples_per_class <= 0:
        raise ValueError(f"samples_per_class must be positive, got {samples_per_class}.")
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}.")
    if noise < 0.0:
        raise ValueError(f"noise must be non-negative, got {noise}.")

    rng = np.random.default_rng(seed)
    radius = np.linspace(0.0, 1.0, samples_per_class)
    features = np.empty((samples_per_class * num_classes, 2), dtype=np.float64)
    labels = np.empty(samples_per_class * num_classes, dtype=np.int64)
    for class_index in range(num_classes):
        start = class_index * samples_per_class
        stop = start + samples_per_class
        angle = 4.0 * class_index + 4.0 * radius + rng.normal(0.0, noise, samples_per_class)
        features[start:stop, 0] = radius * np.sin(angle)
        features[start:stop, 1] = radius * np.cos(angle)
        labels[start:stop] = class_index
    return features, labels


def train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle deterministically and split into train and test sets.

    The test set size is ``round(N * test_fraction)``, kept within ``[1, N-1]``.

    Returns:
        ``(train_features, test_features, train_labels, test_labels)``.
    """
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError(
            f"Expected features (N, D) and labels (N,), got {features.shape} and {labels.shape}."
        )
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}.")
    num_samples = len(labels)
    if num_samples < 2:
        raise ValueError("Need at least 2 samples to split.")

    order = np.random.default_rng(seed).permutation(num_samples)
    num_test = min(max(int(round(num_samples * test_fraction)), 1), num_samples - 1)
    test_index, train_index = order[:num_test], order[num_test:]
    return features[train_index], features[test_index], labels[train_index], labels[test_index]


def iterate_minibatches(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(batch_features, batch_labels)`` covering every sample exactly once.

    Args:
        features: Array of shape ``(N, D)``.
        labels: Array of shape ``(N,)``.
        batch_size: Maximum samples per batch; the last batch may be smaller.
        rng: If given, samples are shuffled with it; otherwise original order.
    """
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    num_samples = len(labels)
    order = rng.permutation(num_samples) if rng is not None else np.arange(num_samples)
    for start in range(0, num_samples, batch_size):
        batch_index = order[start:start + batch_size]
        yield features[batch_index], labels[batch_index]
