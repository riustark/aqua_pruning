"""Pruning schedule and connection selection targeting exact sparsity budgets.

Functions:
- ``target_sparsity_at_step``: Calculates target sparsity at step/epoch ``t``
  using a cubic polynomial schedule.
- ``prune_to_sparsity``: Selects the lowest-scoring active connections across
  all prunable layers, updates masks, and resets Adam moments.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nn.layers import MLP
from nn.optimizers import Adam


def target_sparsity_at_step(
    step: int,
    final_sparsity: float,
    start_step: int,
    end_step: int,
) -> float:
    """Compute the target sparsity at ``step`` using a cubic ramp.

    The schedule is defined by:

    * For ``step < start_step``: ``0.0`` (warm-up / dense training).
    * For ``step >= end_step``: ``final_sparsity`` (target reached).
    * For ``start_step <= step < end_step``:

      .. math::
          s(t) = s_f \\cdot \\left(1 - \\left(1 - \\frac{t - t_0}{t_1 - t_0}\\right)^3\\right)

    This cubic curve starts gently, accelerates pruning in the middle, and
    decelerates as it approaches the final target.

    Args:
        step: Current step or epoch index (1-based or 0-based, must match start/end).
        final_sparsity: Desired final sparsity fraction in ``[0, 1)``.
        start_step: Step at which pruning begins.
        end_step: Step at which the final target sparsity is reached.

    Returns:
        Target sparsity at this step in ``[0, final_sparsity]``.

    Raises:
        ValueError: If arguments violate range constraints.
    """
    if not (0.0 <= final_sparsity < 1.0):
        raise ValueError(f"final_sparsity must be in [0, 1), got {final_sparsity}.")
    if start_step > end_step:
        raise ValueError(f"start_step ({start_step}) must be <= end_step ({end_step}).")

    if step < start_step:
        return 0.0
    if step >= end_step or start_step == end_step:
        return float(final_sparsity)

    progress = (step - start_step) / (end_step - start_step)
    cubic_progress = 1.0 - (1.0 - progress) ** 3
    return float(final_sparsity * cubic_progress)


def prune_to_sparsity(
    model: MLP,
    optimizer: Adam,
    scores: Sequence[np.ndarray],
    target_sparsity: float,
) -> tuple[int, int]:
    """Prune lowest-scoring active connections to reach ``target_sparsity``.

    Only **currently active** prunable weight connections are compared. Biases
    are unpruned and strictly excluded from the connection count and denominator.

    The integer count of connections to prune is calculated as:
    ``prune_count = int(round(total_prunable * target_sparsity))``
    reaching the exact feasible integer connection count.

    Args:
        model: MLP network containing Linear layers.
        optimizer: Adam optimizer tracking model parameters.
        scores: Sequence of 2D score arrays matching ``model.layers``.
        target_sparsity: Desired fraction of total connections to prune in ``[0, 1)``.

    Returns:
        ``(active_connections, total_prunable)``.

    Raises:
        ValueError: If scores length or shapes do not match model layers.
    """
    if len(scores) != len(model.layers):
        raise ValueError(
            f"Expected {len(model.layers)} score arrays, got {len(scores)}."
        )
    for index, (layer, score_array) in enumerate(zip(model.layers, scores)):
        if score_array.shape != layer.mask.shape:
            raise ValueError(
                f"Layer {index} score shape {score_array.shape} != mask shape {layer.mask.shape}."
            )

    total_prunable = sum(layer.mask.size for layer in model.layers)
    target_pruned = min(max(int(round(total_prunable * target_sparsity)), 0), total_prunable - 1)
    target_active = total_prunable - target_pruned
    current_active = sum(layer.num_active() for layer in model.layers)

    num_to_prune = current_active - target_active
    if num_to_prune <= 0:
        # Already at or exceeding target sparsity; no new pruning needed
        return current_active, total_prunable

    # Collect all currently active connections with their global coordinates
    # Format: (score, layer_idx, row, col)
    active_candidates: list[tuple[float, int, int, int]] = []
    for layer_idx, (layer, score_array) in enumerate(zip(model.layers, scores)):
        active_indices = np.argwhere(layer.mask)
        for row, col in active_indices:
            active_candidates.append(
                (float(score_array[row, col]), layer_idx, int(row), int(col))
            )

    # Sort ascending: lowest importance score pruned first
    active_candidates.sort(key=lambda item: item[0])

    # Copy current masks
    new_masks = [layer.mask.copy() for layer in model.layers]

    # Mark the lowest-scoring num_to_prune active connections as pruned
    for _, layer_idx, row, col in active_candidates[:num_to_prune]:
        new_masks[layer_idx][row, col] = False

    # Apply masks to model layers (enforces no-regrowth and zeroes raw weight data)
    for layer, new_mask in zip(model.layers, new_masks):
        layer.apply_mask(new_mask)

    # Reset Adam optimizer states (raw weight, grad, moments m and v) for pruned connections
    param_masks = {id(layer.weight): layer.mask for layer in model.layers}
    optimizer.reset_pruned_state(param_masks)

    achieved_active = sum(layer.num_active() for layer in model.layers)
    return achieved_active, total_prunable
