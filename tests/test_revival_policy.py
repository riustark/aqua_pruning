"""Demonstrate permanent-pruning safety and the low-level Adam clock policy."""

import numpy as np
import pytest

from engine.tensor import Tensor
from nn.layers import Linear
from nn.optimizers import Adam


def test_rejected_layer_revival_is_atomic_and_training_stays_masked():
    layer = Linear(2, 1, np.random.default_rng(7))
    optimizer = Adam(layer.parameters())
    inputs = Tensor(np.array([[1.0, 2.0]]))
    for _ in range(5):
        optimizer.zero_grad()
        layer(inputs).sum().backward()
        optimizer.step()
    mask = np.array([[False], [True]])
    layer.apply_mask(mask)
    optimizer.register_mask(layer.weight, mask)

    def state():
        arrays = [layer.mask, layer.weight.data, layer.weight.grad,
                  layer.bias.data, layer.bias.grad, optimizer.param_masks[id(layer.weight)],
                  *optimizer.first_moments, *optimizer.second_moments, layer(inputs).data]
        return [x.copy() for x in arrays], optimizer.step_count

    before, step = state()
    with pytest.raises(ValueError, match="Regrowth is not supported"):
        layer.apply_mask(np.ones_like(mask))
    after, after_step = state()
    assert after_step == step
    for expected, actual in zip(before, after):
        np.testing.assert_array_equal(actual, expected)
    for _ in range(25):
        optimizer.zero_grad()
        layer(inputs).sum().backward()
        optimizer.step()
        for value in (layer.weight.data, layer.weight.grad,
                      optimizer.first_moments[0], optimizer.second_moments[0]):
            assert value[0, 0] == 0.0


@pytest.mark.parametrize("gradient", [1e-6, 0.25])
def test_low_level_prune_then_reenable_clears_history_and_keeps_global_clock(gradient):
    # This exercises Adam's mask API, not the Linear API or training regrowth.
    param = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    optimizer = Adam([param])
    for _ in range(10):
        param.grad = np.array([0.3, -0.2])
        optimizer.step()
    assert optimizer.first_moments[0][0] != 0
    optimizer.register_mask(param, np.array([False, True]))
    for _ in range(25):
        param.grad = np.array([0.0, -0.2])
        optimizer.step()
    for array in (param.data, optimizer.first_moments[0], optimizer.second_moments[0]):
        assert array[0] == 0.0
    survivor_m = optimizer.first_moments[0][1]
    survivor_v = optimizer.second_moments[0][1]
    optimizer.register_mask(param, np.array([True, True]))
    assert optimizer.first_moments[0][1] == survivor_m
    assert optimizer.second_moments[0][1] == survivor_v
    step = optimizer.step_count + 1
    expected_m = (1 - optimizer.beta1) * gradient
    expected_v = (1 - optimizer.beta2) * gradient**2
    expected_weight = -optimizer.learning_rate * (expected_m / (1 - optimizer.beta1**step)) / (
        np.sqrt(expected_v / (1 - optimizer.beta2**step)) + optimizer.epsilon)
    param.grad = np.array([gradient, -0.2])
    optimizer.step()
    assert optimizer.step_count == step
    np.testing.assert_allclose(optimizer.first_moments[0][0], expected_m, rtol=1e-14)
    np.testing.assert_allclose(optimizer.second_moments[0][0], expected_v, rtol=1e-14)
    np.testing.assert_allclose(param.data[0], expected_weight, rtol=1e-14)
