"""Tests for lrw.world.LatentStateSpace."""

import torch
import pytest
from lrw.metric import PullbackMetric
from lrw.world import LatentStateSpace


def make_model(D: int = 4, seed: int = 0, noise_scale: float = 0.0):
    torch.manual_seed(seed)
    W = torch.randn(16, D)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    metric = PullbackMetric(decoder=decoder)
    return LatentStateSpace(metric=metric, dt=0.1, noise_scale=noise_scale)


def test_step_output_shape():
    model = make_model()
    z = torch.randn(3, 4)
    v = torch.randn(3, 4)
    z_next, v_next = model.step(z, v)
    assert z_next.shape == (3, 4)
    assert v_next.shape == (3, 4)


def test_step_zero_velocity():
    """With zero velocity and no noise, state should not change much."""
    model = make_model()
    z = torch.randn(3, 4)
    v = torch.zeros(3, 4)
    z_next, v_next = model.step(z, v)
    assert torch.allclose(z_next, z, atol=1e-5)


def test_rollout_shape():
    model = make_model()
    z0 = torch.randn(2, 4)
    v0 = torch.randn(2, 4)
    states, velocities = model.rollout(z0, v0, n_steps=5)
    assert states.shape == (6, 2, 4)
    assert velocities.shape == (6, 2, 4)


def test_rollout_initial_state():
    """First state in rollout should equal z0."""
    model = make_model()
    z0 = torch.randn(2, 4)
    v0 = torch.randn(2, 4)
    states, _ = model.rollout(z0, v0, n_steps=5)
    assert torch.allclose(states[0], z0, atol=1e-6)


def test_log_transition_prob_shape():
    model = make_model()
    z = torch.randn(3, 4)
    z_next = torch.randn(3, 4)
    v = torch.randn(3, 4)
    log_p = model.log_transition_prob(z, z_next, v)
    assert log_p.shape == (3,)


def test_stochastic_velocity_differs():
    """With noise, repeated steps should produce different velocities."""
    model = make_model(noise_scale=0.1)
    z = torch.randn(1, 4)
    v = torch.randn(1, 4)
    _, v1 = model.step(z, v)
    _, v2 = model.step(z, v)
    assert not torch.allclose(v1, v2)


def test_repr():
    model = make_model()
    assert "0.1" in repr(model)
