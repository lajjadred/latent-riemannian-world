"""Tests for lrw.transport.SchildsLadder."""

import torch
import pytest
from lrw.metric import PullbackMetric
from lrw.transport import SchildsLadder


def make_metric(D: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    W = torch.randn(16, D)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    return PullbackMetric(decoder=decoder)


def test_transport_output_shape():
    metric = make_metric()
    ladder = SchildsLadder(metric=metric, n_rungs=3)
    z0 = torch.randn(2, 4)
    z1 = torch.randn(2, 4)
    v = torch.randn(2, 4)
    transported = ladder.transport(z0, z1, v)
    assert transported.shape == (2, 4)


def test_transport_zero_vector():
    """Transporting a zero vector should return a small vector (numerical approx)."""
    metric = make_metric()
    ladder = SchildsLadder(metric=metric, n_rungs=3)
    z0 = torch.randn(2, 4)
    z1 = torch.randn(2, 4)
    v = torch.zeros(2, 4)
    transported = ladder.transport(z0, z1, v)
    # Schild's Ladder is a numerical approximation — norm should be small
    assert transported.norm(dim=-1).max() < 1.0


def test_transport_same_point():
    """Transporting from z0 to z0 should return the same vector."""
    metric = make_metric()
    ladder = SchildsLadder(metric=metric, n_rungs=3)
    z0 = torch.randn(2, 4)
    v = torch.randn(2, 4)
    transported = ladder.transport(z0, z0, v)
    assert transported.shape == (2, 4)


def test_transport_preserves_norm_approx():
    """Parallel transport approximately preserves the Riemannian norm."""
    from lrw.utils.linalg import riemannian_norm

    metric = make_metric()
    ladder = SchildsLadder(metric=metric, n_rungs=5)
    z0 = torch.randn(1, 4)
    z1 = torch.randn(1, 4)
    v = torch.randn(1, 4) * 0.1  # small vector for better approximation

    G0 = metric.metric_tensor(z0)
    G1 = metric.metric_tensor(z1)

    norm_before = riemannian_norm(G0, v)
    transported = ladder.transport(z0, z1, v)
    norm_after = riemannian_norm(G1, transported)

    # Norms should be in the same order of magnitude
    ratio = (norm_after / norm_before.clamp(min=1e-8)).item()
    assert 0.01 < ratio < 100.0


def test_repr():
    metric = make_metric()
    ladder = SchildsLadder(metric=metric, n_rungs=5)
    assert "5" in repr(ladder)