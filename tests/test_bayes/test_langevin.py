"""Tests for lrw.bayes.RiemannianSGLD."""

import torch
import pytest
from lrw.metric import PullbackMetric
from lrw.bayes import RiemannianSGLD


def make_sgld(D: int = 4, seed: int = 0, noise_scale: float = 1.0):
    torch.manual_seed(seed)
    W = torch.randn(16, D)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    metric = PullbackMetric(decoder=decoder)

    def log_prob_fn(z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z ** 2).sum(dim=-1)

    return RiemannianSGLD(
        metric=metric,
        log_prob_fn=log_prob_fn,
        step_size=0.01,
        noise_scale=noise_scale,
    )


def test_step_shape():
    sgld = make_sgld()
    z = torch.randn(3, 4)
    z_new = sgld.step(z)
    assert z_new.shape == (3, 4)


def test_step_changes_state():
    sgld = make_sgld()
    z = torch.randn(3, 4)
    z_new = sgld.step(z)
    assert not torch.allclose(z, z_new)


def test_deterministic_no_noise():
    """With noise_scale=0, same input should give same output."""
    sgld = make_sgld(noise_scale=0.0)
    z = torch.randn(3, 4)
    z1 = sgld.step(z)
    z2 = sgld.step(z)
    assert torch.allclose(z1, z2)


def test_sample_shape():
    sgld = make_sgld()
    z0 = torch.randn(2, 4)
    samples = sgld.sample(z0, n_steps=10, burn_in=3)
    assert samples.shape == (7, 2, 4)


def test_repr():
    sgld = make_sgld()
    assert "0.01" in repr(sgld)
