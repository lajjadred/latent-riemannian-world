"""Tests for lrw.bayes.SVGD."""

import torch
import pytest
from lrw.metric import PullbackMetric
from lrw.bayes import SVGD


def make_svgd(D: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    W = torch.randn(16, D)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    metric = PullbackMetric(decoder=decoder)

    def log_prob_fn(z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z ** 2).sum(dim=-1)

    return SVGD(metric=metric, log_prob_fn=log_prob_fn, step_size=0.01)


def test_step_shape():
    svgd = make_svgd()
    z = torch.randn(10, 4)
    z_new = svgd.step(z)
    assert z_new.shape == (10, 4)


def test_step_changes_particles():
    svgd = make_svgd()
    z = torch.randn(10, 4)
    z_new = svgd.step(z)
    assert not torch.allclose(z, z_new)


def test_run_shape():
    svgd = make_svgd()
    z = torch.randn(8, 4)
    z_final = svgd.run(z, n_steps=5)
    assert z_final.shape == (8, 4)


def test_repr():
    svgd = make_svgd()
    assert "0.01" in repr(svgd)
