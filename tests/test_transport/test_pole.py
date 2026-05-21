"""Tests for lrw.transport.PoleLadder."""

import torch
import pytest
from lrw.metric import PullbackMetric
from lrw.transport import PoleLadder


def make_metric(D: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    W = torch.randn(16, D)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    return PullbackMetric(decoder=decoder)


def test_transport_output_shape():
    metric = make_metric()
    ladder = PoleLadder(metric=metric, n_rungs=3)
    z0 = torch.randn(2, 4)
    z1 = torch.randn(2, 4)
    v = torch.randn(2, 4)
    transported = ladder.transport(z0, z1, v)
    assert transported.shape == (2, 4)


def test_transport_zero_vector():
    metric = make_metric()
    ladder = PoleLadder(metric=metric, n_rungs=3)
    z0 = torch.randn(2, 4)
    z1 = torch.randn(2, 4)
    v = torch.zeros(2, 4)
    transported = ladder.transport(z0, z1, v)
    assert transported.norm(dim=-1).max() < 1.0


def test_transport_same_point():
    metric = make_metric()
    ladder = PoleLadder(metric=metric, n_rungs=3)
    z0 = torch.randn(2, 4)
    v = torch.randn(2, 4)
    transported = ladder.transport(z0, z0, v)
    assert transported.shape == (2, 4)


def test_repr():
    metric = make_metric()
    ladder = PoleLadder(metric=metric, n_rungs=5)
    assert "5" in repr(ladder)
