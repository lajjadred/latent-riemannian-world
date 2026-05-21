"""Tests for lrw.metric.PullbackMetric."""

import torch
import pytest
from lrw.metric import PullbackMetric


def make_linear_decoder(M: int = 16, D: int = 4, seed: int = 0):
    """Simple linear decoder: z (B, D) -> x (B, M)."""
    torch.manual_seed(seed)
    W = torch.randn(M, D)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    return decoder, W


def test_metric_shape():
    decoder, _ = make_linear_decoder()
    metric = PullbackMetric(decoder=decoder)
    z = torch.randn(3, 4)
    G = metric.metric_tensor(z)
    assert G.shape == (3, 4, 4)


def test_metric_symmetric():
    decoder, _ = make_linear_decoder()
    metric = PullbackMetric(decoder=decoder)
    z = torch.randn(5, 4)
    G = metric.metric_tensor(z)
    assert torch.allclose(G, G.mT, atol=1e-5)


def test_metric_positive_definite():
    decoder, _ = make_linear_decoder()
    metric = PullbackMetric(decoder=decoder)
    z = torch.randn(5, 4)
    G = metric.metric_tensor(z)
    eigenvalues = torch.linalg.eigvalsh(G)
    assert (eigenvalues > 0).all()


def test_linear_decoder_exact():
    """For f(z) = Wz, G = W^T W + reg * I."""
    W = torch.eye(4) * 2.0

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    reg = 1e-5
    metric = PullbackMetric(decoder=decoder, regularization=reg)
    z = torch.randn(2, 4)
    G = metric.metric_tensor(z)

    expected = W.T @ W + torch.eye(4) * reg
    for b in range(2):
        assert torch.allclose(G[b], expected, atol=1e-4)


def test_volume_element_positive():
    decoder, _ = make_linear_decoder()
    metric = PullbackMetric(decoder=decoder)
    z = torch.randn(4, 4)
    vol = metric.local_volume_element(z)
    assert (vol > 0).all()
