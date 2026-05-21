"""Tests for lrw.metric.FisherMetric."""

import torch
import pytest
from lrw.metric import FisherMetric


def make_gaussian_decoder(D: int = 4, M: int = 8, seed: int = 0):
    """
    Gaussian decoder: p(x|z) = N(x; Wz, I)
    log p(x|z) = -0.5 * ||x - Wz||^2 + const
    score = grad_z log p = W^T (x - Wz)
    """
    torch.manual_seed(seed)
    W = torch.randn(M, D)

    def log_prob_fn(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mean = z @ W.T                          # (B, M)
        return -0.5 * ((x - mean) ** 2).sum(dim=-1)

    def sample_fn(z: torch.Tensor, n: int) -> torch.Tensor:
        B = z.shape[0]
        mean = z @ W.T                          # (B, M)
        samples = mean.unsqueeze(1) + torch.randn(B, n, mean.shape[-1])
        return samples

    return log_prob_fn, sample_fn, W


def test_metric_shape():
    log_prob_fn, sample_fn, _ = make_gaussian_decoder()
    metric = FisherMetric(log_prob_fn=log_prob_fn, sample_fn=sample_fn)
    z = torch.randn(3, 4)
    G = metric.metric_tensor(z)
    assert G.shape == (3, 4, 4)


def test_metric_symmetric():
    log_prob_fn, sample_fn, _ = make_gaussian_decoder()
    metric = FisherMetric(log_prob_fn=log_prob_fn, sample_fn=sample_fn)
    z = torch.randn(3, 4)
    G = metric.metric_tensor(z)
    assert torch.allclose(G, G.mT, atol=1e-5)


def test_metric_positive_definite():
    log_prob_fn, sample_fn, _ = make_gaussian_decoder()
    metric = FisherMetric(log_prob_fn=log_prob_fn, sample_fn=sample_fn, n_samples=32)
    z = torch.randn(3, 4)
    G = metric.metric_tensor(z)
    eigenvalues = torch.linalg.eigvalsh(G)
    assert (eigenvalues > 0).all()


def test_repr():
    log_prob_fn, sample_fn, _ = make_gaussian_decoder()
    metric = FisherMetric(log_prob_fn=log_prob_fn, sample_fn=sample_fn, n_samples=8)
    assert "8" in repr(metric)
