"""Tests for lrw.metric.BayesianMetric."""

import torch
import pytest
from lrw.metric import BayesianMetric


def make_ensemble(n: int = 4, M: int = 16, D: int = 4, seed: int = 0):
    """Create an ensemble of random linear decoders."""
    decoders = []
    for i in range(n):
        torch.manual_seed(seed + i)
        W = torch.randn(M, D)

        def decoder(z: torch.Tensor, W: torch.Tensor = W) -> torch.Tensor:
            return z @ W.T

        decoders.append(decoder)
    return decoders


def test_metric_shape():
    ensemble = make_ensemble()
    metric = BayesianMetric(decoder_ensemble=ensemble)
    z = torch.randn(3, 4)
    G = metric.metric_tensor(z)
    assert G.shape == (3, 4, 4)


def test_metric_symmetric():
    ensemble = make_ensemble()
    metric = BayesianMetric(decoder_ensemble=ensemble)
    z = torch.randn(4, 4)
    G = metric.metric_tensor(z)
    assert torch.allclose(G, G.mT, atol=1e-5)


def test_metric_positive_definite():
    ensemble = make_ensemble()
    metric = BayesianMetric(decoder_ensemble=ensemble)
    z = torch.randn(4, 4)
    G = metric.metric_tensor(z)
    eigenvalues = torch.linalg.eigvalsh(G)
    assert (eigenvalues > 0).all()


def test_empty_ensemble_raises():
    with pytest.raises(ValueError):
        BayesianMetric(decoder_ensemble=[])


def test_metric_variance_shape():
    ensemble = make_ensemble()
    metric = BayesianMetric(decoder_ensemble=ensemble)
    z = torch.randn(3, 4)
    metric.metric_tensor(z)
    var = metric.metric_variance(z)
    assert var.shape == (3, 4, 4)


def test_metric_variance_nonnegative():
    ensemble = make_ensemble()
    metric = BayesianMetric(decoder_ensemble=ensemble)
    z = torch.randn(3, 4)
    metric.metric_tensor(z)
    var = metric.metric_variance(z)
    assert (var >= 0).all()


def test_single_member_equals_pullback():
    """With one decoder, BayesianMetric should equal PullbackMetric."""
    from lrw.metric import PullbackMetric

    torch.manual_seed(0)
    W = torch.randn(16, 4)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    reg = 1e-5
    bayes = BayesianMetric(decoder_ensemble=[decoder], regularization=reg)
    pull = PullbackMetric(decoder=decoder, regularization=reg)

    z = torch.randn(3, 4)
    assert torch.allclose(bayes.metric_tensor(z), pull.metric_tensor(z), atol=1e-5)
