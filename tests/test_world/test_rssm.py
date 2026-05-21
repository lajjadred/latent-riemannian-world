"""Tests for lrw.world.RiemannianRSSM."""

import torch
import pytest
from lrw.metric import PullbackMetric
from lrw.world import RiemannianRSSM


def make_rssm(
    latent_dim: int = 4,
    hidden_dim: int = 16,
    action_dim: int = 2,
    seed: int = 0,
):
    torch.manual_seed(seed)
    W = torch.randn(8, latent_dim)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    metric = PullbackMetric(decoder=decoder)
    return RiemannianRSSM(
        metric=metric,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )


def test_prior_shape():
    rssm = make_rssm()
    h = torch.randn(3, 16)
    mu, sigma = rssm.prior(h)
    assert mu.shape == (3, 4)
    assert sigma.shape == (3, 4)
    assert (sigma > 0).all()


def test_posterior_shape():
    rssm = make_rssm()
    h = torch.randn(3, 16)
    obs = torch.randn(3, 4)
    mu, sigma = rssm.posterior(h, obs)
    assert mu.shape == (3, 4)
    assert sigma.shape == (3, 4)


def test_riemannian_sample_shape():
    rssm = make_rssm()
    mu = torch.randn(3, 4)
    sigma = torch.ones(3, 4) * 0.1
    z = rssm.riemannian_sample(mu, sigma)
    assert z.shape == (3, 4)


def test_step_shapes():
    rssm = make_rssm()
    B = 2
    h = torch.zeros(B, 16)
    z = torch.randn(B, 4)
    action = torch.randn(B, 2)
    h_next, z_prior, mu, sigma = rssm.step(h, z, action)
    assert h_next.shape == (B, 16)
    assert z_prior.shape == (B, 4)
    assert mu.shape == (B, 4)
    assert sigma.shape == (B, 4)


def test_step_no_action():
    rssm = make_rssm()
    B = 2
    h = torch.zeros(B, 16)
    z = torch.randn(B, 4)
    h_next, z_prior, mu, sigma = rssm.step(h, z, action=None)
    assert h_next.shape == (B, 16)
    assert z_prior.shape == (B, 4)


def test_riemannian_kl_positive():
    rssm = make_rssm()
    B = 3
    mu_post = torch.randn(B, 4)
    sigma_post = torch.rand(B, 4) + 0.1
    mu_prior = torch.randn(B, 4)
    sigma_prior = torch.rand(B, 4) + 0.1
    kl = rssm.riemannian_kl(mu_post, sigma_post, mu_prior, sigma_prior)
    assert kl.shape == (B,)


def test_repr():
    rssm = make_rssm()
    assert "4" in repr(rssm)
    assert "16" in repr(rssm)
