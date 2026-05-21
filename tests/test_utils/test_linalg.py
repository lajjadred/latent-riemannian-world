"""Tests for lrw.utils.linalg."""

import torch
import pytest
from lrw.utils import sym_inv, sym_sqrt, riemannian_norm


def make_spd(B: int, D: int, seed: int = 0) -> torch.Tensor:
    """Create a random symmetric positive definite matrix."""
    torch.manual_seed(seed)
    A = torch.randn(B, D, D)
    return A @ A.mT + torch.eye(D).unsqueeze(0) * 0.5


def test_sym_inv_shape():
    M = make_spd(3, 4)
    assert sym_inv(M).shape == (3, 4, 4)


def test_sym_inv_correctness():
    M = make_spd(3, 5)
    M_inv = sym_inv(M)
    product = torch.bmm(M, M_inv)
    eye = torch.eye(5).unsqueeze(0).expand(3, -1, -1)
    assert torch.allclose(product, eye, atol=1e-4)


def test_sym_sqrt_shape():
    M = make_spd(2, 4)
    assert sym_sqrt(M).shape == (2, 4, 4)


def test_sym_sqrt_correctness():
    M = make_spd(2, 4)
    S = sym_sqrt(M)
    M_reconstructed = torch.bmm(S, S)
    assert torch.allclose(M_reconstructed, M, atol=1e-4)


def test_riemannian_norm_positive():
    G = make_spd(4, 3)
    v = torch.randn(4, 3)
    d = riemannian_norm(G, v)
    assert (d >= 0).all()


def test_riemannian_norm_zero():
    G = make_spd(2, 3)
    v = torch.zeros(2, 3)
    d = riemannian_norm(G, v)
    assert torch.allclose(d, torch.zeros(2), atol=1e-6)
