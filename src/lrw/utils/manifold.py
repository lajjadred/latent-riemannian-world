"""Manifold validation utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def manifold_assert_shape(z: Tensor, expected_dim: int | None = None) -> None:
    """
    Assert that a tensor has valid shape for use as a latent point.

    Parameters
    ----------
    z : Tensor
        Latent tensor to validate.
    expected_dim : int or None
        Expected latent dimension D. If None, only checks ndim.

    Raises
    ------
    ValueError
        If shape is invalid.
    """
    if z.ndim != 2:
        raise ValueError(
            f"Latent tensor must be 2D (B, D), got shape {tuple(z.shape)}"
        )
    if expected_dim is not None and z.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected latent dim {expected_dim}, got {z.shape[-1]}"
        )


def manifold_assert_metric(G: Tensor) -> None:
    """
    Assert that a tensor is a valid batch of metric matrices.

    Checks:
    - Shape is (B, D, D)
    - Matrices are symmetric
    - Matrices are positive definite (all eigenvalues > 0)

    Parameters
    ----------
    G : Tensor
        Metric tensor to validate. Shape (B, D, D).

    Raises
    ------
    ValueError
        If G is not a valid metric tensor.
    """
    if G.ndim != 3 or G.shape[-1] != G.shape[-2]:
        raise ValueError(
            f"Metric tensor must be (B, D, D), got shape {tuple(G.shape)}"
        )
    if not torch.allclose(G, G.mT, atol=1e-5):
        raise ValueError("Metric tensor must be symmetric.")

    eigenvalues = torch.linalg.eigvalsh(G)
    if not (eigenvalues > 0).all():
        raise ValueError("Metric tensor must be positive definite.")


def manifold_assert_tangent(z: Tensor, v: Tensor) -> None:
    """
    Assert that tangent vector v is compatible with base point z.

    Parameters
    ----------
    z : Tensor
        Base point. Shape (B, D).
    v : Tensor
        Tangent vector. Shape (B, D).

    Raises
    ------
    ValueError
        If shapes are incompatible.
    """
    if z.shape != v.shape:
        raise ValueError(
            f"Base point shape {tuple(z.shape)} and "
            f"tangent vector shape {tuple(v.shape)} must match."
        )
