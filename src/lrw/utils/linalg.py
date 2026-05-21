"""Linear algebra helpers for metric tensors."""

from __future__ import annotations

import torch
from torch import Tensor


def sym_inv(M: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Numerically stable inverse of a batch of symmetric positive definite matrices.

    Uses eigendecomposition: M = V diag(λ) V^T => M^{-1} = V diag(1/λ) V^T
    Eigenvalues below eps are clamped to avoid division by zero.

    Parameters
    ----------
    M : Tensor
        Shape (B, D, D).
    eps : float
        Minimum eigenvalue threshold.

    Returns
    -------
    Tensor
        Shape (B, D, D).
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = eigenvalues.clamp(min=eps)
    inv_ev = 1.0 / eigenvalues
    return eigenvectors @ torch.diag_embed(inv_ev) @ eigenvectors.mT


def sym_sqrt(M: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Matrix square root of a batch of SPD matrices via eigendecomposition.

    Parameters
    ----------
    M : Tensor
        Shape (B, D, D).

    Returns
    -------
    Tensor
        Shape (B, D, D).
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = eigenvalues.clamp(min=eps)
    return eigenvectors @ torch.diag_embed(eigenvalues.sqrt()) @ eigenvectors.mT


def riemannian_norm(G: Tensor, v: Tensor) -> Tensor:
    """
    Compute the Riemannian norm of tangent vector v under metric G.

    ||v||_G = sqrt(v^T G v)

    Parameters
    ----------
    G : Tensor
        Shape (B, D, D).
    v : Tensor
        Shape (B, D).

    Returns
    -------
    Tensor
        Shape (B,).
    """
    Gv = torch.bmm(G, v.unsqueeze(-1)).squeeze(-1)  # (B, D)
    return (v * Gv).sum(dim=-1).clamp(min=0.0).sqrt()
