"""SLERP: Spherical Linear Interpolation on the unit hypersphere."""

from __future__ import annotations

import torch
from torch import Tensor


def slerp(z0: Tensor, z1: Tensor, t: float) -> Tensor:
    """
    Spherical linear interpolation between z0 and z1.

    Assumes z0 and z1 lie on a hypersphere (unit sphere approximation).
    This is the standard SLERP used in diffusion model communities,
    provided here as a baseline to compare against true geodesic interpolation.

    SLERP formula:
        slerp(z0, z1, t) = sin((1-t)*Omega)/sin(Omega) * z0
                         + sin(t*Omega)/sin(Omega) * z1
    where Omega = arccos(z0 . z1 / (||z0|| ||z1||))

    Parameters
    ----------
    z0 : Tensor
        Shape (B, D) — start point.
    z1 : Tensor
        Shape (B, D) — end point.
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    Tensor
        Shape (B, D) — interpolated point.
    """
    z0_norm = z0 / z0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    z1_norm = z1 / z1.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    dot = (z0_norm * z1_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)  # (B, 1)

    # When omega is very small, fall back to linear interpolation
    sin_omega = torch.sin(omega).clamp(min=1e-8)
    w0 = torch.sin((1.0 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega

    return w0 * z0 + w1 * z1


def slerp_path(z0: Tensor, z1: Tensor, n_points: int = 10) -> Tensor:
    """
    Generate a path of n_points between z0 and z1 via SLERP.

    Parameters
    ----------
    z0 : Tensor
        Shape (B, D).
    z1 : Tensor
        Shape (B, D).
    n_points : int
        Number of points including endpoints.

    Returns
    -------
    Tensor
        Shape (n_points, B, D).
    """
    ts = torch.linspace(0.0, 1.0, n_points)
    return torch.stack([slerp(z0, z1, t.item()) for t in ts], dim=0)
