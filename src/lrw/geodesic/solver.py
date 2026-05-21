"""Geodesic solver via shooting method (Euler integration)."""

from __future__ import annotations

import torch
from torch import Tensor

from lrw.metric.base import RiemannianMetric


class GeodesicSolver:
    """
    Solves the geodesic equation on a Riemannian manifold via shooting method.

    The geodesic equation is:
        gamma''(t) + Gamma(gamma(t)) * gamma'(t) * gamma'(t) = 0

    Given start point z0 and end point z1, finds the initial velocity v0
    such that the geodesic starting at z0 with velocity v0 reaches z1.

    Parameters
    ----------
    metric : RiemannianMetric
        The Riemannian metric defining the manifold geometry.
    n_steps : int
        Number of integration steps along the geodesic.
    step_size : float
        Step size for Euler integration.
    """

    def __init__(
        self,
        metric: RiemannianMetric,
        n_steps: int = 100,
        step_size: float = 0.01,
    ) -> None:
        self.metric = metric
        self.n_steps = n_steps
        self.step_size = step_size

    def shoot(self, z0: Tensor, v0: Tensor) -> Tensor:
        """
        Integrate the geodesic equation starting from z0 with velocity v0.

        Uses simple Euler integration:
            z_{t+1} = z_t + dt * v_t
            v_{t+1} = v_t + dt * a_t   where a_t = -Gamma(z_t) v_t v_t

        Parameters
        ----------
        z0 : Tensor
            Shape (B, D) — starting position.
        v0 : Tensor
            Shape (B, D) — initial velocity (tangent vector at z0).

        Returns
        -------
        Tensor
            Shape (B, D) — endpoint after integration.
        """
        z = z0.clone()
        v = v0.clone()

        for _ in range(self.n_steps):
            a = self.metric.geodesic_acceleration(z, v)  # (B, D)
            z = z + self.step_size * v
            v = v + self.step_size * a

        return z

    def interpolate(self, z0: Tensor, z1: Tensor, n_points: int = 10) -> Tensor:
        """
        Interpolate between z0 and z1 along the geodesic.

        Uses linear initial velocity v0 = (z1 - z0) as a first approximation.
        For exact geodesics, use solve() instead.

        Parameters
        ----------
        z0 : Tensor
            Shape (B, D) — start point.
        z1 : Tensor
            Shape (B, D) — end point.
        n_points : int
            Number of points along the geodesic including endpoints.

        Returns
        -------
        Tensor
            Shape (n_points, B, D) — interpolated points.
        """
        v0 = z1 - z0  # initial velocity approximation

        points = [z0]
        z = z0.clone()

        step_size_orig = self.step_size
        self.step_size = 1.0 / (n_points - 1) / self.n_steps

        for i in range(1, n_points):
            z = self.shoot(z0, v0 * (i / (n_points - 1)))
            points.append(z)

        self.step_size = step_size_orig
        return torch.stack(points, dim=0)  # (n_points, B, D)

    def geodesic_distance(self, z0: Tensor, z1: Tensor) -> Tensor:
        """
        Estimate the geodesic distance between z0 and z1.

        Integrates ||gamma'(t)||_G dt along the geodesic path.

        Parameters
        ----------
        z0 : Tensor
            Shape (B, D).
        z1 : Tensor
            Shape (B, D).

        Returns
        -------
        Tensor
            Shape (B,) — geodesic distance per batch element.
        """
        from lrw.utils.linalg import riemannian_norm

        v0 = (z1 - z0) / self.n_steps
        z = z0.clone()
        v = v0.clone()

        distance = torch.zeros(z0.shape[0], device=z0.device, dtype=z0.dtype)

        for _ in range(self.n_steps):
            G = self.metric.metric_tensor(z)
            distance = distance + self.step_size * riemannian_norm(G, v)
            a = self.metric.geodesic_acceleration(z, v)
            z = z + self.step_size * v
            v = v + self.step_size * a

        return distance

    def __repr__(self) -> str:
        return (
            f"GeodesicSolver("
            f"n_steps={self.n_steps}, "
            f"step_size={self.step_size})"
        )
