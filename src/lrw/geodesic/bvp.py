"""
Boundary Value Problem (BVP) solver for geodesics on Riemannian manifolds.

Given two points z0 and z1 on the manifold, finds the initial velocity v0
such that the geodesic starting at z0 with velocity v0 arrives exactly at z1.

This is the correct formulation of geodesic interpolation.
The simpler GeodesicSolver.shoot() only solves the Initial Value Problem (IVP)
and cannot guarantee arrival at z1.

Method: Shooting method with iterative refinement via gradient descent on
the endpoint error ||gamma(1) - z1||^2_G.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lrw.geodesic.solver import GeodesicSolver
from lrw.metric.base import RiemannianMetric


class BVPSolver:
    """
    Geodesic Boundary Value Problem solver.

    Finds the initial velocity v0 such that:
        shoot(z0, v0) ≈ z1

    Uses gradient descent on the endpoint error to iteratively refine v0.

    The geodesic gamma: [0,1] -> M satisfies:
        gamma(0) = z0
        gamma(1) = z1
        gamma''(t) + Gamma(gamma(t)) gamma'(t) gamma'(t) = 0

    Parameters
    ----------
    metric : RiemannianMetric
        The Riemannian metric defining the manifold.
    n_steps : int
        Number of integration steps for each geodesic shoot.
    step_size : float
        Euler integration step size.
    lr : float
        Learning rate for gradient descent on v0.
    max_iter : int
        Maximum number of refinement iterations.
    tol : float
        Convergence tolerance on endpoint error ||gamma(1) - z1||.
    """

    def __init__(
        self,
        metric: RiemannianMetric,
        n_steps: int = 20,
        step_size: float = 0.05,
        lr: float = 0.1,
        max_iter: int = 50,
        tol: float = 1e-3,
    ) -> None:
        self.metric = metric
        self.n_steps = n_steps
        self.step_size = step_size
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self._solver = GeodesicSolver(
            metric=metric,
            n_steps=n_steps,
            step_size=step_size,
        )

    def _shoot_differentiable(self, z0: Tensor, v0: Tensor) -> Tensor:
        """
        Differentiable geodesic shoot using Euler integration.

        Unlike GeodesicSolver.shoot() which detaches gradients,
        this version keeps the computation graph for backprop through v0.

        Parameters
        ----------
        z0 : Tensor  Shape (B, D) — fixed start point.
        v0 : Tensor  Shape (B, D) — initial velocity (requires_grad=True).

        Returns
        -------
        Tensor
            Shape (B, D) — endpoint after integration.
        """
        z = z0.detach().clone()
        v = v0

        for _ in range(self.n_steps):
            # Geodesic acceleration: a = -Gamma(z) v v
            a = self.metric.geodesic_acceleration(z, v)
            z = z + self.step_size * v
            v = v + self.step_size * a

        return z

    def solve(self, z0: Tensor, z1: Tensor) -> tuple[Tensor, dict]:
        """
        Find initial velocity v0 such that shoot(z0, v0) ≈ z1.

        Parameters
        ----------
        z0 : Tensor
            Shape (B, D) — start point.
        z1 : Tensor
            Shape (B, D) — target end point.

        Returns
        -------
        tuple[Tensor, dict]
            v0_opt : Tensor  Shape (B, D) — optimal initial velocity.
            info   : dict    Convergence info (n_iter, final_error).
        """
        B, D = z0.shape

        # Initialize v0 with Euclidean direction (good starting point)
        v0 = (z1 - z0).detach().clone().requires_grad_(True)

        optimizer = torch.optim.Adam([v0], lr=self.lr)

        z1_detached = z1.detach()
        G1 = self.metric.metric_tensor(z1_detached).detach()  # (B, D, D)

        final_error = float("inf")
        n_iter = 0

        for i in range(self.max_iter):
            optimizer.zero_grad()

            # Shoot geodesic with current v0
            z_end = self._shoot_differentiable(z0, v0)  # (B, D)

            # Endpoint error in Riemannian metric at z1:
            # L = sum_b (z_end[b] - z1[b])^T G1[b] (z_end[b] - z1[b])
            residual = z_end - z1_detached               # (B, D)
            Gr = torch.bmm(G1, residual.unsqueeze(-1)).squeeze(-1)  # (B, D)
            loss = (residual * Gr).sum()

            loss.backward()
            optimizer.step()

            final_error = residual.norm(dim=-1).mean().item()
            n_iter = i + 1

            if final_error < self.tol:
                break

        info = {
            "n_iter": n_iter,
            "final_error": final_error,
            "converged": final_error < self.tol,
        }

        return v0.detach(), info

    def geodesic_path(
        self,
        z0: Tensor,
        z1: Tensor,
        n_points: int = 10,
    ) -> tuple[Tensor, dict]:
        """
        Compute the true geodesic path from z0 to z1.

        First solves the BVP to find optimal v0, then integrates
        to produce n_points along the geodesic.

        Parameters
        ----------
        z0 : Tensor  Shape (B, D).
        z1 : Tensor  Shape (B, D).
        n_points : int  Including endpoints.

        Returns
        -------
        tuple[Tensor, dict]
            path : Tensor  Shape (n_points, B, D).
            info : dict    BVP convergence info.
        """
        v0_opt, info = self.solve(z0, z1)

        # Integrate geodesic with optimal v0
        z = z0.detach().clone()
        v = v0_opt.clone()

        points = [z0.clone()]
        dt = self.step_size
        total_steps = self.n_steps
        # Record at evenly spaced intervals
        record_every = max(1, total_steps // (n_points - 1))

        for step in range(total_steps):
            a = self.metric.geodesic_acceleration(z, v)
            z = z + dt * v
            v = v + dt * a

            if (step + 1) % record_every == 0 and len(points) < n_points - 1:
                points.append(z.clone())

        points.append(z.clone())  # final point

        # Ensure exactly n_points
        while len(points) < n_points:
            points.append(points[-1])
        points = points[:n_points]

        return torch.stack(points, dim=0), info  # (n_points, B, D)

    def __repr__(self) -> str:
        return (
            f"BVPSolver("
            f"n_steps={self.n_steps}, "
            f"lr={self.lr}, "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol})"
        )
