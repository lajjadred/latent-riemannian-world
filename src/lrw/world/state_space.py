"""Latent state space model on a Riemannian manifold."""

from __future__ import annotations

import torch
from torch import Tensor

from lrw.geodesic.solver import GeodesicSolver
from lrw.metric.base import RiemannianMetric


class LatentStateSpace:
    """
    Models temporal transitions in latent space as geodesic flows.

    Instead of treating z_{t+1} = z_t + noise (Euclidean assumption),
    this model treats the transition as movement along a geodesic on
    the Riemannian manifold defined by the metric G(z).

    z_{t+1} = geodesic_step(z_t, v_t, dt)

    where v_t is the latent velocity (tangent vector) at time t.

    Parameters
    ----------
    metric : RiemannianMetric
        The Riemannian metric defining the manifold geometry.
    solver : GeodesicSolver or None
        Geodesic solver. If None, creates one with default settings.
    dt : float
        Time step size for state transitions.
    noise_scale : float
        Scale of Gaussian noise added to velocity at each step.
        Set to 0.0 for deterministic transitions.
    """

    def __init__(
        self,
        metric: RiemannianMetric,
        solver: GeodesicSolver | None = None,
        dt: float = 0.1,
        noise_scale: float = 0.0,
    ) -> None:
        self.metric = metric
        self.solver = solver or GeodesicSolver(metric=metric, n_steps=10)
        self.dt = dt
        self.noise_scale = noise_scale

    def step(self, z: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Advance the state by one time step along the geodesic.

        Parameters
        ----------
        z : Tensor
            Shape (B, D) — current latent state.
        v : Tensor
            Shape (B, D) — current latent velocity.

        Returns
        -------
        tuple[Tensor, Tensor]
            (z_next, v_next) — next state and velocity.
            Both have shape (B, D).
        """
        # Compute geodesic acceleration at current state
        a = self.metric.geodesic_acceleration(z, v)  # (B, D)

        # Euler integration on the manifold
        z_next = z + self.dt * v
        v_next = v + self.dt * a

        # Optional stochastic noise on velocity
        if self.noise_scale > 0.0:
            noise = torch.randn_like(v_next) * self.noise_scale
            v_next = v_next + noise

        return z_next, v_next

    def rollout(
        self,
        z0: Tensor,
        v0: Tensor,
        n_steps: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Roll out a trajectory from initial state z0 with velocity v0.

        Parameters
        ----------
        z0 : Tensor
            Shape (B, D) — initial latent state.
        v0 : Tensor
            Shape (B, D) — initial latent velocity.
        n_steps : int
            Number of steps to roll out.

        Returns
        -------
        tuple[Tensor, Tensor]
            (states, velocities) — shape (n_steps+1, B, D) each.
            Index 0 is the initial state.
        """
        states = [z0]
        velocities = [v0]

        z, v = z0.clone(), v0.clone()

        for _ in range(n_steps):
            z, v = self.step(z, v)
            states.append(z)
            velocities.append(v)

        return torch.stack(states, dim=0), torch.stack(velocities, dim=0)

    def log_transition_prob(self, z: Tensor, z_next: Tensor, v: Tensor) -> Tensor:
        """
        Compute log probability of transition z -> z_next under the model.

        Models the transition as a Gaussian centered on the geodesic prediction:
            p(z_{t+1} | z_t, v_t) = N(z_{t+1}; z_t + dt*v_t, sigma^2 * G^{-1}(z_t))

        Parameters
        ----------
        z : Tensor
            Shape (B, D) — current state.
        z_next : Tensor
            Shape (B, D) — next state.
        v : Tensor
            Shape (B, D) — current velocity.

        Returns
        -------
        Tensor
            Shape (B,) — log transition probability per batch element.
        """

        z_pred = z + self.dt * v               # predicted next state
        residual = z_next - z_pred             # (B, D)

        G = self.metric.metric_tensor(z)       # (B, D, D)

        # Mahalanobis distance: residual^T G residual
        Gr = torch.bmm(G, residual.unsqueeze(-1)).squeeze(-1)  # (B, D)
        maha = (residual * Gr).sum(dim=-1)     # (B,)

        # Log det of G for normalization
        _, logdet = torch.linalg.slogdet(G)

        D = z.shape[-1]
        log_prob = -0.5 * (maha + logdet + D * torch.log(torch.tensor(2 * 3.14159)))

        return log_prob

    def __repr__(self) -> str:
        return (
            f"LatentStateSpace("
            f"dt={self.dt}, "
            f"noise_scale={self.noise_scale})"
        )
