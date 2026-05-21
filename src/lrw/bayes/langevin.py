"""Riemannian Stochastic Gradient Langevin Dynamics (SGLD)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from lrw.metric.base import RiemannianMetric
from lrw.utils.linalg import sym_inv


class RiemannianSGLD:
    """
    Riemannian Stochastic Gradient Langevin Dynamics.

    Samples from the posterior p(z | x) by running Langevin dynamics
    on the Riemannian manifold defined by the metric G(z).

    The update rule is:

        z_{t+1} = z_t + (eps/2) * G^{-1}(z_t) * grad_z log p(z_t)
                + sqrt(eps) * G^{-1/2}(z_t) * noise

    The metric G^{-1} acts as a preconditioning matrix that adapts
    the step size to the local geometry of the manifold.

    Parameters
    ----------
    metric : RiemannianMetric
        The Riemannian metric defining the manifold geometry.
    log_prob_fn : Callable[[Tensor], Tensor]
        Function z (B, D) -> log p(z | x) (B,). Target log density.
    step_size : float
        Langevin step size (eps).
    noise_scale : float
        Scale of injected noise. Set to 0.0 for MAP estimation (no noise).
    """

    def __init__(
        self,
        metric: RiemannianMetric,
        log_prob_fn: Callable[[Tensor], Tensor],
        step_size: float = 0.01,
        noise_scale: float = 1.0,
    ) -> None:
        self.metric = metric
        self.log_prob_fn = log_prob_fn
        self.step_size = step_size
        self.noise_scale = noise_scale

    def _grad_log_prob(self, z: Tensor) -> Tensor:
        """
        Compute grad_z log p(z | x).

        Returns
        -------
        Tensor
            Shape (B, D).
        """
        z_g = z.detach().requires_grad_(True)
        log_p = self.log_prob_fn(z_g)              # (B,)
        log_p.sum().backward()                     # type: ignore[no-untyped-call]
        assert z_g.grad is not None
        return z_g.grad.clone()

    def step(self, z: Tensor) -> Tensor:
        """
        Perform one Riemannian SGLD update.

        Parameters
        ----------
        z : Tensor
            Shape (B, D) — current samples.

        Returns
        -------
        Tensor
            Shape (B, D) — updated samples.
        """
        grad = self._grad_log_prob(z)              # (B, D)
        G = self.metric.metric_tensor(z)           # (B, D, D)
        G_inv = sym_inv(G)                         # (B, D, D)

        # Preconditioned gradient: G^{-1} * grad
        precond_grad = torch.bmm(
            G_inv,
            grad.unsqueeze(-1),
        ).squeeze(-1)                              # (B, D)

        drift = (self.step_size / 2.0) * precond_grad

        # Riemannian noise: G^{-1/2} * eps
        if self.noise_scale > 0.0:
            from lrw.utils.linalg import sym_sqrt
            G_inv_sqrt = sym_sqrt(G_inv)           # (B, D, D)
            raw_noise = torch.randn_like(z)
            noise = self.noise_scale * torch.bmm(
                G_inv_sqrt,
                raw_noise.unsqueeze(-1),
            ).squeeze(-1) * (self.step_size ** 0.5)
        else:
            noise = torch.zeros_like(z)

        return z + drift + noise

    def sample(self, z0: Tensor, n_steps: int, burn_in: int = 0) -> Tensor:
        """
        Run Riemannian SGLD for n_steps and collect samples.

        Parameters
        ----------
        z0 : Tensor
            Shape (B, D) — initial positions.
        n_steps : int
            Total number of steps.
        burn_in : int
            Number of initial steps to discard.

        Returns
        -------
        Tensor
            Shape (n_steps - burn_in, B, D) — collected samples.
        """
        z = z0.clone()
        samples = []

        for i in range(n_steps):
            z = self.step(z)
            if i >= burn_in:
                samples.append(z.clone())

        return torch.stack(samples, dim=0)

    def __repr__(self) -> str:
        return (
            f"RiemannianSGLD("
            f"step_size={self.step_size}, "
            f"noise_scale={self.noise_scale})"
        )
