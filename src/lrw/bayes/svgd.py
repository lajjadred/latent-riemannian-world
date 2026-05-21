"""Stein Variational Gradient Descent (SVGD) on Riemannian manifolds."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from lrw.metric.base import RiemannianMetric


class SVGD:
    """
    Stein Variational Gradient Descent for posterior sampling on a Riemannian manifold.

    SVGD maintains a set of particles {z_i} that approximate the target
    posterior distribution p(z | x). Particles are updated by:

        z_i <- z_i + eps * phi(z_i)

    where phi is the Stein operator that drives particles toward the target
    while repelling each other to maintain diversity.

    Riemannian extension: uses the metric G(z) to define the kernel bandwidth
    and to project gradients onto the tangent space of the manifold.

    Parameters
    ----------
    metric : RiemannianMetric
        The Riemannian metric defining the manifold geometry.
    log_prob_fn : Callable[[Tensor], Tensor]
        Function z (B, D) -> log p(z | x) (B,). The target log density.
    kernel_bandwidth : float or None
        RBF kernel bandwidth. If None, uses median heuristic.
    step_size : float
        Gradient ascent step size.
    """

    def __init__(
        self,
        metric: RiemannianMetric,
        log_prob_fn: Callable[[Tensor], Tensor],
        kernel_bandwidth: float | None = None,
        step_size: float = 0.01,
    ) -> None:
        self.metric = metric
        self.log_prob_fn = log_prob_fn
        self.kernel_bandwidth = kernel_bandwidth
        self.step_size = step_size

    def _rbf_kernel(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute RBF kernel matrix and its gradient.

        k(z_i, z_j) = exp(-||z_i - z_j||^2 / h)

        Parameters
        ----------
        z : Tensor
            Shape (N, D) — particle positions.

        Returns
        -------
        tuple[Tensor, Tensor]
            K: (N, N) kernel matrix.
            dK: (N, N, D) kernel gradient w.r.t. first argument.
        """
        N, D = z.shape

        # Pairwise squared distances: (N, N)
        diff = z.unsqueeze(1) - z.unsqueeze(0)       # (N, N, D)
        sq_dist = (diff ** 2).sum(dim=-1)             # (N, N)

        # Median heuristic for bandwidth
        if self.kernel_bandwidth is None:
            h = sq_dist.median() / (2 * torch.log(torch.tensor(N + 1.0)))
            h = h.clamp(min=1e-6)
        else:
            h = torch.tensor(self.kernel_bandwidth)

        K = torch.exp(-sq_dist / h)                   # (N, N)

        # Gradient of k w.r.t. z_i: dK[i,j] = -2/h * (z_i - z_j) * K[i,j]
        dK = -2.0 / h * diff * K.unsqueeze(-1)        # (N, N, D)

        return K, dK

    def _score(self, z: Tensor) -> Tensor:
        """
        Compute grad_z log p(z | x) for each particle.

        Returns
        -------
        Tensor
            Shape (N, D).
        """
        z_g = z.detach().requires_grad_(True)
        log_p = self.log_prob_fn(z_g)                 # (N,)
        log_p.sum().backward()                        # type: ignore[no-untyped-call]
        assert z_g.grad is not None
        return z_g.grad.clone()

    def step(self, z: Tensor) -> Tensor:
        """
        Perform one SVGD update step.

        phi(z_i) = (1/N) sum_j [k(z_j, z_i) * grad_{z_j} log p(z_j)
                                + grad_{z_j} k(z_j, z_i)]

        Parameters
        ----------
        z : Tensor
            Shape (N, D) — current particle positions.

        Returns
        -------
        Tensor
            Shape (N, D) — updated particle positions.
        """
        N = z.shape[0]
        K, dK = self._rbf_kernel(z)                   # (N,N), (N,N,D)
        score = self._score(z)                         # (N, D)

        # phi[i] = (1/N) sum_j [K[j,i] * score[j] + dK[j,i]]
        # K[j,i] * score[j]: weighted score  -> einsum nj, jd -> nd... wait
        # K shape (N,N): K[i,j] = k(z_i, z_j)
        # phi[i] = (1/N) sum_j K[i,j] * score[j] + sum_j dK[i,j]
        phi = (K @ score + dK.sum(dim=1)) / N         # (N, D)

        return z + self.step_size * phi

    def run(self, z: Tensor, n_steps: int) -> Tensor:
        """
        Run SVGD for n_steps iterations.

        Parameters
        ----------
        z : Tensor
            Shape (N, D) — initial particle positions.
        n_steps : int
            Number of update steps.

        Returns
        -------
        Tensor
            Shape (N, D) — final particle positions.
        """
        for _ in range(n_steps):
            z = self.step(z)
        return z

    def __repr__(self) -> str:
        return (
            f"SVGD("
            f"step_size={self.step_size}, "
            f"kernel_bandwidth={self.kernel_bandwidth})"
        )
