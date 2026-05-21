"""Fisher-Rao metric for probabilistic decoders p(x|z)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from lrw.metric.base import RiemannianMetric


class FisherMetric(RiemannianMetric):
    """
    Fisher-Rao information metric on the latent space of a probabilistic decoder.

    For a decoder p(x|z), the Fisher metric is:

        G_F(z)_{ij} = E_{x ~ p(x|z)} [d_i log p(x|z) * d_j log p(x|z)]

    Estimated via Monte Carlo sampling.

    Parameters
    ----------
    log_prob_fn : Callable[[Tensor, Tensor], Tensor]
        Function (z, x) -> log p(x|z). Shape: (B, D), (B, *X) -> (B,).
    sample_fn : Callable[[Tensor, int], Tensor]
        Function (z, n_samples) -> samples from p(x|z).
        Shape: (B, D), int -> (B, n_samples, *X).
    n_samples : int
        Number of Monte Carlo samples for estimation.
    regularization : float
        Diagonal regularization for numerical stability.
    """

    def __init__(
        self,
        log_prob_fn: Callable[[Tensor, Tensor], Tensor],
        sample_fn: Callable[[Tensor, int], Tensor],
        n_samples: int = 16,
        regularization: float = 1e-5,
    ) -> None:
        self.log_prob_fn = log_prob_fn
        self.sample_fn = sample_fn
        self.n_samples = n_samples
        self.regularization = regularization

    def metric_tensor(self, z: Tensor) -> Tensor:
        """
        Estimate G_F(z) via Monte Carlo.

        G_F(z) ~ (1/N) sum_n [grad_z log p(x_n|z)] [grad_z log p(x_n|z)]^T

        Parameters
        ----------
        z : Tensor
            Shape (B, D).

        Returns
        -------
        Tensor
            Shape (B, D, D).
        """
        B, D = z.shape

        with torch.no_grad():
            x_samples = self.sample_fn(z, self.n_samples)  # (B, N, *X)

        G = torch.zeros(B, D, D, device=z.device, dtype=z.dtype)

        for n in range(self.n_samples):
            x_n = x_samples[:, n]                      # (B, *X)
            z_g = z.detach().requires_grad_(True)
            log_p = self.log_prob_fn(z_g, x_n)         # (B,)
            log_p.sum().backward()                      # type: ignore[no-untyped-call]

            assert z_g.grad is not None
            score = z_g.grad.clone()                    # (B, D)
            G = G + torch.bmm(
                score.unsqueeze(-1),
                score.unsqueeze(-2),
            )

        G = G / self.n_samples
        eye = torch.eye(D, device=z.device, dtype=z.dtype).unsqueeze(0)
        return G + self.regularization * eye

    def __repr__(self) -> str:
        return (
            f"FisherMetric("
            f"n_samples={self.n_samples}, "
            f"regularization={self.regularization})"
        )
