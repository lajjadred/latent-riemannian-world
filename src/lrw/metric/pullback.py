"""Pullback metric: G(z) = J(z)^T J(z) via decoder Jacobian."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from lrw.metric.base import RiemannianMetric


class PullbackMetric(RiemannianMetric):
    """
    Riemannian metric induced by pulling back the Euclidean metric
    through a differentiable decoder f: Z -> X.

    G(z) = J(z)^T J(z)

    where J(z) = df/dz is the Jacobian of the decoder.

    Parameters
    ----------
    decoder : Callable[[Tensor], Tensor]
        Maps latent z (B, D) -> data x (B, *data_shape).
        Must be differentiable.
    chunk_size : int or None
        Jacobian chunk size for VRAM control.
    regularization : float
        Small value added to diagonal for numerical stability.
    """

    def __init__(
        self,
        decoder: Callable[[Tensor], Tensor],
        chunk_size: int | None = None,
        regularization: float = 1e-5,
    ) -> None:
        self.decoder = decoder
        self.chunk_size = chunk_size
        self.regularization = regularization

    def metric_tensor(self, z: Tensor) -> Tensor:
        """
        Compute G(z) = J^T J.

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

        def flat_decoder(z_single: Tensor) -> Tensor:
            out = self.decoder(z_single.unsqueeze(0))  # (1, *data_shape)
            return out.flatten()                        # (M,)

        # J: (B, M, D)
        J = torch.func.vmap(
            torch.func.jacrev(flat_decoder),
            chunk_size=self.chunk_size,
        )(z)

        G = torch.bmm(J.mT, J)  # (B, D, D)

        eye = torch.eye(D, device=z.device, dtype=z.dtype).unsqueeze(0)
        return G + self.regularization * eye

    def local_volume_element(self, z: Tensor) -> Tensor:
        """
        Compute sqrt(det G(z)) — the Riemannian volume element.

        Returns
        -------
        Tensor
            Shape (B,).
        """
        G = self.metric_tensor(z)
        sign, logdet = torch.linalg.slogdet(G)
        return (sign * (logdet / 2).exp()).clamp(min=0.0)

    def __repr__(self) -> str:
        return (
            f"PullbackMetric("
            f"regularization={self.regularization})"
        )
