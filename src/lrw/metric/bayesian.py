"""Bayesian metric: posterior expected pullback metric G_B(z) = E[J^T J | x]."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from lrw.metric.pullback import PullbackMetric


class BayesianMetric(PullbackMetric):
    """
    Uncertainty-aware Riemannian metric defined as the weighted average
    of pullback metrics across a decoder ensemble.

    G_B(z) = sum_i w_i * G_i(z)

    where G_i(z) = J_i(z)^T J_i(z) is the pullback metric of the i-th decoder.

    This encodes both the geometry of the latent space and epistemic
    uncertainty about the decoder mapping.

    Parameters
    ----------
    decoder_ensemble : list[Callable[[Tensor], Tensor]]
        List of decoders. Each maps (B, D) -> (B, *data_shape).
        Can be MC-Dropout samples or deep ensemble members.
    weights : Tensor or None
        Unnormalized weights per ensemble member.
        If None, uniform weighting is used.
    chunk_size : int or None
        Jacobian chunk size passed to each PullbackMetric.
    regularization : float
        Diagonal regularization added once on the final average.
    """

    def __init__(
        self,
        decoder_ensemble: list[Callable[[Tensor], Tensor]],
        weights: Tensor | None = None,
        chunk_size: int | None = None,
        regularization: float = 1e-5,
    ) -> None:
        if len(decoder_ensemble) == 0:
            raise ValueError("decoder_ensemble must contain at least one decoder.")

        super().__init__(
            decoder=decoder_ensemble[0],
            chunk_size=chunk_size,
            regularization=0.0,
        )
        self._ensemble = decoder_ensemble
        self._reg = regularization

        n = len(decoder_ensemble)
        if weights is None:
            self._weights = torch.ones(n) / n
        else:
            self._weights = weights / weights.sum()

    def metric_tensor(self, z: Tensor) -> Tensor:
        """
        Compute G_B(z) = sum_i w_i * G_i(z).

        Also stores per-member metrics for uncertainty quantification
        via metric_variance().

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
        G_avg = torch.zeros(B, D, D, device=z.device, dtype=z.dtype)
        self._member_metrics: list[Tensor] = []

        for i, decoder in enumerate(self._ensemble):
            m = PullbackMetric(
                decoder=decoder,
                chunk_size=self.chunk_size,
                regularization=0.0,
            )
            G_i = m.metric_tensor(z)
            self._member_metrics.append(G_i)
            G_avg = G_avg + self._weights[i].item() * G_i

        eye = torch.eye(D, device=z.device, dtype=z.dtype).unsqueeze(0)
        return G_avg + self._reg * eye

    def metric_variance(self, z: Tensor) -> Tensor:
        """
        Compute element-wise variance of G across ensemble members.

        Var[G(z)] = sum_i w_i * (G_i - G_avg)^2

        Must call metric_tensor() first.

        Returns
        -------
        Tensor
            Shape (B, D, D).
        """
        if not hasattr(self, "_member_metrics"):
            raise RuntimeError("Call metric_tensor() before metric_variance().")

        G_avg = self.metric_tensor(z)
        var = torch.zeros_like(G_avg)
        for i, G_i in enumerate(self._member_metrics):
            var = var + self._weights[i].item() * (G_i - G_avg) ** 2
        return var

    def __repr__(self) -> str:
        return (
            f"BayesianMetric("
            f"n_ensemble={len(self._ensemble)}, "
            f"regularization={self._reg})"
        )
