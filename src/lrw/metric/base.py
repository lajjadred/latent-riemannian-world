"""Abstract base class for all Riemannian metrics in lrw."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from lrw.utils.linalg import sym_inv


class RiemannianMetric(ABC):
    """
    Abstract base for a Riemannian metric G(z) on a latent manifold.

    Subclasses must implement metric_tensor().
    Christoffel symbols and geodesic acceleration are derived
    from it automatically via autograd.
    """

    @abstractmethod
    def metric_tensor(self, z: Tensor) -> Tensor:
        """
        Return the metric matrix G at point z.

        Parameters
        ----------
        z : Tensor
            Shape (B, D).

        Returns
        -------
        Tensor
            Shape (B, D, D) — symmetric positive definite matrix.
        """

    def christoffel(self, z: Tensor) -> Tensor:
        """
        Compute Christoffel symbols via autograd on G.

        Gamma^k_{ij} = 0.5 * G^{kl} (d_i G_{lj} + d_j G_{li} - d_l G_{ij})

        Returns
        -------
        Tensor
            Shape (B, D, D, D).
        """
        def _G_at(z_single: Tensor) -> Tensor:
            return self.metric_tensor(z_single.unsqueeze(0)).squeeze(0)

        # jac_G[b, i, j, d] = dG_{ij}/dz_d
        jac_G = torch.func.vmap(torch.func.jacrev(_G_at))(z)  # (B, D, D, D)

        G = self.metric_tensor(z)
        G_inv = sym_inv(G)

        # bracket[b, l, i, j] = d_i G_{lj} + d_j G_{li} - d_l G_{ij}
        bracket = (
            jac_G.permute(0, 1, 3, 2)   # d_i G_{lj}
            + jac_G                      # d_j G_{li}
            - jac_G.permute(0, 3, 2, 1) # d_l G_{ij}
        )
        return 0.5 * torch.einsum("bkl, blij -> bkij", G_inv, bracket)

    def geodesic_acceleration(self, z: Tensor, v: Tensor) -> Tensor:
        """
        Compute acceleration along geodesic: a = -Gamma^k_{ij} v^i v^j

        Parameters
        ----------
        z : Tensor
            Shape (B, D) — position.
        v : Tensor
            Shape (B, D) — velocity.

        Returns
        -------
        Tensor
            Shape (B, D).
        """
        Gamma = self.christoffel(z)
        return -torch.einsum("bkij, bi, bj -> bk", Gamma, v, v)
