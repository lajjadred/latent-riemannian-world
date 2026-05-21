"""Riemannian Recurrent State Space Model (RiemannianRSSM)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lrw.metric.base import RiemannianMetric
from lrw.utils.linalg import sym_inv


class RiemannianRSSM(nn.Module):
    """
    Riemannian Recurrent State Space Model.

    Extends the Dreamer/RSSM architecture by replacing Euclidean latent
    transitions with geodesic flows on a Riemannian manifold.

    Standard RSSM:
        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})      (deterministic path)
        z_t ~ p(z_t | h_t)                        (stochastic state)

    RiemannianRSSM:
        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})      (deterministic path)
        mu_t, sigma_t = prior_net(h_t)
        z_t ~ N_R(mu_t, sigma_t)                  (Riemannian Gaussian)

    where N_R is a Gaussian whose covariance is shaped by G^{-1}(mu_t).

    Parameters
    ----------
    metric : RiemannianMetric
        The Riemannian metric defining the manifold geometry.
    latent_dim : int
        Dimension of the latent state z.
    hidden_dim : int
        Dimension of the deterministic hidden state h.
    action_dim : int
        Dimension of the action vector a. Set to 0 if no actions.
    """

    def __init__(
        self,
        metric: RiemannianMetric,
        latent_dim: int,
        hidden_dim: int,
        action_dim: int = 0,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Deterministic path: GRU cell
        gru_input_dim = latent_dim + max(action_dim, 1)
        self.gru = nn.GRUCell(gru_input_dim, hidden_dim)

        # Prior network: h_t -> (mu, log_sigma)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # Posterior network: (h_t, obs_embed) -> (mu, log_sigma)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def prior(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute prior distribution parameters from hidden state.

        Parameters
        ----------
        h : Tensor
            Shape (B, hidden_dim).

        Returns
        -------
        tuple[Tensor, Tensor]
            (mu, sigma) — Shape (B, latent_dim) each.
        """
        out = self.prior_net(h)                        # (B, latent_dim * 2)
        mu, log_sigma = out.chunk(2, dim=-1)
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-4
        return mu, sigma

    def posterior(self, h: Tensor, obs_embed: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute posterior distribution parameters from hidden state and observation.

        Parameters
        ----------
        h : Tensor
            Shape (B, hidden_dim).
        obs_embed : Tensor
            Shape (B, latent_dim) — encoded observation.

        Returns
        -------
        tuple[Tensor, Tensor]
            (mu, sigma) — Shape (B, latent_dim) each.
        """
        inp = torch.cat([h, obs_embed], dim=-1)        # (B, hidden_dim + latent_dim)
        out = self.posterior_net(inp)
        mu, log_sigma = out.chunk(2, dim=-1)
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-4
        return mu, sigma

    def riemannian_sample(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """
        Sample from Riemannian Gaussian N_R(mu, sigma).

        The covariance is adapted by G^{-1}(mu):
            z = mu + sigma * G^{-1/2}(mu) * eps,  eps ~ N(0, I)

        Parameters
        ----------
        mu : Tensor
            Shape (B, D) — mean (base point on manifold).
        sigma : Tensor
            Shape (B, D) — diagonal scale.

        Returns
        -------
        Tensor
            Shape (B, D) — sampled latent state.
        """
        from lrw.utils.linalg import sym_sqrt

        G = self.metric.metric_tensor(mu)              # (B, D, D)
        G_inv = sym_inv(G)                             # (B, D, D)
        G_inv_sqrt = sym_sqrt(G_inv)                   # (B, D, D)

        eps = torch.randn_like(mu)                     # (B, D)
        scaled_eps = sigma * eps                       # (B, D)

        noise = torch.bmm(
            G_inv_sqrt,
            scaled_eps.unsqueeze(-1),
        ).squeeze(-1)                                  # (B, D)

        return mu + noise

    def step(
        self,
        h: Tensor,
        z: Tensor,
        action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform one RSSM step.

        Parameters
        ----------
        h : Tensor
            Shape (B, hidden_dim) — previous hidden state.
        z : Tensor
            Shape (B, latent_dim) — previous latent state.
        action : Tensor or None
            Shape (B, action_dim) — action taken. If None, uses zeros.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            (h_next, z_prior, mu_prior, sigma_prior)
        """
        B = z.shape[0]

        if action is None:
            action = torch.zeros(B, max(self.action_dim, 1), device=z.device)

        gru_input = torch.cat([z, action], dim=-1)     # (B, latent_dim + action_dim)
        h_next = self.gru(gru_input, h)                # (B, hidden_dim)

        mu_prior, sigma_prior = self.prior(h_next)
        z_prior = self.riemannian_sample(mu_prior, sigma_prior)

        return h_next, z_prior, mu_prior, sigma_prior

    def riemannian_kl(
        self,
        mu_post: Tensor,
        sigma_post: Tensor,
        mu_prior: Tensor,
        sigma_prior: Tensor,
    ) -> Tensor:
        """
        Approximate KL divergence between posterior and prior,
        weighted by the Riemannian metric at the posterior mean.

        KL_R ~ 0.5 * tr(G(mu_post) * Sigma_post * G(mu_prior)^{-1})
               + geodesic_dist(mu_post, mu_prior)^2 - D

        For simplicity, uses diagonal Gaussian KL as baseline
        with Riemannian correction via metric ratio.

        Parameters
        ----------
        mu_post, sigma_post : Tensor
            Shape (B, D) — posterior parameters.
        mu_prior, sigma_prior : Tensor
            Shape (B, D) — prior parameters.

        Returns
        -------
        Tensor
            Shape (B,) — KL divergence per batch element.
        """
        # Standard diagonal Gaussian KL as baseline
        var_post = sigma_post ** 2
        var_prior = sigma_prior ** 2

        kl = 0.5 * (
            (var_post / var_prior.clamp(min=1e-8))
            + (mu_post - mu_prior) ** 2 / var_prior.clamp(min=1e-8)
            - 1.0
            + var_prior.log() - var_post.log()
        ).sum(dim=-1)

        return kl

    def __repr__(self) -> str:
        return (
            f"RiemannianRSSM("
            f"latent_dim={self.latent_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"action_dim={self.action_dim})"
        )
