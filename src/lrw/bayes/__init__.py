"""lrw.bayes — Bayesian inference on Riemannian latent manifolds."""

from lrw.bayes.langevin import RiemannianSGLD
from lrw.bayes.svgd import SVGD

__all__ = [
    "SVGD",
    "RiemannianSGLD",
]
