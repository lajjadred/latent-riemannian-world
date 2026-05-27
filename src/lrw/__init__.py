"""
latent-riemannian-world (lrw)
==============================
Riemannian geometry and Bayesian inference for diffusion model latent spaces.

Submodules
----------
lrw.metric    : Riemannian metrics (PullbackMetric, FisherMetric, BayesianMetric)
lrw.geodesic  : Geodesic solvers (GeodesicSolver, slerp)
lrw.transport : Parallel transport (SchildsLadder)
lrw.bayes     : Bayesian inference (SVGD, RiemannianSGLD)
lrw.world     : World models (LatentStateSpace, RiemannianRSSM)
lrw.utils     : Shared math utilities

(c) 2025 lajjadred — BSL-1.1 License
"""

__version__ = "0.3.0"
__author__ = "lajjadred"

from lrw import bayes, geodesic, metric, transport, utils, world

__all__ = [
    "metric",
    "geodesic",
    "transport",
    "bayes",
    "world",
    "utils",
    "__version__",
]
