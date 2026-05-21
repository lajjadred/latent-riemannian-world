"""lrw.geodesic — Geodesic solvers on Riemannian latent manifolds."""

from lrw.geodesic.slerp import slerp, slerp_path
from lrw.geodesic.solver import GeodesicSolver

__all__ = [
    "GeodesicSolver",
    "slerp",
    "slerp_path",
]