"""Schild's Ladder: parallel transport along a geodesic."""

from __future__ import annotations

from torch import Tensor

from lrw.geodesic.solver import GeodesicSolver
from lrw.metric.base import RiemannianMetric


class SchildsLadder:
    """
    Parallel transport of a tangent vector along a geodesic via Schild's Ladder.

    Schild's Ladder is a simple geometric construction for parallel transport
    on a Riemannian manifold. It approximates parallel transport by repeatedly
    constructing midpoints along the geodesic.

    Algorithm (one rung):
        Given point z0, tangent vector v, and next point z1 on the geodesic:
        1. Compute the endpoint z0v = z0 + v (point displaced by v)
        2. Find midpoint m of geodesic from z0v to z1
        3. Find point z2 = 2*m - z0  (reflect z0 through m)
        4. Transported vector = z2 - z1

    Parameters
    ----------
    metric : RiemannianMetric
        The Riemannian metric.
    solver : GeodesicSolver or None
        Geodesic solver to use. If None, creates one with default settings.
    n_rungs : int
        Number of ladder rungs (more rungs = more accurate transport).
    """

    def __init__(
        self,
        metric: RiemannianMetric,
        solver: GeodesicSolver | None = None,
        n_rungs: int = 10,
    ) -> None:
        self.metric = metric
        self.solver = solver or GeodesicSolver(metric=metric)
        self.n_rungs = n_rungs

    def transport(
        self,
        z0: Tensor,
        z1: Tensor,
        v: Tensor,
    ) -> Tensor:
        """
        Parallel transport vector v from z0 to z1 along the geodesic.

        Parameters
        ----------
        z0 : Tensor
            Shape (B, D) — start point.
        z1 : Tensor
            Shape (B, D) — end point.
        v : Tensor
            Shape (B, D) — tangent vector at z0 to be transported.

        Returns
        -------
        Tensor
            Shape (B, D) — transported vector at z1.
        """
        # Discretize the geodesic path into n_rungs steps
        path = self.solver.interpolate(z0, z1, n_points=self.n_rungs + 1)
        # path: (n_rungs+1, B, D)

        current_v = v.clone()

        for i in range(self.n_rungs):
            p0 = path[i]       # current base point
            p1 = path[i + 1]   # next base point

            # Step 1: displaced point
            z0v = p0 + current_v  # (B, D)

            # Step 2: midpoint of geodesic from z0v to p1
            mid_path = self.solver.interpolate(z0v, p1, n_points=3)
            m = mid_path[1]  # midpoint: (B, D)

            # Step 3: reflect p0 through m
            z2 = 2.0 * m - p0  # (B, D)

            # Step 4: new transported vector
            current_v = z2 - p1

        return current_v

    def __repr__(self) -> str:
        return f"SchildsLadder(n_rungs={self.n_rungs})"
