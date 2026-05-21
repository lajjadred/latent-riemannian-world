"""Pole Ladder: more accurate parallel transport than Schild's Ladder."""

from __future__ import annotations

from torch import Tensor

from lrw.geodesic.solver import GeodesicSolver
from lrw.metric.base import RiemannianMetric


class PoleLadder:
    """
    Parallel transport via the Pole Ladder algorithm.

    Pole Ladder is a more accurate alternative to Schild's Ladder
    for parallel transport on Riemannian manifolds. It uses a
    symmetric construction that reduces numerical error.

    Algorithm (one rung):
        Given base point z0, tangent vector v, next point z1:
        1. Compute pole p = geodesic midpoint of (z0, z0 + v)
        2. Shoot geodesic from z1 through p to find z2
           such that p is the midpoint of (z1, z2)
        3. Transported vector = z2 - z1

    Compared to Schild's Ladder:
        - Same O(epsilon^2) accuracy per step
        - Symmetric construction reduces systematic bias
        - Requires one fewer geodesic computation per rung

    Parameters
    ----------
    metric : RiemannianMetric
        The Riemannian metric.
    solver : GeodesicSolver or None
        Geodesic solver. If None, creates one with default settings.
    n_rungs : int
        Number of ladder rungs.
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
        Parallel transport vector v from z0 to z1 via Pole Ladder.

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
        path = self.solver.interpolate(z0, z1, n_points=self.n_rungs + 1)
        # path: (n_rungs+1, B, D)

        current_v = v.clone()

        for i in range(self.n_rungs):
            p0 = path[i]        # current base point
            p1 = path[i + 1]    # next base point

            # Step 1: endpoint of tangent vector
            z0v = p0 + current_v                        # (B, D)

            # Step 2: pole = midpoint of geodesic (p0, z0v)
            pole_path = self.solver.interpolate(p0, z0v, n_points=3)
            pole = pole_path[1]                         # (B, D)

            # Step 3: shoot from p1 through pole
            # Find z2 such that pole = midpoint(p1, z2)
            # => z2 = 2 * pole - p1
            z2 = 2.0 * pole - p1                        # (B, D)

            # Step 4: transported vector at p1
            current_v = z2 - p1

        return current_v

    def __repr__(self) -> str:
        return f"PoleLadder(n_rungs={self.n_rungs})"
