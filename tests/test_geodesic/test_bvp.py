"""Tests for lrw.geodesic.BVPSolver."""

import torch
import pytest
from lrw.metric import PullbackMetric
from lrw.geodesic import BVPSolver


def make_metric(D: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    W = torch.randn(16, D)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    return PullbackMetric(decoder=decoder)


def test_bvp_solve_shape():
    metric = make_metric()
    solver = BVPSolver(metric=metric, n_steps=10, max_iter=5)
    z0 = torch.randn(2, 4)
    z1 = torch.randn(2, 4)
    v0, info = solver.solve(z0, z1)
    assert v0.shape == (2, 4)
    assert "n_iter" in info
    assert "final_error" in info
    assert "converged" in info


def test_bvp_endpoint_closer_than_euclidean():
    """BVP solution should bring endpoint closer to z1 than Euclidean init."""
    metric = make_metric()
    solver = BVPSolver(metric=metric, n_steps=15, lr=0.05, max_iter=20)

    torch.manual_seed(42)
    z0 = torch.randn(1, 4)
    z1 = torch.randn(1, 4)

    v0_opt, info = solver.solve(z0, z1)

    # Shoot with optimized v0
    from lrw.geodesic import GeodesicSolver
    gsolver = GeodesicSolver(metric=metric, n_steps=15, step_size=solver.step_size)
    z_end_opt = gsolver.shoot(z0, v0_opt)

    # Euclidean init shoot
    v0_euc = z1 - z0
    z_end_euc = gsolver.shoot(z0, v0_euc)

    err_opt = (z_end_opt - z1).norm().item()
    err_euc = (z_end_euc - z1).norm().item()

    assert err_opt <= err_euc + 1e-3, (
        f"BVP error {err_opt:.4f} should be <= Euclidean error {err_euc:.4f}"
    )


def test_bvp_geodesic_path_shape():
    metric = make_metric()
    solver = BVPSolver(metric=metric, n_steps=10, max_iter=5)
    z0 = torch.randn(2, 4)
    z1 = torch.randn(2, 4)
    path, info = solver.geodesic_path(z0, z1, n_points=5)
    assert path.shape == (5, 2, 4)


def test_bvp_path_starts_at_z0():
    metric = make_metric()
    solver = BVPSolver(metric=metric, n_steps=10, max_iter=5)
    z0 = torch.randn(2, 4)
    z1 = torch.randn(2, 4)
    path, _ = solver.geodesic_path(z0, z1, n_points=5)
    assert torch.allclose(path[0], z0, atol=1e-5)


def test_bvp_repr():
    metric = make_metric()
    solver = BVPSolver(metric=metric)
    assert "BVPSolver" in repr(solver)
