"""Tests for lrw.geodesic.GeodesicSolver."""

import torch
import pytest
from lrw.metric import PullbackMetric
from lrw.geodesic import GeodesicSolver


def make_metric(D: int = 4, seed: int = 0):
    """Simple linear decoder for testing."""
    torch.manual_seed(seed)
    W = torch.randn(16, D)

    def decoder(z: torch.Tensor) -> torch.Tensor:
        return z @ W.T

    return PullbackMetric(decoder=decoder)


def test_shoot_output_shape():
    metric = make_metric()
    solver = GeodesicSolver(metric=metric, n_steps=10, step_size=0.01)
    z0 = torch.randn(3, 4)
    v0 = torch.randn(3, 4)
    z1 = solver.shoot(z0, v0)
    assert z1.shape == (3, 4)


def test_shoot_zero_velocity():
    """With zero velocity, endpoint should equal startpoint."""
    metric = make_metric()
    solver = GeodesicSolver(metric=metric, n_steps=20, step_size=0.01)
    z0 = torch.randn(3, 4)
    v0 = torch.zeros(3, 4)
    z1 = solver.shoot(z0, v0)
    assert torch.allclose(z0, z1, atol=1e-5)


def test_interpolate_shape():
    metric = make_metric()
    solver = GeodesicSolver(metric=metric, n_steps=10, step_size=0.01)
    z0 = torch.randn(2, 4)
    z1 = torch.randn(2, 4)
    path = solver.interpolate(z0, z1, n_points=5)
    assert path.shape == (5, 2, 4)


def test_interpolate_endpoints():
    """First point should be z0."""
    metric = make_metric()
    solver = GeodesicSolver(metric=metric, n_steps=10, step_size=0.01)
    z0 = torch.randn(2, 4)
    z1 = torch.randn(2, 4)
    path = solver.interpolate(z0, z1, n_points=5)
    assert torch.allclose(path[0], z0, atol=1e-5)


def test_geodesic_distance_positive():
    metric = make_metric()
    solver = GeodesicSolver(metric=metric, n_steps=10, step_size=0.01)
    z0 = torch.randn(3, 4)
    z1 = torch.randn(3, 4)
    dist = solver.geodesic_distance(z0, z1)
    assert dist.shape == (3,)
    assert (dist >= 0).all()


def test_geodesic_distance_zero():
    """Distance from a point to itself should be zero."""
    metric = make_metric()
    solver = GeodesicSolver(metric=metric, n_steps=10, step_size=0.01)
    z0 = torch.randn(3, 4)
    dist = solver.geodesic_distance(z0, z0)
    assert torch.allclose(dist, torch.zeros(3), atol=1e-5)
