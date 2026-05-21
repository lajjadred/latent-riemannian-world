"""Tests for lrw.utils.manifold."""

import torch
import pytest
from lrw.utils.manifold import (
    manifold_assert_shape,
    manifold_assert_metric,
    manifold_assert_tangent,
)


def test_assert_shape_valid():
    z = torch.randn(3, 4)
    manifold_assert_shape(z)


def test_assert_shape_wrong_ndim():
    z = torch.randn(4)
    with pytest.raises(ValueError):
        manifold_assert_shape(z)


def test_assert_shape_wrong_dim():
    z = torch.randn(3, 4)
    with pytest.raises(ValueError):
        manifold_assert_shape(z, expected_dim=8)


def test_assert_metric_valid():
    A = torch.randn(3, 4, 4)
    G = A @ A.mT + torch.eye(4).unsqueeze(0) * 0.5
    manifold_assert_metric(G)


def test_assert_metric_wrong_shape():
    G = torch.randn(3, 4, 5)
    with pytest.raises(ValueError):
        manifold_assert_metric(G)


def test_assert_metric_not_symmetric():
    G = torch.randn(2, 4, 4)
    with pytest.raises(ValueError):
        manifold_assert_metric(G)


def test_assert_tangent_valid():
    z = torch.randn(3, 4)
    v = torch.randn(3, 4)
    manifold_assert_tangent(z, v)


def test_assert_tangent_shape_mismatch():
    z = torch.randn(3, 4)
    v = torch.randn(3, 8)
    with pytest.raises(ValueError):
        manifold_assert_tangent(z, v)
