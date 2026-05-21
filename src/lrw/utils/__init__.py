"""lrw.utils — shared mathematical utilities."""

from lrw.utils.jacobian import batch_jacobian
from lrw.utils.linalg import riemannian_norm, sym_inv, sym_sqrt
from lrw.utils.manifold import (
    manifold_assert_metric,
    manifold_assert_shape,
    manifold_assert_tangent,
)

__all__ = [
    "batch_jacobian",
    "sym_inv",
    "sym_sqrt",
    "riemannian_norm",
    "manifold_assert_shape",
    "manifold_assert_metric",
    "manifold_assert_tangent",
]
