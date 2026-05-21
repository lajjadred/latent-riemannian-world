"""Jacobian computation utilities using torch.func."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def batch_jacobian(
    func: Callable[[Tensor], Tensor],
    inputs: Tensor,
    chunk_size: int | None = None,
) -> Tensor:
    """
    Compute the Jacobian of func w.r.t. inputs for each batch element.

    Parameters
    ----------
    func : Callable
        A function mapping (D,) -> (M,). Applied independently per batch.
    inputs : Tensor
        Shape (B, D).
    chunk_size : int or None
        If set, computes Jacobian in chunks to save VRAM.

    Returns
    -------
    Tensor
        Shape (B, M, D).
    """
    jac_fn = torch.func.vmap(
        torch.func.jacrev(func),
        chunk_size=chunk_size,
    )
    return jac_fn(inputs)
