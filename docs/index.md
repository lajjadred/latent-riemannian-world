# latent-riemannian-world

**Riemannian geometry + Bayesian inference + world models for diffusion model latent spaces.**

## Overview

Existing tools treat latent space as Euclidean.  treats it as a **Riemannian manifold** — computing geodesics, parallel transport, and curvature. Bayesian extensions add uncertainty quantification. World model extensions add temporal state transitions.

## Installation

```bash
pip install latent-riemannian-world
```

## Quick Example

```python
import torch
from lrw.metric import PullbackMetric
from lrw.geodesic import GeodesicSolver

decoder = your_model.decode  # (B, D) -> (B, C, H, W)

metric = PullbackMetric(decoder=decoder)
solver = GeodesicSolver(metric=metric)

z0 = torch.randn(1, 16)
z1 = torch.randn(1, 16)
path = solver.interpolate(z0, z1, n_points=10)
```

## Modules

| Module | Description |
|---|---|
|  | Riemannian metrics (Pullback, Fisher, Bayesian) |
|  | Geodesic solvers and SLERP |
|  | Parallel transport (Schild, Pole Ladder) |
|  | Bayesian inference (SVGD, Riemannian SGLD) |
|  | World models (LatentStateSpace, RiemannianRSSM) |
|  | Math utilities |
