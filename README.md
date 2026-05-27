# latent-riemannian-world

[![CI](https://github.com/lajjadred/latent-riemannian-world/actions/workflows/ci.yml/badge.svg)](https://github.com/lajjadred/latent-riemannian-world/actions)
[![PyPI](https://img.shields.io/pypi/v/latent-riemannian-world)](https://pypi.org/project/latent-riemannian-world/)
[![Python](https://img.shields.io/pypi/pyversions/latent-riemannian-world)](https://pypi.org/project/latent-riemannian-world/)
[![License: BSL-1.1](https://img.shields.io/badge/License-BSL%201.1-yellow.svg)](LICENSE)

**Riemannian geometry + Bayesian inference + world models for diffusion model latent spaces.**

## Why this library?

| | latent-geometry | Diffusion-Pullback | **lrw** |
|---|---|---|---|
| Pullback metric | ✓ | ✓ | ✓ |
| Fisher-Rao metric | ✗ | ✗ | ✓ |
| Bayesian metric | ✗ | ✗ | ✓ |
| Geodesic solver (IVP) | ✗ | ✗ | ✓ |
| **Geodesic solver (BVP)** | ✗ | ✗ | **✓** |
| Parallel transport | ✗ | ✗ | ✓ |
| SVGD / Riemannian SGLD | ✗ | ✗ | ✓ |
| World model / temporal | ✗ | ✗ | ✓ |
| Python 3.12+ / PyTorch 2.4+ | ✗ | ✗ | ✓ |

## Installation

```bash
pip install latent-riemannian-world
```

## Quick Start

```python
import torch
from lrw.metric import PullbackMetric, BayesianMetric
from lrw.geodesic import GeodesicSolver, BVPSolver
from lrw.transport import SchildsLadder, PoleLadder
from lrw.bayes import SVGD, RiemannianSGLD
from lrw.world import LatentStateSpace, RiemannianRSSM

decoder = your_model.decode   # (B, D) -> (B, C, H, W)
metric = PullbackMetric(decoder=decoder)
z = torch.randn(4, 16)

# IVP solver — fast, approximate
solver = GeodesicSolver(metric=metric)
path = solver.interpolate(z[0:1], z[1:2], n_points=10)

# BVP solver — true geodesic, guaranteed arrival at z1
bvp = BVPSolver(metric=metric, lr=0.1, max_iter=50)
true_path, info = bvp.geodesic_path(z[0:1], z[1:2], n_points=10)
print(f"Converged: {info['converged']}, error: {info['final_error']:.4f}")
```

## IVP vs BVP

| | GeodesicSolver (IVP) | BVPSolver |
|---|---|---|
| Speed | Fast | Slower (iterative) |
| Arrival at z1 | Not guaranteed | Guaranteed |
| Use case | Prototyping | WAN keyframes, final quality |

## Module Structure

```
lrw/
├── metric/      PullbackMetric, FisherMetric, BayesianMetric
├── geodesic/    GeodesicSolver (IVP), BVPSolver (true geodesic), slerp
├── transport/   SchildsLadder, PoleLadder
├── bayes/       SVGD, RiemannianSGLD
├── world/       LatentStateSpace, RiemannianRSSM
└── utils/       sym_inv, sym_sqrt, riemannian_norm, manifold_assert_*
```

## References

- Shao et al. (2018) *The Riemannian Geometry of Deep Generative Models*. CVPR.
- Arvanitidis et al. (2018) *Latent Space Oddity*. ICLR.
- Park et al. (2023) *Riemannian Geometry of Diffusion Models*. NeurIPS.
- Liu et al. (2016) *Stein Variational Gradient Descent*. NeurIPS.
- Hafner et al. (2020) *Dream to Control*. ICLR.

## License

BSL-1.1 — (c) 2025 lajjadred
