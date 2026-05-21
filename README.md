# latent-riemannian-world

[![PyPI](https://img.shields.io/pypi/v/latent-riemannian-world)](https://pypi.org/project/latent-riemannian-world/)
[![Python](https://img.shields.io/pypi/pyversions/latent-riemannian-world)](https://pypi.org/project/latent-riemannian-world/)
[![License: BSL-1.1](https://img.shields.io/badge/License-BSL%201.1-yellow.svg)](LICENSE)

**Riemannian geometry + Bayesian inference + world models for diffusion model latent spaces.**

Existing tools treat latent space as Euclidean. This library treats it as a **Riemannian manifold** — computing geodesics (true shortest paths on curved space), parallel transport, and curvature. Bayesian extensions add uncertainty quantification. World model extensions add temporal state transitions.

## Why this library?

| | latent-geometry | Diffusion-Pullback | **lrw** |
|---|---|---|---|
| Pullback metric | ✓ | ✓ | ✓ |
| Fisher-Rao metric | ✗ | ✗ | ✓ |
| Bayesian (uncertain) metric | ✗ | ✗ | ✓ |
| Geodesic solver | ✗ | ✗ | ✓ |
| Parallel transport | ✗ | ✗ | ✓ |
| SVGD / Riemannian SGLD | ✗ | ✗ | ✓ |
| World model / temporal | ✗ | ✗ | ✓ |
| Python 3.12+ / PyTorch 2.4+ | ✗ | ✗ | ✓ |
| pip installable | ✓ | ✗ | ✓ |

## Installation

```bash
pip install latent-riemannian-world
```

## Quick Start

```python
import torch
from lrw.metric import PullbackMetric, BayesianMetric
from lrw.geodesic import GeodesicSolver
from lrw.transport import SchildsLadder
from lrw.bayes import SVGD, RiemannianSGLD
from lrw.world import LatentStateSpace, RiemannianRSSM

# Your diffusion model decoder
decoder = your_model.decode   # (B, D) -> (B, C, H, W)

# Pullback metric: G(z) = J(z)^T J(z)
metric = PullbackMetric(decoder=decoder)
z = torch.randn(4, 16)
G = metric.metric_tensor(z)   # (4, 16, 16) Riemannian metric matrices

# Geodesic interpolation between two latent points
solver = GeodesicSolver(metric=metric)
path = solver.interpolate(z[0:1], z[1:2], n_points=10)

# Parallel transport of a style vector along the geodesic
ladder = SchildsLadder(metric=metric)
v_transported = ladder.transport(z[0:1], z[1:2], style_vector)

# Bayesian metric with MC-Dropout ensemble
decoders = [decoder_with_dropout() for _ in range(8)]
bayes_metric = BayesianMetric(decoder_ensemble=decoders)
G_bayes = bayes_metric.metric_tensor(z)
G_var   = bayes_metric.metric_variance(z)

# World model: temporal transitions as geodesic flows
state_space = LatentStateSpace(metric=metric, dt=0.1)
z_next, v_next = state_space.step(z, velocity)
states, velocities = state_space.rollout(z, velocity, n_steps=20)
```

## Module Structure

```
lrw/
├── metric/      PullbackMetric, FisherMetric, BayesianMetric
├── geodesic/    GeodesicSolver, slerp, slerp_path
├── transport/   SchildsLadder
├── bayes/       SVGD, RiemannianSGLD
├── world/       LatentStateSpace, RiemannianRSSM
└── utils/       sym_inv, sym_sqrt, riemannian_norm, batch_jacobian
```

## References

- Shao et al. (2018) *The Riemannian Geometry of Deep Generative Models*. CVPR.
- Arvanitidis et al. (2018) *Latent Space Oddity: on the Curvature of Deep Generative Models*. ICLR.
- Park et al. (2023) *Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry*. NeurIPS.
- Liu et al. (2016) *Stein Variational Gradient Descent*. NeurIPS.
- Hafner et al. (2020) *Dream to Control: Learning Behaviors by Latent Imagination*. ICLR.

## License

This project is licensed under the Business Source License 1.1 (BSL-1.1).
Non-production use (research, personal projects, evaluation) is free.
For commercial/production use, contact the author.

(c) 2025 lajjadred
