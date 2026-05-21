# Getting Started

## Installation

```bash
pip install latent-riemannian-world
```

## Basic Usage

### 1. Define a Decoder

```python
import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=16, output_dim=64):
        super().__init__()
        self.net = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.net(z)

decoder = SimpleDecoder()
```

### 2. Create a Pullback Metric

```python
from lrw.metric import PullbackMetric

metric = PullbackMetric(decoder=decoder)
z = torch.randn(4, 16)
G = metric.metric_tensor(z)  # (4, 16, 16)
```

### 3. Compute Geodesics

```python
from lrw.geodesic import GeodesicSolver

solver = GeodesicSolver(metric=metric)
z0 = torch.randn(1, 16)
z1 = torch.randn(1, 16)

# Interpolate along geodesic
path = solver.interpolate(z0, z1, n_points=10)  # (10, 1, 16)

# Geodesic distance
dist = solver.geodesic_distance(z0, z1)
```

### 4. Parallel Transport

```python
from lrw.transport import SchildsLadder, PoleLadder

ladder = PoleLadder(metric=metric)
v = torch.randn(1, 16)
v_transported = ladder.transport(z0, z1, v)
```

### 5. Bayesian Metric

```python
from lrw.metric import BayesianMetric

decoders = [SimpleDecoder() for _ in range(8)]
bayes_metric = BayesianMetric(decoder_ensemble=decoders)
G = bayes_metric.metric_tensor(z)
uncertainty = bayes_metric.metric_variance(z)
```
