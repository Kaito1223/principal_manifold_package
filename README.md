# Principal Manifold

Principal-curve and principal-graph estimators packaged with a `src/` layout.

The package exports:

- `KeglKrzyzakPrincipalCurve`
- `HSPrincipalCurve`
- `ElasticGraphPrincipalCurve`
- `ElasticPrincipalGraph`

The demo scripts are kept in `tests/` as runnable smoke/demo scripts.

## Install

```bash
python -m pip install -e .
```

## Quick example

```python
import numpy as np
from principal_manifold import HSConfig, HSPrincipalCurve

X = np.random.default_rng(0).normal(size=(100, 2))
model = HSPrincipalCurve(HSConfig(store_trace=True)).fit(X)
print(model.result_.vertices.shape)
```

## Layout

```text
src/principal_manifold/
├── _types.py
├── geometry.py
├── curves/
│   ├── hs.py
│   └── kk.py
├── elastic/
│   ├── optimizer.py
│   ├── fixed_chain.py
│   ├── primitive.py
│   ├── grammar.py
│   └── principal_graph.py
└── visual/
    ├── plotting.py
    ├── animation.py
    ├── tree.py
    └── comparison.py
```

## Demo scripts

Runnable smoke/demo scripts live in `tests/`.

- `uv run python tests/curve_evolution_demo.py` -> `tests/demo_2d_outputs/`
- `uv run python tests/curve_evolution_demo_3d.py` -> `tests/demo_3d_outputs/`
- `uv run python tests/surface_evolution_demo.py` -> `tests/demo_surface_outputs/`
- `uv run python tests/intrinsic_k_demo.py` -> `tests/demo_intrinsic_k_outputs/`

Each demo writes `summary.json` plus generated figures/animations.
