from __future__ import annotations

import numpy as np

from principal_manifold import IntrinsicHSConfig, IntrinsicHSPrincipalManifold


def make_sheet(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.0, 1.0, 7)
    v = np.linspace(-0.8, 0.8, 6)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    zz = 0.5 * uu - 0.3 * vv + 0.15 * uu * vv
    X = np.column_stack([uu.ravel(), vv.ravel(), zz.ravel()])
    X += 0.01 * rng.normal(size=X.shape)
    return X


def test_intrinsic_hs_kd_happy_path() -> None:
    X = make_sheet()
    model = IntrinsicHSPrincipalManifold(
        IntrinsicHSConfig(
            intrinsic_dim=2,
            grid_shape=(3, 4),
            w=0.2,
            max_iter=3,
            tol=1e-6,
            store_trace=True,
            ridge=1e-10,
        )
    ).fit(X)

    result = model.result_
    assert result.vertices.ndim == 2
    assert result.vertices.shape[1] == X.shape[1]
    assert result.projected_points.shape == X.shape
    assert result.edges.ndim == 2 and result.edges.shape[1] == 2
    assert result.faces is not None
    assert result.faces.ndim == 2 and result.faces.shape[1] == 3
    assert result.cells_by_dim is not None and 1 in result.cells_by_dim and 2 in result.cells_by_dim
    assert model.intrinsic_vertices_ is not None
    assert model.intrinsic_vertices_.shape[1] == 2
    transformed = model.transform(X[:5])
    assert transformed.shape == (5, 2)
    assert np.isfinite(result.vertices).all()
    assert np.isfinite(result.projected_points).all()
    assert model.history_
    assert model.trace_


def test_singular_local_regression_guard() -> None:
    x = np.linspace(-1.0, 1.0, 20)
    X = np.column_stack([x, 2.0 * x, -x])
    model = IntrinsicHSPrincipalManifold(
        IntrinsicHSConfig(
            intrinsic_dim=2,
            grid_shape=(2, 2),
            w=0.15,
            max_iter=2,
            tol=1e-6,
            ridge=1e-12,
        )
    ).fit(X)

    assert model.vertices_ is not None
    assert np.isfinite(model.vertices_).all()
    assert np.isfinite(model.project(X)).all()
