from __future__ import annotations

import numpy as np
import pytest

from principal_manifold import (
    IntrinsicElasticMapConfig,
    IntrinsicElasticMapPrincipalManifold,
)


def make_intrinsic_sheet_3d(
    n_u: int = 9,
    n_v: int = 8,
    noise: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.0, 1.0, n_u)
    v = np.linspace(-0.7, 0.7, n_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    zz = 0.4 * uu - 0.25 * vv + 0.1 * uu * vv
    X = np.column_stack([uu.ravel(), vv.ravel(), zz.ravel()])
    X += noise * rng.normal(size=X.shape)
    return X


def test_intrinsic_elastic_map_fit_shapes_and_finiteness() -> None:
    X = make_intrinsic_sheet_3d()
    model = IntrinsicElasticMapPrincipalManifold(
        IntrinsicElasticMapConfig(
            intrinsic_dim=2,
            grid_shape=(3, 4),
            lam=0.01,
            mu=0.02,
            optimizer_max_iter=4,
            optimizer_tol=1e-6,
            softening=(1.0,),
            topology_epochs=1,
            store_trace=True,
        )
    ).fit(X)

    result = model.result_
    assert result.vertices.ndim == 2
    assert result.vertices.shape[1] == X.shape[1]
    assert result.projected_points.shape == X.shape
    assert result.edges.ndim == 2 and result.edges.shape[1] == 2
    assert result.faces is not None
    assert result.faces.ndim == 2 and result.faces.shape[1] == 3
    assert result.cells_by_dim is not None
    assert 1 in result.cells_by_dim
    assert 2 in result.cells_by_dim
    assert np.isfinite(result.vertices).all()
    assert np.isfinite(result.projected_points).all()
    assert model.history_
    assert np.isfinite(model.history_[-1]["elastic_energy"])
    assert np.isfinite(model.history_[-1]["mean_squared_distance"])
    assert model.trace_


def test_nan_guard_and_abort() -> None:
    X = make_intrinsic_sheet_3d()
    X[0, 0] = np.nan
    model = IntrinsicElasticMapPrincipalManifold(
        IntrinsicElasticMapConfig(
            intrinsic_dim=2,
            grid_shape=(3, 3),
            optimizer_max_iter=3,
            softening=(1.0,),
            topology_epochs=0,
        )
    )

    with pytest.raises(FloatingPointError, match="Non-finite X"):
        model.fit(X)
