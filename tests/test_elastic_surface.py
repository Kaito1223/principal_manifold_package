from __future__ import annotations

import numpy as np

from principal_manifold import (
    ElasticSurfaceConfig,
    ElasticSurfacePrincipalManifold,
)


def make_plane_patch_3d(
    n_u: int = 8,
    n_v: int = 7,
    noise: float = 0.03,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.0, 1.0, n_u)
    v = np.linspace(-0.8, 0.8, n_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    zz = 0.35 * uu - 0.2 * vv
    X = np.column_stack([uu.ravel(), vv.ravel(), zz.ravel()])
    X += noise * rng.normal(size=X.shape)
    return X


def test_surface_public_fit_shapes() -> None:
    X = make_plane_patch_3d()
    model = ElasticSurfacePrincipalManifold(
        ElasticSurfaceConfig(
            n_u=4,
            n_v=5,
            max_iter=5,
            tol=1e-6,
            softening=(1.0,),
            store_trace=True,
        )
    ).fit(X)

    result = model.result_
    assert result.vertices.shape == (20, 3)
    assert result.edges.ndim == 2 and result.edges.shape[1] == 2
    assert result.faces.ndim == 2 and result.faces.shape[1] == 3
    assert result.edges.shape[0] == 43
    assert result.faces.shape[0] == 24
    assert np.min(result.edges) >= 0
    assert np.max(result.edges) < result.vertices.shape[0]
    assert np.min(result.faces) >= 0
    assert np.max(result.faces) < result.vertices.shape[0]
    assert result.projected_points.shape == X.shape
    assert len(result.history) > 0
    assert len(result.trace) > 0


def test_surface_projection_and_score_are_finite() -> None:
    X = make_plane_patch_3d(seed=1)
    model = ElasticSurfacePrincipalManifold(
        ElasticSurfaceConfig(
            n_u=3,
            n_v=4,
            max_iter=4,
            tol=1e-6,
            softening=(1.0,),
        )
    ).fit(X)

    projected = model.project(X)
    score = model.score(X)

    assert projected.shape == X.shape
    assert np.isfinite(projected).all()
    assert np.isfinite(score)


def test_surface_history_energy_is_finite() -> None:
    X = make_plane_patch_3d(seed=2)
    model = ElasticSurfacePrincipalManifold(
        ElasticSurfaceConfig(
            n_u=3,
            n_v=3,
            lam=0.01,
            mu=0.01,
            max_iter=3,
            tol=1e-6,
            softening=(1.0,),
        )
    ).fit(X)

    assert model.history_
    last = model.history_[-1]
    assert np.isfinite(last["elastic_energy"])
    assert np.isfinite(last["mean_squared_distance"])
    assert np.isfinite(last["root_mean_squared_distance"])


def test_surface_vertices_change_after_tiny_optimization() -> None:
    X = make_plane_patch_3d(seed=3)
    model = ElasticSurfacePrincipalManifold(
        ElasticSurfaceConfig(
            n_u=4,
            n_v=4,
            max_iter=2,
            tol=1e-6,
            softening=(1.0,),
            store_trace=True,
        )
    )

    model.fit(X)

    initial = np.asarray(model.trace_[0].vertices, dtype=float)
    final = np.asarray(model.vertices_, dtype=float)
    assert initial.shape == final.shape
    assert np.max(np.linalg.norm(final - initial, axis=1)) > 1e-8


def test_surface_requires_at_least_two_features() -> None:
    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    model = ElasticSurfacePrincipalManifold(
        ElasticSurfaceConfig(
            n_u=3,
            n_v=4,
            max_iter=2,
            tol=1e-6,
            softening=(1.0,),
        )
    )

    try:
        model.fit(X)
    except ValueError as exc:
        assert "at least 2 features" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ambient dimension < 2.")
