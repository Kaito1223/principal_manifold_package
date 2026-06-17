import numpy as np

from principal_manifold import (
    HSConfig,
    HSPrincipalCurve,
    KeglKrzyzakConfig,
    KeglKrzyzakPrincipalCurve,
    ElasticGraphConfig,
    ElasticGraphPrincipalCurve,
    ElasticPrincipalGraphConfig,
    ElasticPrincipalGraph,
    ElasticSurfaceConfig,
    ElasticSurfacePrincipalManifold,
)


def test_public_imports():
    assert HSPrincipalCurve is not None
    assert KeglKrzyzakPrincipalCurve is not None
    assert ElasticGraphPrincipalCurve is not None
    assert ElasticPrincipalGraph is not None
    assert ElasticSurfacePrincipalManifold is not None


def test_hs_smoke_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    model = HSPrincipalCurve(HSConfig(max_iter=1, store_trace=True)).fit(X)
    assert model.vertices_ is not None
    assert model.result_.vertices.shape[1] == 2
