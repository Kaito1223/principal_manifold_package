from __future__ import annotations

import numpy as np

from principal_manifold import (
    HSConfig,
    HSPrincipalCurve,
    KeglKrzyzakConfig,
    KeglKrzyzakPrincipalCurve,
    OptimizerConfig,
    ElasticGraphConfig,
    ElasticGraphPrincipalCurve,
    ElasticPrincipalGraphConfig,
    ElasticPrincipalGraph,
    PrincipalManifoldRegressor,
)


def make_line_data(seed=0):
    rng = np.random.default_rng(seed)

    X = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    y = y + 0.02 * rng.normal(size=y.shape)

    return X, y


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def test_hs_regressor_predicts_line():
    X, y = make_line_data()

    reg = PrincipalManifoldRegressor(
        HSPrincipalCurve(
            HSConfig(
                w=0.15,
                max_iter=3,
                tol=1e-5,
            )
        )
    )

    reg.fit(X, y)
    y_pred = reg.predict(X)

    assert y_pred.shape == y.shape
    assert rmse(y, y_pred) < 0.35


def test_kk_regressor_predicts_line():
    X, y = make_line_data()

    reg = PrincipalManifoldRegressor(
        KeglKrzyzakPrincipalCurve(
            KeglKrzyzakConfig(
                max_segments=4,
                optimizer=OptimizerConfig(
                    max_inner_sweeps=5,
                    max_vertex_iterations=5,
                ),
            )
        )
    )

    reg.fit(X, y)
    y_pred = reg.predict(X)

    assert y_pred.shape == y.shape
    assert rmse(y, y_pred) < 0.35


def test_elastic_chain_regressor_predicts_line():
    X, y = make_line_data()

    reg = PrincipalManifoldRegressor(
        ElasticGraphPrincipalCurve(
            ElasticGraphConfig(
                lam=0.01,
                mu=0.01,
                n_nodes=8,
                max_iter=10,
                tol=1e-6,
            )
        )
    )

    reg.fit(X, y)
    y_pred = reg.predict(X)

    assert y_pred.shape == y.shape
    assert rmse(y, y_pred) < 0.35


def test_elastic_graph_regressor_predicts_line():
    X, y = make_line_data()

    reg = PrincipalManifoldRegressor(
        ElasticPrincipalGraph(
            ElasticPrincipalGraphConfig(
                lam=0.01,
                mu=0.01,
                init_nodes=2,
                max_nodes=6,
                passes=2,
                optimizer_max_iter=10,
                optimizer_tol=1e-6,
                store_trace=True,
            )
        )
    )

    reg.fit(X, y)
    y_pred = reg.predict(X)

    assert y_pred.shape == y.shape
    assert rmse(y, y_pred) < 0.45


def test_predict_details():
    X, y = make_line_data()

    reg = PrincipalManifoldRegressor(
        KeglKrzyzakPrincipalCurve(
            KeglKrzyzakConfig(
                max_segments=4,
                optimizer=OptimizerConfig(
                    max_inner_sweeps=5,
                    max_vertex_iterations=5,
                ),
            )
        )
    )

    reg.fit(X, y)
    details = reg.predict(X[:10], return_details=True)

    assert details.y_pred.shape == (10,)
    assert details.projected_points.shape == (10, 2)
    assert details.x_squared_distance.shape == (10,)
    assert details.nearest_edge_index.shape == (10,)
    assert details.edge_t.shape == (10,)