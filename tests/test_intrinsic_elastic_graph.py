from __future__ import annotations

import numpy as np

from principal_manifold.elastic.principal_graph import ElasticPrincipalGraph, ElasticPrincipalGraphConfig


def _branching_data() -> np.ndarray:
    trunk = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    branch_up = np.array(
        [
            [0.25, 0.25],
            [0.5, 0.5],
            [0.75, 0.75],
        ],
        dtype=float,
    )
    branch_down = np.array(
        [
            [0.25, -0.25],
            [0.5, -0.5],
            [0.75, -0.75],
        ],
        dtype=float,
    )
    return np.vstack([trunk, branch_up, branch_down])


def test_intrinsic_k_happy_path_graph() -> None:
    X = _branching_data()
    model = ElasticPrincipalGraph(
        ElasticPrincipalGraphConfig(
            intrinsic_topology_enabled=True,
            intrinsic_topology_ops=("split", "prune"),
            intrinsic_topology_max_ops_per_epoch=2,
            init_nodes=2,
            max_nodes=5,
            passes=2,
            optimizer_max_iter=25,
            softening=(10.0, 1.0),
        )
    )

    result = model.fit_result(X)

    assert result.vertices.shape[1] == X.shape[1]
    assert result.edges.shape[1] == 2
    assert result.vertices.shape[0] >= 2
    assert np.all(np.isfinite(result.vertices))
    assert np.all(np.isfinite(result.projected_points))
    assert np.isfinite(model.score(X))
    assert len(result.history) > 0
    assert any(entry["operation"] == "add_node,bisect_edge" for entry in result.history)


def test_intrinsic_topology_op_cap_enforced() -> None:
    X = _branching_data()
    model = ElasticPrincipalGraph(
        ElasticPrincipalGraphConfig(
            intrinsic_topology_enabled=True,
            intrinsic_topology_ops=("split", "prune"),
            intrinsic_topology_max_ops_per_epoch=1,
            init_nodes=2,
            max_nodes=4,
            passes=2,
            optimizer_max_iter=20,
            softening=(10.0, 1.0),
            store_trace=True,
        )
    )

    model.fit(X)

    accepted_ops = [
        (snapshot.outer_iteration, snapshot.operation)
        for snapshot in model.trace_
        if snapshot.phase == "accepted" and snapshot.operation is not None
    ]
    assert accepted_ops
    accepted_outer_iterations = [outer_iteration for outer_iteration, _ in accepted_ops]
    accepted_operation_labels = [operation for _, operation in accepted_ops]
    assert accepted_outer_iterations == list(range(1, len(accepted_ops) + 1))
    assert accepted_operation_labels == [
        "add_node,bisect_edge" if idx % 2 == 0 else "remove_leaf,remove_edge"
        for idx in range(len(accepted_operation_labels))
    ]
    assert model.graph_ is not None
    assert model.graph_.n_nodes <= 4
