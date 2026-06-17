from __future__ import annotations

import numpy as np

from principal_manifold._types import GraphSnapshot, copy_graph_snapshot


def test_intrinsic_types_validation() -> None:
    snapshot = GraphSnapshot(
        phase="init",
        outer_iteration=0,
        sweep=None,
        vertices=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        edges=np.array([[0, 1], [1, 2]], dtype=int),
        mean_squared_distance=0.0,
        root_mean_squared_distance=0.0,
        lambda_p=0.0,
        segments=2,
        polyline_length=2.0,
        elastic_energy=0.0,
        operation=None,
        construction_complexity=0,
        structural_complexity=3.0,
        faces=np.array([[0, 1, 2]], dtype=int),
        cells_by_dim={1: np.array([[0, 1], [1, 2]], dtype=int), 2: np.array([[0, 1, 2]], dtype=int)},
    )

    copied = copy_graph_snapshot(snapshot)
    assert copied.edges.shape == (2, 2)
    assert copied.faces is not None and copied.faces.shape == (1, 3)
    assert copied.cells_by_dim is not None
    assert set(copied.cells_by_dim.keys()) == {1, 2}
    copied.cells_by_dim[1][0, 0] = 99
    assert snapshot.cells_by_dim is not None
    assert snapshot.cells_by_dim[1][0, 0] == 0
