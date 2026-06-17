from __future__ import annotations

import numpy as np

from principal_manifold.geometry import (
    _project_onto_complex,
    _project_onto_graph_edges,
    _project_onto_surface,
)


def test_graph_projection_parity_with_dispatcher() -> None:
    X = np.array(
        [
            [0.1, 0.0],
            [0.8, 0.2],
            [1.2, -0.1],
        ],
        dtype=float,
    )
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    edges = [(0, 1), (1, 2)]

    old_proj = _project_onto_graph_edges(X, vertices, edges)
    new_proj = _project_onto_complex(
        X,
        vertices,
        edge_index_pairs=edges,
        prefer_dim=1,
    )
    assert old_proj.shape == new_proj.shape == X.shape
    assert np.allclose(old_proj, new_proj)


def test_surface_projection_parity_with_dispatcher() -> None:
    X = np.array(
        [
            [0.2, 0.2, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, -0.3],
        ],
        dtype=float,
    )
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)

    old_proj = _project_onto_surface(X, vertices, faces)
    new_proj = _project_onto_complex(
        X,
        vertices,
        faces=faces,
        prefer_dim=2,
    )
    assert old_proj.shape == new_proj.shape == X.shape
    assert np.allclose(old_proj, new_proj)


def test_degenerate_element_handling() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 1.0],
            [-1.0, 0.5],
        ],
        dtype=float,
    )
    vertices = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    edges = [(0, 1), (1, 2)]

    proj_edge = _project_onto_complex(X, vertices, edge_index_pairs=edges, prefer_dim=1)
    assert proj_edge.shape == X.shape
    assert np.all(np.isfinite(proj_edge))

    proj_fallback = _project_onto_complex(
        X,
        vertices,
        edge_index_pairs=[],
        faces=np.empty((0, 3), dtype=int),
    )
    assert proj_fallback.shape == X.shape
    assert np.all(np.isfinite(proj_fallback))
