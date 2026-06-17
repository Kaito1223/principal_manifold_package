from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np

Array = np.ndarray

def _as_float_matrix(name: str, value) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d).")
    return arr


def _normalize_method_name(method_name: str) -> str:
    normalized = str(method_name).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "ELASTICMAP": "ELASTIC_MAP",
        "ELASTIC": "ELASTIC_MAP",
        "EM": "ELASTIC_MAP",
        "PEG": "PRINCIPAL_ELASTIC_GRAPH",
        "PRINCIPAL_TREE": "PRINCIPAL_ELASTIC_GRAPH",
        "PRINCIPALGRAPH": "PRINCIPAL_ELASTIC_GRAPH",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"HS", "KK", "ELASTIC_MAP", "PRINCIPAL_ELASTIC_GRAPH"}
    if normalized not in allowed:
        raise ValueError(
            "method_name must be one of: 'HS', 'KK', 'ELASTIC_MAP', 'PRINCIPAL_ELASTIC_GRAPH'."
        )
    return normalized


def _normalize_render_mode(render_mode: str) -> str:
    normalized = str(render_mode).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "LINE": "CURVE",
        "POLYLINE": "CURVE",
        "MESH": "SURFACE",
        "MANIFOLD": "INTRINSIC",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"CURVE", "GRAPH", "SURFACE", "INTRINSIC"}
    if normalized not in allowed:
        raise ValueError(
            "render_mode must be one of: 'curve', 'graph', 'surface', 'intrinsic'."
        )
    return normalized


def _infer_render_mode(
    render_mode: Optional[str] = None,
    method_name: Optional[str] = None,
    snapshot=None,
    edges: Optional[Array] = None,
    faces: Optional[Array] = None,
    cells_by_dim: Optional[Mapping[int, Array]] = None,
) -> str:
    if render_mode is not None:
        return _normalize_render_mode(render_mode)

    if cells_by_dim is None and snapshot is not None and hasattr(snapshot, "cells_by_dim"):
        cells_by_dim = snapshot.cells_by_dim
    if cells_by_dim:
        return "INTRINSIC"

    if faces is None and snapshot is not None and hasattr(snapshot, "faces"):
        faces = snapshot.faces
    if faces is not None and np.asarray(faces, dtype=int).size > 0:
        return "SURFACE"

    if edges is None and snapshot is not None and hasattr(snapshot, "edges"):
        snapshot_edges = snapshot.edges
        if snapshot_edges is not None:
            edges = np.asarray(snapshot_edges, dtype=int)
    if edges is not None and np.asarray(edges, dtype=int).size > 0:
        return "GRAPH"

    if method_name is None:
        return "CURVE"

    method = _normalize_method_name(method_name)
    if method in {"HS", "KK"}:
        return "CURVE"
    return "GRAPH"


def _supported_plot_dimension(X: Array, vertices: Array, target_dim: Optional[int]) -> Optional[int]:
    data_dim = int(max(X.shape[1], vertices.shape[1]))
    if data_dim > 3:
        print(
            f"Input dimension is {data_dim}. Direct plotting is implemented only up to 3D; "
            "this object can still be plotted after dimensionality reduction."
        )
        return None

    if target_dim is None:
        return 2 if data_dim <= 2 else 3

    target_dim = int(target_dim)
    if target_dim not in (2, 3):
        raise ValueError("target_dim must be 2 or 3.")
    if target_dim < data_dim:
        raise ValueError(
            f"target_dim={target_dim} is too small for ambient dimension {data_dim}. "
            "Use target_dim=None or a value >= ambient dimension."
        )
    return target_dim


def _pad_to_dim(points: Array, target_dim: int) -> Array:
    points = _as_float_matrix("points", points)
    if points.shape[1] == target_dim:
        return points.copy()
    if points.shape[1] > target_dim:
        raise ValueError("Cannot truncate dimensions automatically here.")
    return np.pad(points, ((0, 0), (0, target_dim - points.shape[1])), mode="constant")


def _polyline_edges(n_vertices: int) -> Array:
    if n_vertices <= 1:
        return np.empty((0, 2), dtype=int)
    return np.column_stack([
        np.arange(n_vertices - 1, dtype=int),
        np.arange(1, n_vertices, dtype=int),
    ])


def _snapshot_edges(snapshot) -> Array:
    if hasattr(snapshot, "edges") and snapshot.edges is not None:
        edges = np.asarray(snapshot.edges, dtype=int)
        if edges.size == 0:
            return np.empty((0, 2), dtype=int)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("snapshot.edges must have shape (m, 2).")
        return edges
    vertices = _as_float_matrix("snapshot.vertices", snapshot.vertices)
    return _polyline_edges(vertices.shape[0])


def _build_snapshot_title(snapshot) -> str:
    parts = [f"{snapshot.phase}", f"outer {snapshot.outer_iteration}"]
    if getattr(snapshot, "sweep", None) is not None:
        parts.append(f"sweep {snapshot.sweep}")
    parts.append(f"segments={snapshot.segments}")
    #parts.append(f"MSD={snapshot.mean_squared_distance:.4f}")
    return " | ".join(parts)


def _get_projection_cache(snapshot):
    if not hasattr(snapshot, "_visual_cache"):
        snapshot._visual_cache = {}
    return snapshot._visual_cache


def _segment_distance_squared(points: Array, a: Array, b: Array) -> Tuple[Array, Array, Array]:
    direction = b - a
    denom = float(np.dot(direction, direction))
    if denom <= 1e-15:
        projection = np.repeat(a[None, :], points.shape[0], axis=0)
        t = np.zeros(points.shape[0], dtype=float)
        d2 = np.sum((points - projection) ** 2, axis=1)
        return d2, projection, t
    t = ((points - a[None, :]) @ direction) / denom
    t = np.clip(t, 0.0, 1.0)
    projection = a[None, :] + t[:, None] * direction[None, :]
    d2 = np.sum((points - projection) ** 2, axis=1)
    return d2, projection, t


def _project_points_onto_graph(points: Array, vertices: Array, edges: Array) -> Array:
    points = _as_float_matrix("points", points)
    vertices = _as_float_matrix("vertices", vertices)
    edges = np.asarray(edges, dtype=int)

    n_points = points.shape[0]
    if n_points == 0:
        return np.empty_like(points)
    if vertices.shape[0] == 0:
        raise ValueError("vertices must contain at least one point.")

    if len(edges) == 0:
        d2 = np.sum((points[:, None, :] - vertices[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(d2, axis=1)
        return vertices[nearest]

    best_d2 = np.full(n_points, np.inf, dtype=float)
    best_proj = np.zeros_like(points)
    for i, j in edges:
        d2, proj, _ = _segment_distance_squared(points, vertices[i], vertices[j])
        mask = d2 < best_d2
        best_d2[mask] = d2[mask]
        best_proj[mask] = proj[mask]
    return best_proj


def _get_cached_projection(X: Array, snapshot, vertices: Optional[Array] = None) -> Array:
    X = _as_float_matrix("X", X)
    if vertices is None:
        vertices = snapshot.vertices
    vertices = _as_float_matrix("vertices", vertices)
    edges = _snapshot_edges(snapshot)

    cache = _get_projection_cache(snapshot)
    key = (
        X.shape,
        tuple(np.asarray(vertices, dtype=float).ravel()),
        tuple(np.asarray(edges, dtype=int).ravel()),
    )
    if cache.get("projection_key") == key:
        return cache["projection"]

    projection = _project_points_onto_graph(X, vertices, edges)
    cache["projection_key"] = key
    cache["projection"] = projection
    return projection


def _visual_options_for_method(method_name: str, show_projections: bool):
    method = _normalize_method_name(method_name)

    if method == "HS":
        return {
            "need_projection": True,
            "show_projections": True,
            "show_projected_points": False,
            "show_structure_segments": True,
            "show_structure_vertices": False,
        }

    return _visual_options_for_render_mode("graph", show_projections, method_name=method)


def _visual_options_for_render_mode(
    render_mode: str,
    show_projections: bool,
    method_name: Optional[str] = None,
):
    resolved_mode = _normalize_render_mode(render_mode)

    if method_name is not None and _normalize_method_name(method_name) == "HS":
        return {
            "need_projection": True,
            "show_projections": True,
            "show_projected_points": False,
            "show_structure_segments": True,
            "show_structure_vertices": False,
        }

    return {
        "need_projection": bool(show_projections) and resolved_mode in {"CURVE", "GRAPH"},
        "show_projections": bool(show_projections) and resolved_mode in {"CURVE", "GRAPH"},
        "show_projected_points": False,
        "show_structure_segments": True,
        "show_structure_vertices": True,
    }
