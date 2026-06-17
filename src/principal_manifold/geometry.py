from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray

def _as_2d_float_array(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Expected a 2D array of shape (n_samples, n_features).")
    return X


def _dataset_radius(X: Array, mean: Array) -> float:
    return float(np.max(np.linalg.norm(X - mean[None, :], axis=1)))


def _polyline_length(vertices: Array) -> float:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(vertices, axis=0), axis=1)))


def _pca_init_line(X: Array) -> Tuple[Array, Array, Array]:
    mean = X.mean(axis=0)
    centered = X - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction = direction / max(np.linalg.norm(direction), 1e-15)
    lam = centered @ direction
    return mean, direction, lam


def _orient_pc_component(component: Array, scores: Array) -> Tuple[Array, Array]:
    component = np.asarray(component, dtype=float)
    scores = np.asarray(scores, dtype=float)
    anchor = int(np.argmax(np.abs(component)))
    if component[anchor] < 0.0:
        return -component, -scores
    return component, scores


def _sort_nodes_by_lambda(lam: Array, vertices: Array) -> Tuple[Array, Array]:
    order = np.argsort(np.asarray(lam, dtype=float))
    lam_sorted = np.asarray(lam, dtype=float)[order]
    vertices_sorted = np.asarray(vertices, dtype=float)[order]
    return lam_sorted, vertices_sorted


def _choose_window_size(w: float, n_samples: int) -> int:
    k = int(np.floor(float(w) * n_samples))
    k = max(3, min(k, n_samples))
    if k % 2 == 0 and k < n_samples:
        k += 1
    if k > n_samples:
        k = n_samples
    return int(k)


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


def _project_onto_polyline(X: Array, vertices: Array) -> Tuple[Array, Array, Array, Array]:
    n_samples = X.shape[0]
    n_segments = vertices.shape[0] - 1

    vertex_d2 = np.stack([np.sum((X - v[None, :]) ** 2, axis=1) for v in vertices], axis=1)
    segment_d2_list: List[Array] = []
    segment_projection_list: List[Array] = []
    segment_t_list: List[Array] = []
    for i in range(n_segments):
        d2, projection, t = _segment_distance_squared(X, vertices[i], vertices[i + 1])
        segment_d2_list.append(d2)
        segment_projection_list.append(projection)
        segment_t_list.append(t)

    if segment_d2_list:
        segment_d2 = np.stack(segment_d2_list, axis=1)
        all_d2 = np.concatenate([vertex_d2, segment_d2], axis=1)
    else:
        all_d2 = vertex_d2

    labels = np.argmin(all_d2, axis=1)
    kinds = np.where(labels < vertices.shape[0], "vertex", "segment")
    indices = np.where(labels < vertices.shape[0], labels, labels - vertices.shape[0])

    projected = np.empty_like(X)
    arc = np.empty(n_samples, dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))])

    for idx in range(n_samples):
        if kinds[idx] == "vertex":
            j = int(indices[idx])
            projected[idx] = vertices[j]
            arc[idx] = cumulative[j]
        else:
            j = int(indices[idx])
            projected[idx] = segment_projection_list[j][idx]
            seg_len = cumulative[j + 1] - cumulative[j]
            arc[idx] = cumulative[j] + segment_t_list[j][idx] * seg_len

    return projected, arc, kinds.astype(object), indices.astype(int)


def _project_to_polyline_parameter(X: Array, vertices: Array) -> Tuple[Array, Array, Array]:
    """Return segment-coordinate parameter t together with projections and d^2.

    If there are L vertices, t lies in [0, L-1] and equals segment_index + alpha.
    """
    A = vertices[:-1]
    B = vertices[1:]
    V = B - A
    VV = np.sum(V * V, axis=1) + 1e-12

    XA = X[:, None, :] - A[None, :, :]
    dot = np.sum(XA * V[None, :, :], axis=2)
    alpha = dot / VV[None, :]
    alpha = np.clip(alpha, 0.0, 1.0)

    Pseg = A[None, :, :] + alpha[:, :, None] * V[None, :, :]
    diff = X[:, None, :] - Pseg
    dist2 = np.sum(diff * diff, axis=2)

    segment_index = np.argmin(dist2, axis=1)
    projected = Pseg[np.arange(X.shape[0]), segment_index, :]
    d2 = dist2[np.arange(X.shape[0]), segment_index]
    t = segment_index.astype(np.float64) + alpha[np.arange(X.shape[0]), segment_index]
    return t, projected, d2


def _polyline_lambda_from_segment_parameter(t: Array, lambda_nodes: Array) -> Array:
    t = np.asarray(t, dtype=float)
    lambda_nodes = np.asarray(lambda_nodes, dtype=float)
    if lambda_nodes.shape[0] < 2:
        return np.repeat(lambda_nodes[0], t.shape[0])

    left = np.floor(t).astype(np.int64)
    left = np.clip(left, 0, lambda_nodes.shape[0] - 2)
    alpha = t - left
    return (1.0 - alpha) * lambda_nodes[left] + alpha * lambda_nodes[left + 1]


def _mean_squared_distance_to_polyline(X: Array, vertices: Array) -> float:
    projected, _, _, _ = _project_onto_polyline(X, vertices)
    return float(np.mean(np.sum((X - projected) ** 2, axis=1)))

def _initialize_nodes_on_first_pc(X: Array, n_nodes: int) -> Array:
    mean = X.mean(axis=0)
    centered = X - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction = direction / max(np.linalg.norm(direction), 1e-15)
    scores = centered @ direction
    grid = np.linspace(scores.min(), scores.max(), int(n_nodes))
    return mean[None, :] + grid[:, None] * direction[None, :]


def _initialize_intrinsic_coordinates_on_first_k_pcs(X: Array, k: int) -> Array:
    X = _as_2d_float_array(X)
    if k < 1:
        raise ValueError("k must be at least 1.")
    if k > X.shape[1]:
        raise ValueError(f"k must be <= n_features ({X.shape[1]}).")

    mean = X.mean(axis=0)
    centered = X - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = np.asarray(vt[:k], dtype=float).copy()
    scores = centered @ basis.T

    for idx in range(k):
        basis[idx], scores[:, idx] = _orient_pc_component(basis[idx], scores[:, idx])

    return scores


def _initialize_surface_grid_on_first_two_pcs(X: Array, n_u: int, n_v: int) -> Array:
    X = _as_2d_float_array(X)
    if X.shape[1] < 2:
        raise ValueError("ElasticSurfacePrincipalManifold requires at least 2 features.")
    mean = X.mean(axis=0)
    centered = X - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: min(2, vt.shape[0])]
    basis = basis / np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-15)
    scores = centered @ basis.T
    u_grid = np.linspace(scores[:, 0].min(), scores[:, 0].max(), int(n_u))
    v_grid = np.linspace(scores[:, 1].min(), scores[:, 1].max(), int(n_v))
    uu, vv = np.meshgrid(u_grid, v_grid, indexing="ij")
    offsets = uu[..., None] * basis[0][None, None, :] + vv[..., None] * basis[1][None, None, :]
    return (mean[None, None, :] + offsets).reshape(-1, X.shape[1])


def _assign_points_to_nodes(X: Array, vertices: Array) -> Array:
    return np.argmin(_squared_distances(X, vertices), axis=1)


def _squared_distances(X: Array, Y: Array) -> Array:
    xx = np.sum(X * X, axis=1, keepdims=True)
    yy = np.sum(Y * Y, axis=1, keepdims=True).T
    xy = X @ Y.T
    return xx + yy - 2.0 * xy

def _graph_total_edge_length(vertices: Array, edge_index_pairs: Sequence[Tuple[int, int]]) -> float:
    if len(edge_index_pairs) == 0:
        return 0.0
    return float(sum(np.linalg.norm(vertices[i] - vertices[j]) for i, j in edge_index_pairs))


def _project_onto_complex(
    X: Array,
    vertices: Array,
    edge_index_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    faces: Optional[Array] = None,
    prefer_dim: Optional[int] = None,
) -> Array:
    X = _as_2d_float_array(X)
    vertices = _as_2d_float_array(vertices)

    edge_pairs: Sequence[Tuple[int, int]] = [] if edge_index_pairs is None else edge_index_pairs
    face_array = np.empty((0, 3), dtype=int) if faces is None else np.asarray(faces, dtype=int)

    use_faces = (prefer_dim == 2) or (prefer_dim is None and face_array.size > 0)
    use_edges = (prefer_dim == 1) or (prefer_dim is None and len(edge_pairs) > 0)

    if use_faces and face_array.size > 0:
        best_d2 = np.full(X.shape[0], np.inf, dtype=float)
        best_proj = np.zeros_like(X)
        for i, j, k in face_array:
            d2, proj = _triangle_distance_squared(X, vertices[i], vertices[j], vertices[k])
            mask = d2 < best_d2
            best_d2[mask] = d2[mask]
            best_proj[mask] = proj[mask]
        return best_proj

    if use_edges and len(edge_pairs) > 0:
        best_d2 = np.full(X.shape[0], np.inf, dtype=float)
        best_proj = np.zeros_like(X)
        for i, j in edge_pairs:
            d2, proj, _ = _segment_distance_squared(X, vertices[i], vertices[j])
            mask = d2 < best_d2
            best_d2[mask] = d2[mask]
            best_proj[mask] = proj[mask]
        return best_proj

    nearest = _assign_points_to_nodes(X, vertices)
    return vertices[nearest]


def _project_onto_graph_edges(X: Array, vertices: Array, edge_index_pairs: Sequence[Tuple[int, int]]) -> Array:
    return _project_onto_complex(
        X=X,
        vertices=vertices,
        edge_index_pairs=edge_index_pairs,
        prefer_dim=1,
    )



def _mean_squared_distance_to_graph(X: Array, vertices: Array, edges: Sequence[Tuple[int, int]]) -> float:
    projected = _project_onto_graph_edges(X, vertices, edges)
    return float(np.mean(np.sum((X - projected) ** 2, axis=1)))


def _triangle_distance_squared(points: Array, a: Array, b: Array, c: Array) -> Tuple[Array, Array]:
    points = _as_2d_float_array(points)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ab = b - a
    ac = c - a
    gram = np.array(
        [
            [float(np.dot(ab, ab)), float(np.dot(ab, ac))],
            [float(np.dot(ab, ac)), float(np.dot(ac, ac))],
        ],
        dtype=float,
    )
    rhs = np.column_stack([(points - a[None, :]) @ ab, (points - a[None, :]) @ ac])
    if abs(np.linalg.det(gram)) > 1e-15:
        weights = np.linalg.solve(gram, rhs.T).T
        u = weights[:, 0]
        v = weights[:, 1]
        inside = (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0)
        plane_projection = a[None, :] + u[:, None] * ab[None, :] + v[:, None] * ac[None, :]
    else:
        inside = np.zeros(points.shape[0], dtype=bool)
        plane_projection = np.repeat(a[None, :], points.shape[0], axis=0)

    best_d2 = np.full(points.shape[0], np.inf, dtype=float)
    best_proj = np.zeros_like(points)

    if np.any(inside):
        inside_d2 = np.sum((points[inside] - plane_projection[inside]) ** 2, axis=1)
        best_d2[inside] = inside_d2
        best_proj[inside] = plane_projection[inside]

    for p0, p1 in ((a, b), (b, c), (c, a)):
        edge_d2, edge_proj, _ = _segment_distance_squared(points, p0, p1)
        mask = edge_d2 < best_d2
        best_d2[mask] = edge_d2[mask]
        best_proj[mask] = edge_proj[mask]

    return best_d2, best_proj


def _project_onto_surface(X: Array, vertices: Array, faces: Array) -> Array:
    return _project_onto_complex(
        X=X,
        vertices=vertices,
        faces=faces,
        prefer_dim=2,
    )


def _mean_squared_distance_to_surface(X: Array, vertices: Array, faces: Array) -> float:
    projected = _project_onto_surface(X, vertices, faces)
    return float(np.mean(np.sum((X - projected) ** 2, axis=1)))
