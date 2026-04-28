from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ._utils import _as_float_matrix, _build_snapshot_title, _snapshot_edges

Array = np.ndarray

def _build_adjacency(n_nodes: int, edges: Array):
    adj = [[] for _ in range(n_nodes)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _is_tree(n_nodes: int, edges: Array) -> bool:
    if n_nodes == 0:
        return False
    if len(edges) != n_nodes - 1:
        return False
    adj = _build_adjacency(n_nodes, edges)
    seen = np.zeros(n_nodes, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return bool(np.all(seen))


def _choose_root(vertices: Array, edges: Array) -> int:
    adj = _build_adjacency(vertices.shape[0], edges)
    degrees = np.array([len(a) for a in adj], dtype=int)
    candidates = np.where(degrees == degrees.max())[0]
    if len(candidates) == 1:
        return int(candidates[0])
    centroid = vertices.mean(axis=0)
    d2 = np.sum((vertices[candidates] - centroid[None, :]) ** 2, axis=1)
    return int(candidates[np.argmin(d2)])


def _pca_plane_basis(points: Array) -> Array:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    if basis.shape[1] < 2:
        basis = np.pad(basis, ((0, 0), (0, 2 - basis.shape[1])), mode="constant")
    return basis


def _angles_from_center(vertices: Array, center: int, neighbors: Sequence[int], basis2: Array) -> Array:
    if len(neighbors) == 0:
        return np.empty(0, dtype=float)
    vecs = vertices[np.asarray(neighbors)] - vertices[center]
    proj = vecs @ basis2
    return np.arctan2(proj[:, 1], proj[:, 0])


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _place_children_equiangular(pos: Array, vertices_hd: Array, adj, node: int, parent: Optional[int], incoming_angle: Optional[float], basis2: Array) -> None:
    neighbors = list(adj[node])
    if parent is None:
        local_angles = _angles_from_center(vertices_hd, node, neighbors, basis2)
        ordered = [neighbors[i] for i in np.argsort(local_angles)]
        k = len(ordered)
        if k == 0:
            return
        incident_angles = 2.0 * np.pi * np.arange(k) / k
    else:
        other = [v for v in neighbors if v != parent]
        local_angles = _angles_from_center(vertices_hd, node, other, basis2)
        ordered_other = [other[i] for i in np.argsort(local_angles)]
        ordered = [parent] + ordered_other
        k = len(ordered)
        incident_angles = incoming_angle + 2.0 * np.pi * np.arange(k) / k

    for idx, nbr in enumerate(ordered):
        if nbr == parent:
            continue
        length = float(np.linalg.norm(vertices_hd[nbr] - vertices_hd[node]))
        theta = incident_angles[idx]
        pos[nbr] = pos[node] + length * np.array([np.cos(theta), np.sin(theta)])
        _place_children_equiangular(pos, vertices_hd, adj, nbr, node, _wrap_angle(theta + np.pi), basis2)


def _metro_layout_tree(vertices_hd: Array, edges: Array, X: Optional[Array] = None) -> Array:
    vertices_hd = _as_float_matrix("vertices_hd", vertices_hd)
    edges = np.asarray(edges, dtype=int)
    if not _is_tree(vertices_hd.shape[0], edges):
        raise ValueError("Metro-map layout requires a connected acyclic tree.")

    order_points = vertices_hd if X is None else np.vstack([X, vertices_hd])
    basis2 = _pca_plane_basis(order_points)
    root = _choose_root(vertices_hd, edges)
    adj = _build_adjacency(vertices_hd.shape[0], edges)

    pos = np.zeros((vertices_hd.shape[0], 2), dtype=float)
    _place_children_equiangular(pos, vertices_hd, adj, root, None, None, basis2)
    return pos


def _nearest_node_indices(X: Array, vertices: Array) -> Array:
    d2 = np.sum((X[:, None, :] - vertices[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1)


def plot_principal_tree_snapshot_2d(X, snapshot, ax=None, title: Optional[str] = None):
    X = _as_float_matrix("X", X)
    vertices = _as_float_matrix("snapshot.vertices", snapshot.vertices)
    edges = _snapshot_edges(snapshot)

    if vertices.shape[1] > 3 or X.shape[1] > 3:
        print(
            f"Input dimension is {max(X.shape[1], vertices.shape[1])}. Direct plotting is implemented only up to 3D; "
            "this object can still be plotted after dimensionality reduction."
        )
        return None

    if title is None:
        title = _build_snapshot_title(snapshot)

    layout = _metro_layout_tree(vertices, edges, X=X)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    nearest = _nearest_node_indices(X, vertices)
    counts = np.bincount(nearest, minlength=vertices.shape[0]).astype(float)
    sizes = 40.0 + 240.0 * (counts / max(np.max(counts), 1.0))

    if len(edges) > 0:
        segs = np.stack([layout[edges[:, 0]], layout[edges[:, 1]]], axis=1)
        ax.add_collection(LineCollection(segs, linewidths=2, alpha=0.95))

    ax.scatter(layout[:, 0], layout[:, 1], s=sizes, alpha=0.95, zorder=3)
    for i, (x, y) in enumerate(layout):
        ax.text(x, y, str(i), fontsize=8, ha="center", va="center", zorder=4)

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("metro-map x")
    ax.set_ylabel("metro-map y")
    ax.set_title(title)
    return ax


def save_principal_tree_trace_frames_2d(X, snapshots, output_dir="principal_tree_steps", dpi: int = 150):
    if len(snapshots) == 0:
        raise ValueError("snapshots must contain at least one snapshot.")

    X = _as_float_matrix("X", X)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, snapshot in enumerate(snapshots):
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_principal_tree_snapshot_2d(X, snapshot, ax=ax)

        filename = f"{idx:03d}_outer_{snapshot.outer_iteration:03d}_{snapshot.phase}"
        if getattr(snapshot, "sweep", None) is not None:
            filename += f"_sweep_{snapshot.sweep:03d}"
        filename += ".png"

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return output_dir

