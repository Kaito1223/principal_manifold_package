from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import (
    _as_2d_float_array,
    _assign_points_to_nodes,
    _graph_total_edge_length,
    _project_onto_graph_edges,
)

Array = np.ndarray

@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    lam: float


@dataclass(frozen=True)
class Star:
    center: int
    leaves: Tuple[int, ...]
    mu: float


@dataclass
class FixedElasticGraphResult:
    vertices: Array
    assignments: Array
    node_weights: Array
    energy: float
    n_iter: int
    converged: bool


class FixedElasticGraphOptimizer:
    """Shared fixed-topology optimizer for elastic graphs.

    The optimizer itself is general: it can work with any supplied set of
    stars, including 2-stars, 3-stars, 4-stars, and so on.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if self.tol < 0:
            raise ValueError("tol must be nonnegative.")

    def optimize(
        self,
        X: Array,
        vertices: Array,
        edges: List[Edge],
        stars: List[Star],
        sample_weight: Optional[Array] = None,
        return_history: bool = False,
        allowed_k_stars: Optional[Tuple[int, ...]] = None,
    ) -> FixedElasticGraphResult | Tuple[FixedElasticGraphResult, List[Dict[str, object]]]:
        X = _as_2d_float_array(X)
        Y = np.asarray(vertices, dtype=float).copy()
        if Y.ndim != 2:
            raise ValueError("vertices must be a 2D array.")
        if Y.shape[0] < 1:
            raise ValueError("vertices must contain at least one node.")
        if Y.shape[1] != X.shape[1]:
            raise ValueError("vertices and X must have the same number of features.")

        w = _validate_sample_weight(X, sample_weight)
        total_weight = float(np.sum(w))
        if total_weight <= 0:
            raise ValueError("sum of sample_weight must be positive.")

        _validate_edges_and_stars(Y.shape[0], edges, stars)
        filtered_stars = _filter_stars_by_allowed_k(stars, allowed_k_stars)

        history: List[Dict[str, object]] = []
        prev_energy: Optional[float] = None
        converged = False

        for sweep in range(1, self.max_iter + 1):
            assignment = _assign_points_to_nodes(X, Y)
            updated = _solve_paper_linear_system_weighted(
                X=X,
                vertices=Y,
                assignment=assignment,
                edges=edges,
                stars=filtered_stars,
                sample_weight=w,
            )
            updated_assignment = _assign_points_to_nodes(X, updated)
            energy = _elastic_energy_from_node_assignments_weighted(
                X=X,
                vertices=updated,
                assignment=updated_assignment,
                edges=edges,
                stars=filtered_stars,
                sample_weight=w,
            )

            if return_history:
                node_msd = _node_mean_squared_distance_weighted(X, updated, updated_assignment, w)
                edge_msd = _mean_squared_distance_to_graph_weighted(
                    X=X,
                    vertices=updated,
                    edge_index_pairs=[(e.i, e.j) for e in edges],
                    sample_weight=w,
                )
                history.append(
                    {
                        "sweep": int(sweep),
                        "vertices": updated.copy(),
                        "assignments": updated_assignment.copy(),
                        "node_mean_squared_distance": float(node_msd),
                        "mean_squared_distance": float(edge_msd),
                        "polyline_length": float(_graph_total_edge_length(updated, [(e.i, e.j) for e in edges])),
                        "elastic_energy": float(energy),
                    }
                )

            if prev_energy is not None and abs(prev_energy - energy) <= self.tol * max(1.0, abs(prev_energy)):
                Y = updated
                converged = True
                break

            Y = updated
            prev_energy = energy

        final_assignment = _assign_points_to_nodes(X, Y)
        final_energy = _elastic_energy_from_node_assignments_weighted(
            X=X,
            vertices=Y,
            assignment=final_assignment,
            edges=edges,
            stars=filtered_stars,
            sample_weight=w,
        )
        node_weights = np.bincount(final_assignment, weights=w, minlength=Y.shape[0]).astype(float)
        result = FixedElasticGraphResult(
            vertices=Y,
            assignments=final_assignment,
            node_weights=node_weights,
            energy=float(final_energy),
            n_iter=int(len(history) if return_history else sweep),
            converged=bool(converged),
        )

        if return_history:
            return result, history
        return result


def _validate_allowed_k_stars(allowed_k_stars: Optional[Tuple[int, ...]]) -> None:
    if allowed_k_stars is None:
        return
    if len(allowed_k_stars) == 0:
        raise ValueError("allowed_k_stars must be None or a nonempty tuple of integers.")
    for k in allowed_k_stars:
        if int(k) != k or k < 2:
            raise ValueError("allowed_k_stars must contain integers >= 2.")


def _filter_stars_by_allowed_k(stars: List[Star], allowed_k_stars: Optional[Tuple[int, ...]]) -> List[Star]:
    _validate_allowed_k_stars(allowed_k_stars)
    if allowed_k_stars is None:
        return list(stars)
    allowed = {int(k) for k in allowed_k_stars}
    return [star for star in stars if len(star.leaves) in allowed]


def _as_2d_float_array(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Expected a 2D array of shape (n_samples, n_features).")
    return X


def _validate_sample_weight(X: Array, sample_weight: Optional[Array]) -> Array:
    if sample_weight is None:
        return np.ones(X.shape[0], dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    if w.shape != (X.shape[0],):
        raise ValueError("sample_weight must have shape (n_samples,).")
    if np.any(w < 0):
        raise ValueError("sample_weight must be nonnegative.")
    return w


def _validate_edges_and_stars(n_nodes: int, edges: List[Edge], stars: List[Star]) -> None:
    for edge in edges:
        if not (0 <= edge.i < n_nodes and 0 <= edge.j < n_nodes):
            raise ValueError("edge endpoints are out of bounds.")
        if edge.i == edge.j:
            raise ValueError("self-loop edges are not allowed.")
        if edge.lam < 0:
            raise ValueError("edge elasticity must be nonnegative.")
    for star in stars:
        if not (0 <= star.center < n_nodes):
            raise ValueError("star center is out of bounds.")
        if star.mu < 0:
            raise ValueError("star elasticity must be nonnegative.")
        for leaf in star.leaves:
            if not (0 <= leaf < n_nodes):
                raise ValueError("star leaf is out of bounds.")
            if leaf == star.center:
                raise ValueError("star center cannot also be a leaf.")


def _paper_e_matrix(n_nodes: int, edges: List[Edge]) -> Array:
    E = np.zeros((n_nodes, n_nodes), dtype=float)
    for edge in edges:
        i = int(edge.i)
        j = int(edge.j)
        lam = float(edge.lam)
        E[i, i] += lam
        E[j, j] += lam
        E[i, j] -= lam
        E[j, i] -= lam
    return E


def _paper_s_matrix(n_nodes: int, stars: List[Star]) -> Array:
    S = np.zeros((n_nodes, n_nodes), dtype=float)
    for star in stars:
        center = int(star.center)
        leaves = list(star.leaves)
        k = len(leaves)
        if k == 0:
            continue
        mu = float(star.mu)
        S[center, center] += mu
        incr = mu / float(k * k)
        for a in leaves:
            for b in leaves:
                S[a, b] += incr
        off = mu / float(k)
        for leaf in leaves:
            S[center, leaf] -= off
            S[leaf, center] -= off
    return S


def _solve_paper_linear_system_weighted(
    X: Array,
    vertices: Array,
    assignment: Array,
    edges: List[Edge],
    stars: List[Star],
    sample_weight: Array,
) -> Array:
    n_nodes = vertices.shape[0]
    _, n_features = X.shape
    w = np.asarray(sample_weight, dtype=float)
    total_weight = float(np.sum(w))
    if total_weight <= 0:
        raise ValueError("sum of sample_weight must be positive.")

    E = _paper_e_matrix(n_nodes, edges)
    S = _paper_s_matrix(n_nodes, stars)

    counts = np.bincount(assignment, weights=w, minlength=n_nodes).astype(float)
    sums = np.zeros((n_nodes, n_features), dtype=float)
    for j in range(n_nodes):
        mask = assignment == j
        if np.any(mask):
            sums[j] = np.sum(w[mask, None] * X[mask], axis=0)

    A = np.diag(counts / total_weight) + E + S
    A = A + 1e-10 * np.eye(n_nodes)
    B = sums / total_weight
    return np.linalg.solve(A, B)


def _penalty_matrix(n_nodes: int, edges: List[Edge], stars: List[Star]) -> Array:
    return _paper_e_matrix(n_nodes, edges) + _paper_s_matrix(n_nodes, stars)


def _elastic_energy_from_node_assignments_weighted(
    X: Array,
    vertices: Array,
    assignment: Array,
    edges: List[Edge],
    stars: List[Star],
    sample_weight: Array,
) -> float:
    w = np.asarray(sample_weight, dtype=float)
    total_weight = float(np.sum(w))
    if total_weight <= 0:
        raise ValueError("sum of sample_weight must be positive.")
    msd = _node_mean_squared_distance_weighted(X, vertices, assignment, w)
    P = _penalty_matrix(vertices.shape[0], edges, stars)
    penalty = 0.0
    for d in range(vertices.shape[1]):
        penalty += float(vertices[:, d].T @ P @ vertices[:, d])
    return float(msd + penalty)


def _node_mean_squared_distance_weighted(X: Array, vertices: Array, assignment: Array, sample_weight: Array) -> float:
    w = np.asarray(sample_weight, dtype=float)
    total_weight = float(np.sum(w))
    return float(np.sum(w * np.sum((X - vertices[assignment]) ** 2, axis=1)) / total_weight)


def _mean_squared_distance_to_graph_weighted(
    X: Array,
    vertices: Array,
    edge_index_pairs: Sequence[Tuple[int, int]],
    sample_weight: Array,
) -> float:
    w = np.asarray(sample_weight, dtype=float)
    total_weight = float(np.sum(w))
    if total_weight <= 0:
        raise ValueError("sum of sample_weight must be positive.")
    projected = _project_onto_graph_edges(X, vertices, edge_index_pairs)
    return float(np.sum(w * np.sum((X - projected) ** 2, axis=1)) / total_weight)

