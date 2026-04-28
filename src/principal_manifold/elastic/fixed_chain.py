from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .._types import CurveSnapshot, PrincipalCurveResult, copy_curve_snapshot as _copy_snapshot
from ..geometry import (
    _as_2d_float_array,
    _initialize_nodes_on_first_pc,
    _mean_squared_distance_to_polyline,
    _polyline_length,
    _project_onto_polyline,
)
from .optimizer import Edge, Star, FixedElasticGraphOptimizer, _validate_allowed_k_stars

Array = np.ndarray

@dataclass
class ElasticGraphConfig:
    """
    Fixed-topology elastic graph with softening.

    Notes
    -----
    This model keeps a *chain* topology, so in the default configuration only
    primitive 2-stars can appear. The shared optimizer below is more general and
    supports arbitrary k-stars; the chain restriction comes from the chosen
    topology, not from the optimizer itself.
    """

    lam: float = 0.02
    mu: float = 0.05
    n_nodes: int = 15
    max_iter: int = 100
    tol: float = 1e-6
    softening: Tuple[float, ...] = (1e3, 1e2, 1e1, 1.0)
    verbose: bool = False
    store_trace: bool = False
    allowed_k_stars: Optional[Tuple[int, ...]] = None


class ElasticGraphPrincipalCurve:
    """Fixed-chain elastic graph principal curve.

    This class keeps the public API used by the other models while delegating
    the embedding updates to the shared fixed-topology optimizer.
    """

    def __init__(self, config: Optional[ElasticGraphConfig] = None) -> None:
        self.config = config or ElasticGraphConfig()
        _validate_elastic_graph_config(self.config)

        self.vertices_: Optional[Array] = None
        self.history_: List[Dict[str, float]] = []
        self.trace_: List[CurveSnapshot] = []
        self._X_fit_: Optional[Array] = None
        self._base_edges: List[Edge] = []
        self._base_stars: List[Star] = []
        self._optimizer = FixedElasticGraphOptimizer(
            max_iter=self.config.max_iter,
            tol=self.config.tol,
        )

    def fit(self, X: Array) -> "ElasticGraphPrincipalCurve":
        X = _as_2d_float_array(X)
        if X.shape[0] < 2:
            raise ValueError("At least two samples are required.")
        if self.config.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2 for a chain elastic graph.")

        self._X_fit_ = X.copy()
        self.history_ = []
        self.trace_ = []

        self.vertices_ = _initialize_nodes_on_first_pc(X, self.config.n_nodes)
        self._base_edges, self._base_stars = _build_chain_topology(
            n_nodes=self.config.n_nodes,
            lam=self.config.lam,
            mu=self.config.mu,
        )

        if self.config.store_trace:
            self._append_trace(
                phase="init",
                outer_iteration=0,
                sweep=None,
                vertices=self.vertices_,
            )

        for epoch_idx, multiplier in enumerate(self.config.softening, start=1):
            edges, stars = _scale_topology(self._base_edges, self._base_stars, float(multiplier))
            result, opt_history = self._optimizer.optimize(
                X=X,
                vertices=self.vertices_,
                edges=edges,
                stars=stars,
                sample_weight=None,
                return_history=True,
                allowed_k_stars=self.config.allowed_k_stars,
            )
            self.vertices_ = _sort_vertices_along_polyline(result.vertices)

            for item in opt_history:
                record = {
                    "iteration": float(epoch_idx),
                    "epoch": float(epoch_idx),
                    "sweep": float(item["sweep"]),
                    "multiplier": float(multiplier),
                    "nodes": float(self.vertices_.shape[0]),
                    "mean_squared_distance": float(item["mean_squared_distance"]),
                    "root_mean_squared_distance": float(np.sqrt(max(float(item["mean_squared_distance"]), 0.0))),
                    "node_mean_squared_distance": float(item["node_mean_squared_distance"]),
                    "polyline_length": float(item["polyline_length"]),
                    "elastic_energy": float(item["elastic_energy"]),
                    "relative_improvement": 0.0,
                    "converged": float(result.converged),
                }
                self.history_.append(record)
                if self.config.verbose:
                    print(record)

                if self.config.store_trace:
                    self._append_trace(
                        phase="updated",
                        outer_iteration=epoch_idx,
                        sweep=int(item["sweep"]),
                        vertices=np.asarray(item["vertices"], dtype=float),
                        mean_squared_distance=float(item["mean_squared_distance"]),
                    )

        return self

    def fit_result(self, X: Array) -> PrincipalCurveResult:
        self.fit(X)
        return self.result_

    @property
    def result_(self) -> PrincipalCurveResult:
        if self.vertices_ is None or self._X_fit_ is None:
            raise AttributeError("Model has not been fitted yet.")
        projected, arc_coords, kinds, indices = self.project(self._X_fit_)
        return PrincipalCurveResult(
            vertices=self.vertices_.copy(),
            projected_points=projected,
            arc_length_coordinates=arc_coords,
            nearest_object_kind=kinds,
            nearest_object_index=indices,
            history=[dict(item) for item in self.history_],
            trace=[_copy_snapshot(s) for s in self.trace_],
        )

    def project(self, X: Array) -> Tuple[Array, Array, Array, Array]:
        if self.vertices_ is None:
            raise ValueError("Call fit before project.")
        X = _as_2d_float_array(X)
        return _project_onto_polyline(X, self.vertices_)

    def transform(self, X: Array) -> Array:
        _, arc, _, _ = self.project(X)
        return arc

    def predict(self, X: Array) -> Array:
        return self.transform(X)

    def score(self, X: Array) -> float:
        if self.vertices_ is None:
            raise ValueError("Call fit before score.")
        X = _as_2d_float_array(X)
        return -_mean_squared_distance_to_polyline(X, self.vertices_)

    def _append_trace(
        self,
        phase: str,
        outer_iteration: int,
        sweep: Optional[int],
        vertices: Array,
        mean_squared_distance: Optional[float] = None,
    ) -> None:
        if self._X_fit_ is None:
            raise ValueError("No fitted data available for trace creation.")
        verts = np.asarray(vertices, dtype=float)
        msd = (
            float(mean_squared_distance)
            if mean_squared_distance is not None
            else _mean_squared_distance_to_polyline(self._X_fit_, verts)
        )
        self.trace_.append(
            CurveSnapshot(
                phase=str(phase),
                outer_iteration=int(outer_iteration),
                sweep=None if sweep is None else int(sweep),
                vertices=verts.copy(),
                mean_squared_distance=float(msd),
                root_mean_squared_distance=float(np.sqrt(max(msd, 0.0))),
                lambda_p=0.0,
                segments=int(max(verts.shape[0] - 1, 0)),
                polyline_length=float(_polyline_length(verts)),
            )
        )


# Aliases kept for compatibility.
PrincipalGraphConfig = ElasticGraphConfig
ElasticPrincipalGraphConfig = ElasticGraphConfig
PrincipalGraphCurve = ElasticGraphPrincipalCurve
ElasticPrincipalGraph = ElasticGraphPrincipalCurve
Model = ElasticGraphPrincipalCurve


def _validate_elastic_graph_config(config: ElasticGraphConfig) -> None:
    if config.lam < 0:
        raise ValueError("lam must be nonnegative.")
    if config.mu < 0:
        raise ValueError("mu must be nonnegative.")
    if config.n_nodes < 2:
        raise ValueError("n_nodes must be at least 2.")
    if config.max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if config.tol < 0:
        raise ValueError("tol must be nonnegative.")
    if len(config.softening) == 0:
        raise ValueError("softening must contain at least one multiplier.")
    if any(mult <= 0 for mult in config.softening):
        raise ValueError("softening multipliers must be positive.")
    _validate_allowed_k_stars(config.allowed_k_stars)


def _build_chain_topology(n_nodes: int, lam: float, mu: float) -> Tuple[List[Edge], List[Star]]:
    edges = [Edge(i=i, j=i + 1, lam=float(lam)) for i in range(n_nodes - 1)]
    stars = [Star(center=i, leaves=(i - 1, i + 1), mu=float(mu)) for i in range(1, n_nodes - 1)]
    return edges, stars


def _scale_topology(base_edges: List[Edge], base_stars: List[Star], multiplier: float) -> Tuple[List[Edge], List[Star]]:
    edges = [Edge(i=e.i, j=e.j, lam=float(e.lam * multiplier)) for e in base_edges]
    stars = [Star(center=s.center, leaves=s.leaves, mu=float(s.mu * multiplier)) for s in base_stars]
    return edges, stars


def _sort_vertices_along_polyline(vertices: Array) -> Array:
    mean = vertices.mean(axis=0)
    centered = vertices - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    order = np.argsort(centered @ axis)
    return vertices[order]
