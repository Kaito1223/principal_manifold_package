from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .._types import GraphSnapshot, PrincipalGraphResult, copy_graph_snapshot
from ..geometry import (
    _as_2d_float_array,
    _graph_total_edge_length,
    _initialize_surface_grid_on_first_two_pcs,
    _mean_squared_distance_to_surface,
    _project_onto_complex,
    _project_onto_surface,
)
from .optimizer import Edge, Star, FixedElasticGraphOptimizer, _validate_allowed_k_stars
from .primitive import PrimitiveElasticGraph

Array = np.ndarray


@dataclass
class ElasticSurfaceConfig:
    lam: float = 0.02
    mu: float = 0.05
    n_u: int = 6
    n_v: int = 6
    max_iter: int = 100
    tol: float = 1e-6
    softening: Tuple[float, ...] = (1e3, 1e2, 1e1, 1.0)
    verbose: bool = False
    store_trace: bool = False
    allowed_k_stars: Optional[Tuple[int, ...]] = None


class ElasticSurfacePrincipalManifold:
    """Fixed-topology 2D elastic principal manifold embedded in R^d.

    Optimization reuses the existing elastic-graph solver: each sweep minimizes
    node-assignment energy plus edge/star elastic regularization. Reported
    `mean_squared_distance` and `score()` use projection onto the triangulated
    surface induced by the fixed grid.
    """

    def __init__(self, config: Optional[ElasticSurfaceConfig] = None) -> None:
        self.config = config or ElasticSurfaceConfig()
        _validate_elastic_surface_config(self.config)

        self.vertices_: Optional[Array] = None
        self.history_: List[Dict[str, float]] = []
        self.trace_: List[GraphSnapshot] = []
        self._X_fit_: Optional[Array] = None
        self._base_graph: Optional[PrimitiveElasticGraph] = None
        self._faces: Array = np.empty((0, 3), dtype=int)
        self._optimizer = FixedElasticGraphOptimizer(
            max_iter=self.config.max_iter,
            tol=self.config.tol,
        )

    def fit(self, X: Array) -> "ElasticSurfacePrincipalManifold":
        X = _as_2d_float_array(X)
        if X.shape[0] < 4:
            raise ValueError("At least four samples are required.")

        self._X_fit_ = X.copy()
        self.history_ = []
        self.trace_ = []

        vertices = _initialize_surface_grid_on_first_two_pcs(X, self.config.n_u, self.config.n_v)
        edges, faces = _build_surface_topology(self.config.n_u, self.config.n_v)
        self._faces = faces
        self._base_graph = PrimitiveElasticGraph(
            vertices=vertices,
            edges=[tuple(map(int, edge)) for edge in edges.tolist()],
            lam=float(self.config.lam),
            mu=float(self.config.mu),
        )
        self.vertices_ = np.asarray(vertices, dtype=float).copy()

        if self.config.store_trace:
            self._append_trace("init", 0, None, self.vertices_, 0.0)

        for epoch_idx, multiplier in enumerate(self.config.softening, start=1):
            graph = PrimitiveElasticGraph(
                vertices=np.asarray(self.vertices_, dtype=float).copy(),
                edges=list(self._base_graph.edges),
                lam=float(self.config.lam * multiplier),
                mu=float(self.config.mu * multiplier),
            )
            result, opt_history = self._optimizer.optimize(
                X=X,
                vertices=graph.vertices,
                edges=graph.edge_objects(),
                stars=graph.star_objects(allowed_k_stars=self.config.allowed_k_stars),
                sample_weight=None,
                return_history=True,
                allowed_k_stars=self.config.allowed_k_stars,
            )
            self.vertices_ = np.asarray(result.vertices, dtype=float).copy()

            for item in opt_history:
                msd = float(_mean_squared_distance_to_surface(X, np.asarray(item["vertices"], dtype=float), self._faces))
                record = {
                    "iteration": float(epoch_idx),
                    "epoch": float(epoch_idx),
                    "sweep": float(item["sweep"]),
                    "multiplier": float(multiplier),
                    "nodes": float(self.vertices_.shape[0]),
                    "segments": float(len(self._base_graph.edges)),
                    "faces": float(self._faces.shape[0]),
                    "mean_squared_distance": msd,
                    "root_mean_squared_distance": float(np.sqrt(max(msd, 0.0))),
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
                        elastic_energy=float(item["elastic_energy"]),
                    )
        return self

    def fit_result(self, X: Array) -> PrincipalGraphResult:
        self.fit(X)
        return self.result_

    @property
    def result_(self) -> PrincipalGraphResult:
        if self.vertices_ is None or self._X_fit_ is None or self._base_graph is None:
            raise AttributeError("Model has not been fitted yet.")
        return PrincipalGraphResult(
            vertices=self.vertices_.copy(),
            edges=np.asarray(self._base_graph.edges, dtype=int).copy(),
            projected_points=self.project(self._X_fit_),
            history=[dict(item) for item in self.history_],
            trace=[copy_graph_snapshot(s) for s in self.trace_],
            faces=self._faces.copy(),
        )

    def project(self, X: Array) -> Array:
        if self.vertices_ is None:
            raise ValueError("Call fit before project.")
        X = _as_2d_float_array(X)
        return _project_onto_complex(
            X,
            self.vertices_,
            faces=self._faces,
            prefer_dim=2,
        )

    def transform(self, X: Array) -> Array:
        return self.project(X)

    def predict(self, X: Array) -> Array:
        return self.transform(X)

    def score(self, X: Array) -> float:
        if self.vertices_ is None:
            raise ValueError("Call fit before score.")
        X = _as_2d_float_array(X)
        return -_mean_squared_distance_to_surface(X, self.vertices_, self._faces)

    def _append_trace(
        self,
        phase: str,
        outer_iteration: int,
        sweep: Optional[int],
        vertices: Array,
        elastic_energy: float,
    ) -> None:
        if self._X_fit_ is None or self._base_graph is None:
            raise ValueError("No fitted data available for trace creation.")
        verts = np.asarray(vertices, dtype=float)
        msd = _mean_squared_distance_to_surface(self._X_fit_, verts, self._faces)
        self.trace_.append(
            GraphSnapshot(
                phase=str(phase),
                outer_iteration=int(outer_iteration),
                sweep=None if sweep is None else int(sweep),
                vertices=verts.copy(),
                edges=np.asarray(self._base_graph.edges, dtype=int).copy(),
                mean_squared_distance=float(msd),
                root_mean_squared_distance=float(np.sqrt(max(msd, 0.0))),
                lambda_p=0.0,
                segments=int(len(self._base_graph.edges)),
                polyline_length=float(_graph_total_edge_length(verts, self._base_graph.edges)),
                elastic_energy=float(elastic_energy),
                operation=None,
                construction_complexity=0,
                structural_complexity=float(verts.shape[0]),
                faces=self._faces.copy(),
            )
        )


def _validate_elastic_surface_config(config: ElasticSurfaceConfig) -> None:
    if config.lam < 0:
        raise ValueError("lam must be nonnegative.")
    if config.mu < 0:
        raise ValueError("mu must be nonnegative.")
    if config.n_u < 2 or config.n_v < 2:
        raise ValueError("n_u and n_v must both be at least 2.")
    if config.max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if config.tol < 0:
        raise ValueError("tol must be nonnegative.")
    if len(config.softening) == 0:
        raise ValueError("softening must contain at least one multiplier.")
    if any(mult <= 0 for mult in config.softening):
        raise ValueError("softening multipliers must be positive.")
    _validate_allowed_k_stars(config.allowed_k_stars)


def _grid_index(i: int, j: int, n_v: int) -> int:
    return i * n_v + j


def _build_surface_topology(n_u: int, n_v: int) -> Tuple[Array, Array]:
    edge_set = set()
    faces: List[Tuple[int, int, int]] = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            v00 = _grid_index(i, j, n_v)
            v10 = _grid_index(i + 1, j, n_v)
            v01 = _grid_index(i, j + 1, n_v)
            v11 = _grid_index(i + 1, j + 1, n_v)
            triangles = ((v00, v10, v11), (v00, v11, v01))
            faces.extend(triangles)
            for a, b in (
                (v00, v10),
                (v10, v11),
                (v11, v01),
                (v01, v00),
                (v00, v11),
            ):
                edge_set.add(tuple(sorted((int(a), int(b)))))
    edges = np.asarray(sorted(edge_set), dtype=int)
    face_array = np.asarray(faces, dtype=int)
    return edges, face_array
