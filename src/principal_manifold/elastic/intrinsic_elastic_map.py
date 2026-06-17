from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from .._types import GraphSnapshot, PrincipalGraphResult, copy_graph_snapshot
from ..geometry import (
    _as_2d_float_array,
    _graph_total_edge_length,
    _initialize_intrinsic_coordinates_on_first_k_pcs,
    _mean_squared_distance_to_graph,
    _mean_squared_distance_to_surface,
    _project_onto_complex,
)
from .intrinsic_topology import IntrinsicTopologyAdapter
from .optimizer import FixedElasticGraphOptimizer, _validate_allowed_k_stars
from .primitive import PrimitiveElasticGraph

Array = np.ndarray


@dataclass
class IntrinsicElasticMapConfig:
    intrinsic_dim: int = 2
    grid_shape: Optional[Tuple[int, ...]] = None
    lam: float = 0.02
    mu: float = 0.05
    optimizer_max_iter: int = 100
    optimizer_tol: float = 1e-6
    softening: Tuple[float, ...] = (1e2, 1e1, 1.0)
    verbose: bool = False
    store_trace: bool = False
    allowed_k_stars: Optional[Tuple[int, ...]] = None
    topology_ops: Tuple[str, ...] = ("split", "prune")
    topology_epochs: int = 2
    topology_max_ops_per_epoch: int = 1
    min_nodes_per_dim: int = 2
    max_nodes_per_dim: int = 8


class IntrinsicElasticMapPrincipalManifold:
    """Fixed-k intrinsic elastic map MVP with split/prune topology updates."""

    def __init__(self, config: Optional[IntrinsicElasticMapConfig] = None) -> None:
        self.config = config or IntrinsicElasticMapConfig()
        _validate_intrinsic_elastic_map_config(self.config)

        self.vertices_: Optional[Array] = None
        self.intrinsic_vertices_: Optional[Array] = None
        self.history_: List[Dict[str, float]] = []
        self.trace_: List[GraphSnapshot] = []
        self.cells_by_dim_: Optional[Dict[int, Array]] = None
        self._faces: Array = np.empty((0, 3), dtype=int)
        self._graph: Optional[PrimitiveElasticGraph] = None
        self._X_fit_: Optional[Array] = None
        self._intrinsic_bounds_: Optional[List[Tuple[float, float]]] = None
        self._optimizer = FixedElasticGraphOptimizer(
            max_iter=self.config.optimizer_max_iter,
            tol=self.config.optimizer_tol,
        )
        self._topology_adapter = IntrinsicTopologyAdapter(
            ops=tuple(self.config.topology_ops),
            max_ops_per_epoch=int(self.config.topology_max_ops_per_epoch),
        )

    def fit(self, X: Array) -> "IntrinsicElasticMapPrincipalManifold":
        X = _as_2d_float_array(X)
        _require_all_finite("X", X)
        if X.shape[0] < 2:
            raise ValueError("At least two samples are required.")
        if self.config.intrinsic_dim > X.shape[1]:
            raise ValueError(
                f"intrinsic_dim must be <= n_features ({X.shape[1]})."
            )

        self._X_fit_ = X.copy()
        self.history_ = []
        self.trace_ = []

        intrinsic_samples = _initialize_intrinsic_coordinates_on_first_k_pcs(
            X,
            self.config.intrinsic_dim,
        )
        _require_all_finite("intrinsic_samples", intrinsic_samples)
        self._intrinsic_bounds_ = _intrinsic_bounds(intrinsic_samples)

        grid_shape = _resolved_grid_shape(self.config)
        graph, intrinsic_vertices, cells_by_dim = self._build_graph(
            X=X,
            intrinsic_samples=intrinsic_samples,
            grid_shape=grid_shape,
        )

        self._graph = graph
        self.vertices_ = np.asarray(graph.vertices, dtype=float).copy()
        self.intrinsic_vertices_ = np.asarray(intrinsic_vertices, dtype=float).copy()
        self.cells_by_dim_ = _copy_cells_by_dim(cells_by_dim)
        self._faces = np.asarray(cells_by_dim.get(2, np.empty((0, 3), dtype=int)), dtype=int)

        if self.config.store_trace:
            self._append_trace("init", 0, None, "init")

        self._optimize_current_graph(outer_iteration=0, operation="init")

        for epoch_index in range(self.config.topology_epochs):
            changed = False
            for op in self._topology_adapter.select_epoch_ops(epoch_index):
                updated_shape = self._updated_grid_shape(op, grid_shape)
                if updated_shape == grid_shape:
                    continue
                candidate_graph, candidate_intrinsic_vertices, candidate_cells = self._build_graph(
                    X=X,
                    intrinsic_samples=intrinsic_samples,
                    grid_shape=updated_shape,
                )
                self._graph = candidate_graph
                self.vertices_ = np.asarray(candidate_graph.vertices, dtype=float).copy()
                self.intrinsic_vertices_ = np.asarray(candidate_intrinsic_vertices, dtype=float).copy()
                self.cells_by_dim_ = _copy_cells_by_dim(candidate_cells)
                self._faces = np.asarray(candidate_cells.get(2, np.empty((0, 3), dtype=int)), dtype=int)
                grid_shape = updated_shape
                changed = True

                if self.config.store_trace:
                    self._append_trace("topology", epoch_index + 1, None, op)
                self._optimize_current_graph(outer_iteration=epoch_index + 1, operation=op)
                break
            if not changed:
                break

        return self

    def fit_result(self, X: Array) -> PrincipalGraphResult:
        self.fit(X)
        return self.result_

    @property
    def result_(self) -> PrincipalGraphResult:
        if self.vertices_ is None or self._X_fit_ is None or self._graph is None:
            raise AttributeError("Model has not been fitted yet.")
        return PrincipalGraphResult(
            vertices=self.vertices_.copy(),
            edges=np.asarray(self._graph.edges, dtype=int).copy(),
            projected_points=self.project(self._X_fit_),
            history=[dict(item) for item in self.history_],
            trace=[copy_graph_snapshot(item) for item in self.trace_],
            faces=None if self._faces.size == 0 else self._faces.copy(),
            cells_by_dim=_copy_cells_by_dim(self.cells_by_dim_),
        )

    def project(self, X: Array) -> Array:
        if self.vertices_ is None or self._graph is None:
            raise ValueError("Call fit before project.")
        X = _as_2d_float_array(X)
        projected = _project_onto_complex(
            X,
            self.vertices_,
            edge_index_pairs=self._graph.edges,
            faces=None if self._faces.size == 0 else self._faces,
            prefer_dim=2 if self._faces.size > 0 else 1,
        )
        _require_all_finite("projected_points", projected)
        return projected

    def transform(self, X: Array) -> Array:
        return self.project(X)

    def predict(self, X: Array) -> Array:
        return self.transform(X)

    def score(self, X: Array) -> float:
        if self.vertices_ is None or self._graph is None:
            raise ValueError("Call fit before score.")
        X = _as_2d_float_array(X)
        score = -_projected_msd(
            X,
            self.vertices_,
            self._graph.edges,
            None if self._faces.size == 0 else self._faces,
        )
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite score encountered; aborting intrinsic elastic map fit.")
        return float(score)

    def _build_graph(
        self,
        X: Array,
        intrinsic_samples: Array,
        grid_shape: Tuple[int, ...],
    ) -> Tuple[PrimitiveElasticGraph, Array, Dict[int, Array]]:
        if self._intrinsic_bounds_ is None:
            raise ValueError("Intrinsic bounds are not available.")
        intrinsic_vertices = _build_intrinsic_grid(self._intrinsic_bounds_, grid_shape)
        ambient_vertices = _fit_affine_embedding(intrinsic_samples, X, intrinsic_vertices)
        _require_all_finite("ambient_vertices", ambient_vertices)
        cells_by_dim = _build_cells_by_dim(grid_shape)
        edge_list = [
            (int(edge[0]), int(edge[1]))
            for edge in np.asarray(cells_by_dim[1], dtype=int)
        ]
        graph = PrimitiveElasticGraph(
            vertices=ambient_vertices,
            edges=edge_list,
            lam=float(self.config.lam),
            mu=float(self.config.mu),
        )
        return graph, intrinsic_vertices, cells_by_dim

    def _updated_grid_shape(self, op: str, grid_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        counts = list(grid_shape)
        spans = [hi - lo for lo, hi in self._intrinsic_bounds_ or []]
        if op == "split":
            scores = [
                (spans[idx] / max(counts[idx] - 1, 1), -idx)
                for idx in range(len(counts))
                if counts[idx] < self.config.max_nodes_per_dim
            ]
            if not scores:
                return grid_shape
            chosen = max(range(len(scores)), key=lambda idx: scores[idx][0])
            dim = [
                i for i in range(len(counts)) if counts[i] < self.config.max_nodes_per_dim
            ][chosen]
            counts[dim] += 1
            return tuple(counts)
        if op == "prune":
            candidates = [
                idx for idx in range(len(counts)) if counts[idx] > self.config.min_nodes_per_dim
            ]
            if not candidates:
                return grid_shape
            dim = min(candidates, key=lambda idx: (counts[idx], idx))
            counts[dim] -= 1
            return tuple(counts)
        raise ValueError(f"Unsupported topology operation: {op}")

    def _optimize_current_graph(self, outer_iteration: int, operation: str) -> None:
        if self._X_fit_ is None or self._graph is None or self.vertices_ is None:
            raise ValueError("No graph available for optimization.")
        current_vertices = np.asarray(self.vertices_, dtype=float).copy()
        for epoch_idx, multiplier in enumerate(self.config.softening, start=1):
            graph = PrimitiveElasticGraph(
                vertices=current_vertices,
                edges=list(self._graph.edges),
                lam=float(self.config.lam),
                mu=float(self.config.mu),
            )
            result_obj = self._optimizer.optimize(
                X=self._X_fit_,
                vertices=graph.vertices,
                edges=graph.edge_objects(multiplier=float(multiplier)),
                stars=graph.star_objects(multiplier=float(multiplier), allowed_k_stars=self.config.allowed_k_stars),
                sample_weight=None,
                return_history=True,
                allowed_k_stars=self.config.allowed_k_stars,
            )
            result, opt_history = cast(Tuple[Any, List[Dict[str, object]]], result_obj)
            _guard_optimizer_state(result.vertices, result.energy)
            current_vertices = np.asarray(result.vertices, dtype=float).copy()
            self.vertices_ = current_vertices.copy()
            self._graph = PrimitiveElasticGraph(
                vertices=current_vertices.copy(),
                edges=list(graph.edges),
                lam=float(graph.lam),
                mu=float(graph.mu),
            )

            for item in opt_history:
                item_dict = cast(Dict[str, Any], item)
                sweep_vertices = np.asarray(item_dict["vertices"], dtype=float)
                elastic_energy = float(item_dict["elastic_energy"])
                _guard_optimizer_state(sweep_vertices, elastic_energy)
                msd = _projected_msd(
                    self._X_fit_,
                    sweep_vertices,
                    self._graph.edges,
                    None if self._faces.size == 0 else self._faces,
                )
                record = {
                    "outer_iteration": float(outer_iteration),
                    "epoch": float(epoch_idx),
                    "sweep": float(item_dict["sweep"]),
                    "operation": str(operation),
                    "nodes": float(sweep_vertices.shape[0]),
                    "edges": float(len(self._graph.edges)),
                    "segments": float(len(self._graph.edges)),
                    "faces": float(self._faces.shape[0]),
                    "mean_squared_distance": float(msd),
                    "root_mean_squared_distance": float(np.sqrt(max(msd, 0.0))),
                    "node_mean_squared_distance": float(item_dict["node_mean_squared_distance"]),
                    "polyline_length": float(item_dict["polyline_length"]),
                    "elastic_energy": elastic_energy,
                    "multiplier": float(multiplier),
                    "converged": float(result.converged),
                }
                self.history_.append(record)
                if self.config.verbose:
                    print(record)
                if self.config.store_trace:
                    self.trace_.append(
                        GraphSnapshot(
                            phase="updated",
                            outer_iteration=int(outer_iteration),
                            sweep=int(item_dict["sweep"]),
                            vertices=sweep_vertices.copy(),
                            edges=np.asarray(self._graph.edges, dtype=int).copy(),
                            mean_squared_distance=float(msd),
                            root_mean_squared_distance=float(np.sqrt(max(msd, 0.0))),
                            lambda_p=0.0,
                            segments=int(len(self._graph.edges)),
                            polyline_length=float(item_dict["polyline_length"]),
                            elastic_energy=elastic_energy,
                            operation=str(operation),
                            construction_complexity=int(outer_iteration),
                            structural_complexity=float(sweep_vertices.shape[0]),
                            faces=None if self._faces.size == 0 else self._faces.copy(),
                            cells_by_dim=_copy_cells_by_dim(self.cells_by_dim_),
                        )
                    )

    def _append_trace(
        self,
        phase: str,
        outer_iteration: int,
        sweep: Optional[int],
        operation: Optional[str],
    ) -> None:
        if self._X_fit_ is None or self._graph is None or self.vertices_ is None:
            raise ValueError("No fitted data available for trace creation.")
        msd = _projected_msd(
            self._X_fit_,
            self.vertices_,
            self._graph.edges,
            None if self._faces.size == 0 else self._faces,
        )
        self.trace_.append(
            GraphSnapshot(
                phase=str(phase),
                outer_iteration=int(outer_iteration),
                sweep=None if sweep is None else int(sweep),
                vertices=self.vertices_.copy(),
                edges=np.asarray(self._graph.edges, dtype=int).copy(),
                mean_squared_distance=float(msd),
                root_mean_squared_distance=float(np.sqrt(max(msd, 0.0))),
                lambda_p=0.0,
                segments=int(len(self._graph.edges)),
                polyline_length=float(_graph_total_edge_length(self.vertices_, self._graph.edges)),
                elastic_energy=float(self.history_[-1]["elastic_energy"]) if self.history_ else 0.0,
                operation=None if operation is None else str(operation),
                construction_complexity=int(outer_iteration),
                structural_complexity=float(self.vertices_.shape[0]),
                faces=None if self._faces.size == 0 else self._faces.copy(),
                cells_by_dim=_copy_cells_by_dim(self.cells_by_dim_),
            )
        )


def _validate_intrinsic_elastic_map_config(config: IntrinsicElasticMapConfig) -> None:
    if config.intrinsic_dim < 1:
        raise ValueError("intrinsic_dim must be at least 1.")
    if config.lam < 0:
        raise ValueError("lam must be nonnegative.")
    if config.mu < 0:
        raise ValueError("mu must be nonnegative.")
    if config.optimizer_max_iter < 1:
        raise ValueError("optimizer_max_iter must be at least 1.")
    if config.optimizer_tol < 0:
        raise ValueError("optimizer_tol must be nonnegative.")
    if len(config.softening) == 0:
        raise ValueError("softening must contain at least one multiplier.")
    if any(mult <= 0 for mult in config.softening):
        raise ValueError("softening multipliers must be positive.")
    if config.topology_epochs < 0:
        raise ValueError("topology_epochs must be nonnegative.")
    if config.min_nodes_per_dim < 2:
        raise ValueError("min_nodes_per_dim must be at least 2.")
    if config.max_nodes_per_dim < config.min_nodes_per_dim:
        raise ValueError("max_nodes_per_dim must be >= min_nodes_per_dim.")
    if config.grid_shape is not None:
        if len(config.grid_shape) != config.intrinsic_dim:
            raise ValueError("grid_shape length must match intrinsic_dim.")
        if any(int(count) < 2 for count in config.grid_shape):
            raise ValueError("grid_shape entries must be at least 2.")
    _validate_allowed_k_stars(config.allowed_k_stars)


def _resolved_grid_shape(config: IntrinsicElasticMapConfig) -> Tuple[int, ...]:
    if config.grid_shape is not None:
        return tuple(int(count) for count in config.grid_shape)
    return tuple(3 for _ in range(config.intrinsic_dim))


def _intrinsic_bounds(U: Array) -> List[Tuple[float, float]]:
    return [
        (float(np.min(U[:, idx])), float(np.max(U[:, idx])))
        for idx in range(U.shape[1])
    ]


def _build_intrinsic_grid(bounds: Sequence[Tuple[float, float]], grid_shape: Tuple[int, ...]) -> Array:
    axes = [
        np.linspace(float(lo), float(hi), int(count))
        for (lo, hi), count in zip(bounds, grid_shape)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    stacked = np.stack([axis.ravel() for axis in mesh], axis=1)
    return np.asarray(stacked, dtype=float)


def _fit_affine_embedding(U: Array, X: Array, grid_points: Array) -> Array:
    design = np.column_stack([U, np.ones(U.shape[0], dtype=float)])
    coeffs, _, _, _ = np.linalg.lstsq(design, X, rcond=None)
    grid_design = np.column_stack([grid_points, np.ones(grid_points.shape[0], dtype=float)])
    return np.asarray(grid_design @ coeffs, dtype=float)


def _flat_index(index: Tuple[int, ...], grid_shape: Tuple[int, ...]) -> int:
    flat = 0
    stride = 1
    for axis in range(len(grid_shape) - 1, -1, -1):
        flat += int(index[axis]) * stride
        stride *= int(grid_shape[axis])
    return int(flat)


def _build_cells_by_dim(grid_shape: Tuple[int, ...]) -> Dict[int, Array]:
    k = len(grid_shape)
    edge_set = set()
    for index in np.ndindex(*grid_shape):
        for axis in range(k):
            if index[axis] + 1 >= grid_shape[axis]:
                continue
            neighbor = list(index)
            neighbor[axis] += 1
            edge_set.add(
                tuple(
                    sorted(
                        (
                            _flat_index(index, grid_shape),
                            _flat_index(tuple(neighbor), grid_shape),
                        )
                    )
                )
            )

    cells_by_dim: Dict[int, Array] = {
        1: np.asarray(sorted(edge_set), dtype=int),
    }

    if k == 2:
        faces: List[Tuple[int, int, int]] = []
        for i in range(grid_shape[0] - 1):
            for j in range(grid_shape[1] - 1):
                v00 = _flat_index((i, j), grid_shape)
                v10 = _flat_index((i + 1, j), grid_shape)
                v01 = _flat_index((i, j + 1), grid_shape)
                v11 = _flat_index((i + 1, j + 1), grid_shape)
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))
        cells_by_dim[2] = np.asarray(faces, dtype=int)
    return cells_by_dim


def _copy_cells_by_dim(cells_by_dim: Optional[Dict[int, Array]]) -> Optional[Dict[int, Array]]:
    if cells_by_dim is None:
        return None
    return {
        int(dim): np.asarray(cells, dtype=int).copy()
        for dim, cells in cells_by_dim.items()
    }


def _projected_msd(
    X: Array,
    vertices: Array,
    edges: Sequence[Tuple[int, int]],
    faces: Optional[Array],
) -> float:
    if faces is not None and np.asarray(faces).size > 0:
        msd = _mean_squared_distance_to_surface(X, vertices, np.asarray(faces, dtype=int))
    else:
        msd = _mean_squared_distance_to_graph(X, vertices, edges)
    if not np.isfinite(msd):
        raise FloatingPointError("Non-finite projection energy encountered; aborting intrinsic elastic map fit.")
    return float(msd)


def _require_all_finite(name: str, values: Array) -> None:
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise FloatingPointError(f"Non-finite {name} encountered; aborting intrinsic elastic map fit.")


def _guard_optimizer_state(vertices: Array, energy: float) -> None:
    _require_all_finite("optimizer vertices", vertices)
    if not np.isfinite(float(energy)):
        raise FloatingPointError("Non-finite elastic energy encountered; aborting intrinsic elastic map fit.")
