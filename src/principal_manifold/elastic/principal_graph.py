from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._types import GraphSnapshot, PrincipalGraphResult, copy_graph_snapshot
from ..geometry import (
    _as_2d_float_array,
    _assign_points_to_nodes,
    _graph_total_edge_length,
    _initialize_nodes_on_first_pc,
    _mean_squared_distance_to_graph,
    _project_onto_graph_edges,
)
from .grammar import (
    _canonicalize_graph,
    _deduplicate_graphs,
    _op_add_node_all,
    _op_bisect_edge_all,
    _op_remove_edge_all,
    _op_remove_leaf_all,
)
from .optimizer import (
    Edge,
    Star,
    FixedElasticGraphOptimizer,
    _penalty_matrix,
    _validate_allowed_k_stars,
)
from .primitive import PrimitiveElasticGraph

Array = np.ndarray

@dataclass
class ElasticPrincipalGraphConfig:
    """Principal elastic graph / principal tree configuration.

    This dataclass keeps the old API fields (`init_nodes`, `max_nodes`,
    `grammar`, `passes`) while also supporting a more paper-like configuration
    via (`grammar_sequence`, `sc_max`, `cc_max`).

    Parameters
    ----------
    lam, mu
        Base edge and star elasticities.
    init_nodes, max_nodes, grammar, passes
        Backward-compatible controls from the earlier implementation.
    grammar_sequence, sc_max, cc_max
        Paper-style controls. When omitted, sensible values are derived from the
        backward-compatible fields.
    sc_measure
        Structural complexity measure. Supported values:
        - "n_vertices"
        - "branch_limited"
    bmax
        Used for `branch_limited`; counts branch nodes with degree >= 3.
    allowed_k_stars
        Optional tuple of allowed star sizes. Examples:
        - None -> allow any k >= 2 induced by the topology
        - (2,) -> only 2-stars
        - (2, 3, 4) -> only 2-, 3-, and 4-stars
    """

    lam: float = 0.02
    mu: float = 0.05

    # Backward-compatible fields.
    init_nodes: int = 2
    max_nodes: int = 15
    grammar: Tuple[str, ...] = ("grow", "grow", "shrink")
    passes: int = 10

    # Paper-style fields.
    grammar_sequence: Optional[Tuple[Tuple[str, ...], ...]] = None
    sc_max: Optional[float] = None
    cc_max: Optional[int] = None
    sc_measure: str = "n_vertices"
    bmax: Optional[int] = None

    optimizer_max_iter: int = 100
    optimizer_tol: float = 1e-6
    softening: Tuple[float, ...] = (1e3, 1e2, 1e1, 1.0)
    verbose: bool = False
    store_trace: bool = False
    allowed_k_stars: Optional[Tuple[int, ...]] = None


class ElasticPrincipalGraph:
    """Grammar-based principal elastic graph / principal tree.

    Public API:
      - fit
      - fit_result
      - result_
      - project
      - transform
      - predict
      - score
    """

    def __init__(self, config: Optional[ElasticPrincipalGraphConfig] = None) -> None:
        self.config = config or ElasticPrincipalGraphConfig()
        _validate_config(self.config)

        self.graph_: Optional[PrimitiveElasticGraph] = None
        self.history_: List[Dict[str, float]] = []
        self.trace_: List[GraphSnapshot] = []
        self._X_fit_: Optional[Array] = None
        self._optimizer = FixedElasticGraphOptimizer(
            max_iter=self.config.optimizer_max_iter,
            tol=self.config.optimizer_tol,
        )
        self._construction_complexity_: int = 0
        self._grammar_sequence_: Tuple[Tuple[str, ...], ...] = _resolved_grammar_sequence(self.config)
        self._sc_max_: float = _resolved_sc_max(self.config)
        self._cc_max_: int = _resolved_cc_max(self.config, self._grammar_sequence_)

    def fit(self, X: Array) -> "ElasticPrincipalGraph":
        X = _as_2d_float_array(X)
        if X.shape[0] < 2:
            raise ValueError("At least two samples are required.")

        self._X_fit_ = X.copy()
        self.history_ = []
        self.trace_ = []
        self._construction_complexity_ = 0

        initial_vertices = _initialize_nodes_on_first_pc(X, max(self.config.init_nodes, 2))
        initial_edges = [(i, i + 1) for i in range(initial_vertices.shape[0] - 1)]
        self.graph_ = PrimitiveElasticGraph(
            vertices=initial_vertices,
            edges=initial_edges,
            lam=float(self.config.lam),
            mu=float(self.config.mu),
        )

        if self.config.store_trace:
            self._append_trace("init", 0, None, "init", self.graph_)

        init_opt_graph, init_history, init_trace, _ = self._optimize_graph(
            X=X,
            graph=self.graph_,
            outer_iteration=0,
            operation="init",
            construction_complexity=self._construction_complexity_,
        )
        self.graph_ = init_opt_graph
        self.history_.extend(init_history)
        self.trace_.extend(init_trace)

        while self._construction_complexity_ < self._cc_max_:
            any_update = False
            outer_iteration = self._construction_complexity_ + 1

            for grammar in self._grammar_sequence_:
                candidates = self._apply_grammar_all_ways(self.graph_, grammar)
                permissible = [g for g in candidates if self._structural_complexity(g) <= self._sc_max_]
                if not permissible:
                    continue

                best_graph: Optional[PrimitiveElasticGraph] = None
                best_history: List[Dict[str, float]] = []
                best_trace: List[GraphSnapshot] = []
                best_objective = np.inf
                operation_label = ",".join(grammar)

                for candidate in permissible:
                    optimized_graph, candidate_history, candidate_trace, candidate_objective = self._optimize_graph(
                        X=X,
                        graph=candidate,
                        outer_iteration=outer_iteration,
                        operation=operation_label,
                        construction_complexity=self._construction_complexity_ + 1,
                    )
                    if candidate_objective < best_objective:
                        best_objective = candidate_objective
                        best_graph = optimized_graph
                        best_history = candidate_history
                        best_trace = candidate_trace

                if best_graph is None:
                    continue

                self.graph_ = best_graph
                self._construction_complexity_ += 1
                self.history_.extend(best_history)
                self.trace_.extend(best_trace)

                if self.config.store_trace:
                    self._append_trace(
                        phase="accepted",
                        outer_iteration=outer_iteration,
                        sweep=None,
                        operation=operation_label,
                        graph=self.graph_,
                    )

                any_update = True
                if self._construction_complexity_ >= self._cc_max_:
                    break

            if not any_update:
                break

        return self

    def fit_result(self, X: Array) -> PrincipalGraphResult:
        self.fit(X)
        return self.result_

    @property
    def result_(self) -> PrincipalGraphResult:
        if self.graph_ is None or self._X_fit_ is None:
            raise AttributeError("Model has not been fitted yet.")
        return PrincipalGraphResult(
            vertices=self.graph_.vertices.copy(),
            edges=np.asarray(self.graph_.edges, dtype=int).copy(),
            projected_points=self.project(self._X_fit_),
            history=[dict(h) for h in self.history_],
            trace=[_copy_snapshot(t) for t in self.trace_],
        )

    def project(self, X: Array) -> Array:
        if self.graph_ is None:
            raise ValueError("Call fit before project.")
        X = _as_2d_float_array(X)
        return _project_onto_graph_edges(X, self.graph_.vertices, self.graph_.edges)

    def transform(self, X: Array) -> Array:
        return self.project(X)

    def predict(self, X: Array) -> Array:
        return self.transform(X)

    def score(self, X: Array) -> float:
        if self.graph_ is None:
            raise ValueError("Call fit before score.")
        X = _as_2d_float_array(X)
        return -_mean_squared_distance_to_graph(X, self.graph_.vertices, self.graph_.edges)

    def _apply_grammar_all_ways(
        self,
        graph: PrimitiveElasticGraph,
        grammar: Sequence[str],
    ) -> List[PrimitiveElasticGraph]:
        candidates: List[PrimitiveElasticGraph] = []
        for op in grammar:
            if op == "add_node":
                candidates.extend(_op_add_node_all(graph))
            elif op == "bisect_edge":
                candidates.extend(_op_bisect_edge_all(graph))
            elif op == "remove_leaf":
                candidates.extend(_op_remove_leaf_all(graph))
            elif op == "remove_edge":
                candidates.extend(_op_remove_edge_all(graph))
            elif op == "grow":
                candidates.extend(_op_add_node_all(graph))
                candidates.extend(_op_bisect_edge_all(graph))
            elif op == "shrink":
                candidates.extend(_op_remove_leaf_all(graph))
                candidates.extend(_op_remove_edge_all(graph))
            else:
                raise ValueError(f"Unsupported grammar operation: {op}")
        candidates = [_canonicalize_graph(g) for g in candidates]
        return _deduplicate_graphs(candidates)

    def _structural_complexity(self, graph: PrimitiveElasticGraph) -> float:
        deg = graph.degree()
        n_vertices = graph.n_nodes
        n_branch_nodes = int(np.sum(deg >= 3))

        if self.config.sc_measure == "n_vertices":
            return float(n_vertices)
        if self.config.sc_measure == "branch_limited":
            if self.config.bmax is None:
                raise ValueError("bmax must be set for branch_limited complexity.")
            return float(n_vertices) if n_branch_nodes <= self.config.bmax else np.inf
        raise ValueError(f"Unsupported structural complexity: {self.config.sc_measure}")

    def _optimize_graph(
        self,
        X: Array,
        graph: PrimitiveElasticGraph,
        outer_iteration: int,
        operation: Optional[str],
        construction_complexity: int,
    ) -> Tuple[PrimitiveElasticGraph, List[Dict[str, float]], List[GraphSnapshot], float]:
        current_vertices = np.asarray(graph.vertices, dtype=float).copy()
        local_history: List[Dict[str, float]] = []
        local_trace: List[GraphSnapshot] = []

        for epoch_idx, multiplier in enumerate(self.config.softening, start=1):
            result, opt_history = self._optimizer.optimize(
                X=X,
                vertices=current_vertices,
                edges=graph.edge_objects(multiplier=float(multiplier)),
                stars=graph.star_objects(multiplier=float(multiplier), allowed_k_stars=self.config.allowed_k_stars),
                sample_weight=None,
                return_history=True,
                allowed_k_stars=self.config.allowed_k_stars,
            )
            current_vertices = np.asarray(result.vertices, dtype=float).copy()

            for item in opt_history:
                msd = float(item["mean_squared_distance"])
                record = {
                    "outer_iteration": float(outer_iteration),
                    "epoch": float(epoch_idx),
                    "sweep": float(item["sweep"]),
                    "operation": "" if operation is None else str(operation),
                    "construction_complexity": float(construction_complexity),
                    "structural_complexity": float(self._structural_complexity(graph)),
                    "nodes": float(graph.n_nodes),
                    "edges": float(len(graph.edges)),
                    "segments": float(len(graph.edges)),
                    "mean_squared_distance": msd,
                    "root_mean_squared_distance": float(np.sqrt(max(msd, 0.0))),
                    "node_mean_squared_distance": float(item["node_mean_squared_distance"]),
                    "polyline_length": float(item["polyline_length"]),
                    "elastic_energy": float(item["elastic_energy"]),
                    "multiplier": float(multiplier),
                    "converged": float(result.converged),
                }
                local_history.append(record)
                if self.config.verbose:
                    print(record)

                if self.config.store_trace:
                    local_trace.append(
                        GraphSnapshot(
                            phase="updated",
                            outer_iteration=int(outer_iteration),
                            sweep=int(item["sweep"]),
                            vertices=np.asarray(item["vertices"], dtype=float).copy(),
                            edges=np.asarray(graph.edges, dtype=int).copy(),
                            mean_squared_distance=msd,
                            root_mean_squared_distance=float(np.sqrt(max(msd, 0.0))),
                            lambda_p=0.0,
                            segments=int(len(graph.edges)),
                            polyline_length=float(item["polyline_length"]),
                            elastic_energy=float(item["elastic_energy"]),
                            operation=None if operation is None else str(operation),
                            construction_complexity=int(construction_complexity),
                            structural_complexity=float(self._structural_complexity(graph)),
                        )
                    )

        optimized_graph = PrimitiveElasticGraph(
            vertices=current_vertices,
            edges=[tuple(map(int, e)) for e in graph.edges],
            lam=float(graph.lam),
            mu=float(graph.mu),
        )
        objective = self._objective(X, optimized_graph)
        return optimized_graph, local_history, local_trace, objective

    def _objective(self, X: Array, graph: PrimitiveElasticGraph) -> float:
        assignment = _assign_points_to_nodes(X, graph.vertices)
        return _elastic_energy_from_node_assignments_unweighted(
            X=X,
            vertices=graph.vertices,
            assignment=assignment,
            edges=graph.edge_objects(1.0),
            stars=graph.star_objects(1.0, allowed_k_stars=self.config.allowed_k_stars),
        )

    def _append_trace(
        self,
        phase: str,
        outer_iteration: int,
        sweep: Optional[int],
        operation: Optional[str],
        graph: PrimitiveElasticGraph,
    ) -> None:
        if self._X_fit_ is None:
            raise ValueError("No fitted data available for trace creation.")
        projected_msd = _mean_squared_distance_to_graph(self._X_fit_, graph.vertices, graph.edges)
        node_assignment = _assign_points_to_nodes(self._X_fit_, graph.vertices)
        elastic_energy = _elastic_energy_from_node_assignments_unweighted(
            X=self._X_fit_,
            vertices=graph.vertices,
            assignment=node_assignment,
            edges=graph.edge_objects(1.0),
            stars=graph.star_objects(1.0, allowed_k_stars=self.config.allowed_k_stars),
        )
        self.trace_.append(
            GraphSnapshot(
                phase=str(phase),
                outer_iteration=int(outer_iteration),
                sweep=None if sweep is None else int(sweep),
                vertices=graph.vertices.copy(),
                edges=np.asarray(graph.edges, dtype=int).copy(),
                mean_squared_distance=float(projected_msd),
                root_mean_squared_distance=float(np.sqrt(max(projected_msd, 0.0))),
                lambda_p=0.0,
                segments=int(len(graph.edges)),
                polyline_length=float(_graph_total_edge_length(graph.vertices, graph.edges)),
                elastic_energy=float(elastic_energy),
                operation=None if operation is None else str(operation),
                construction_complexity=int(self._construction_complexity_),
                structural_complexity=float(self._structural_complexity(graph)),
            )
        )


PrincipalElasticGraph = ElasticPrincipalGraph
ElasticGraphFramework = ElasticPrincipalGraph
Model = ElasticPrincipalGraph



def _copy_snapshot(snapshot: GraphSnapshot) -> GraphSnapshot:
    return copy_graph_snapshot(snapshot)

def _validate_config(config: ElasticPrincipalGraphConfig) -> None:
    if config.lam < 0:
        raise ValueError("lam must be nonnegative.")
    if config.mu < 0:
        raise ValueError("mu must be nonnegative.")
    if config.init_nodes < 2:
        raise ValueError("init_nodes must be at least 2.")
    if config.max_nodes < config.init_nodes:
        raise ValueError("max_nodes must be >= init_nodes.")
    if config.passes < 1:
        raise ValueError("passes must be at least 1.")
    if config.optimizer_max_iter < 1:
        raise ValueError("optimizer_max_iter must be at least 1.")
    if config.optimizer_tol < 0:
        raise ValueError("optimizer_tol must be nonnegative.")
    if len(config.softening) == 0:
        raise ValueError("softening must contain at least one multiplier.")
    if any(mult <= 0 for mult in config.softening):
        raise ValueError("softening multipliers must be positive.")
    if config.sc_measure not in {"n_vertices", "branch_limited"}:
        raise ValueError("sc_measure must be 'n_vertices' or 'branch_limited'.")
    if config.sc_max is not None and config.sc_max < 0:
        raise ValueError("sc_max must be nonnegative when provided.")
    if config.cc_max is not None and config.cc_max < 0:
        raise ValueError("cc_max must be nonnegative when provided.")
    _validate_allowed_k_stars(config.allowed_k_stars)
    if config.grammar_sequence is not None:
        if len(config.grammar_sequence) == 0:
            raise ValueError("grammar_sequence must be nonempty when provided.")
        for grammar in config.grammar_sequence:
            if len(grammar) == 0:
                raise ValueError("each grammar in grammar_sequence must be nonempty.")


def _resolved_grammar_sequence(config: ElasticPrincipalGraphConfig) -> Tuple[Tuple[str, ...], ...]:
    if config.grammar_sequence is not None:
        return tuple(tuple(str(op) for op in grammar) for grammar in config.grammar_sequence)
    return tuple((_expand_macro_grammar(op)) for op in config.grammar)


def _expand_macro_grammar(op: str) -> Tuple[str, ...]:
    op = str(op)
    if op == "grow":
        return ("add_node", "bisect_edge")
    if op == "shrink":
        return ("remove_leaf", "remove_edge")
    if op in {"add_node", "bisect_edge", "remove_leaf", "remove_edge"}:
        return (op,)
    raise ValueError(f"Unsupported grammar operation: {op}")


def _resolved_sc_max(config: ElasticPrincipalGraphConfig) -> float:
    if config.sc_max is not None:
        return float(config.sc_max)
    return float(config.max_nodes)


def _resolved_cc_max(config: ElasticPrincipalGraphConfig, grammar_sequence: Tuple[Tuple[str, ...], ...]) -> int:
    if config.cc_max is not None:
        return int(config.cc_max)
    return int(max(1, config.passes) * max(1, len(grammar_sequence)))


def _elastic_energy_from_node_assignments_unweighted(
    X: Array,
    vertices: Array,
    assignment: Array,
    edges: List[Edge],
    stars: List[Star],
) -> float:
    msd = float(np.mean(np.sum((X - vertices[assignment]) ** 2, axis=1)))
    P = _penalty_matrix(vertices.shape[0], edges, stars)
    penalty = 0.0
    for d in range(vertices.shape[1]):
        penalty += float(vertices[:, d].T @ P @ vertices[:, d])
    return float(msd + penalty)

