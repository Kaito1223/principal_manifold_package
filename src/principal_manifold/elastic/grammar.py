from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .primitive import PrimitiveElasticGraph

Array = np.ndarray

def _op_add_node_all(graph: PrimitiveElasticGraph) -> List[PrimitiveElasticGraph]:
    candidates: List[PrimitiveElasticGraph] = []
    adj = graph.adjacency()
    for center in range(graph.n_nodes):
        new_graph = graph.copy()
        new_index = new_graph.n_nodes
        center_pos = new_graph.vertices[center]
        if len(adj[center]) > 0:
            neighbor_mean = np.mean(new_graph.vertices[adj[center]], axis=0)
            new_pos = center_pos + (center_pos - neighbor_mean)
        else:
            new_pos = center_pos.copy()
        new_graph.vertices = np.vstack([new_graph.vertices, new_pos[None, :]])
        new_graph.edges.append((center, new_index))
        candidates.append(_canonicalize_graph(new_graph))
    return candidates


def _op_bisect_edge_all(graph: PrimitiveElasticGraph) -> List[PrimitiveElasticGraph]:
    candidates: List[PrimitiveElasticGraph] = []
    for edge_idx, (i, j) in enumerate(graph.edges):
        new_graph = graph.copy()
        new_index = new_graph.n_nodes
        midpoint = 0.5 * (new_graph.vertices[i] + new_graph.vertices[j])
        new_graph.vertices = np.vstack([new_graph.vertices, midpoint[None, :]])
        new_edges = [e for k, e in enumerate(new_graph.edges) if k != edge_idx]
        new_edges.append((i, new_index))
        new_edges.append((new_index, j))
        new_graph.edges = new_edges
        candidates.append(_canonicalize_graph(new_graph))
    return candidates


def _op_remove_leaf_all(graph: PrimitiveElasticGraph) -> List[PrimitiveElasticGraph]:
    if graph.n_nodes <= 2:
        return []
    candidates: List[PrimitiveElasticGraph] = []
    deg = graph.degree()
    leaf_nodes = [i for i in range(graph.n_nodes) if deg[i] == 1]
    for leaf in leaf_nodes:
        new_graph = _remove_node(graph, leaf)
        if new_graph is not None and _is_connected_graph(new_graph.n_nodes, new_graph.edges):
            candidates.append(_canonicalize_graph(new_graph))
    return candidates


def _op_remove_edge_all(graph: PrimitiveElasticGraph) -> List[PrimitiveElasticGraph]:
    if len(graph.edges) == 0:
        return []
    candidates: List[PrimitiveElasticGraph] = []
    for idx in range(len(graph.edges)):
        new_edges = [e for k, e in enumerate(graph.edges) if k != idx]
        if _is_connected_graph(graph.n_nodes, new_edges):
            new_graph = PrimitiveElasticGraph(
                vertices=graph.vertices.copy(),
                edges=new_edges,
                lam=graph.lam,
                mu=graph.mu,
            )
            candidates.append(_canonicalize_graph(new_graph))
    return candidates


def _remove_node(graph: PrimitiveElasticGraph, node: int) -> Optional[PrimitiveElasticGraph]:
    if graph.n_nodes <= 1:
        return None
    keep = [i for i in range(graph.n_nodes) if i != node]
    if len(keep) == 0:
        return None
    index_map = {old: new for new, old in enumerate(keep)}
    new_edges: List[Tuple[int, int]] = []
    for i, j in graph.edges:
        if i == node or j == node:
            continue
        new_edges.append((index_map[i], index_map[j]))
    return PrimitiveElasticGraph(
        vertices=graph.vertices[keep],
        edges=sorted(set(tuple(sorted(e)) for e in new_edges if e[0] != e[1])),
        lam=graph.lam,
        mu=graph.mu,
    )


def _is_connected_graph(n_nodes: int, edges: Sequence[Tuple[int, int]]) -> bool:
    if n_nodes == 0:
        return False
    if n_nodes == 1:
        return True
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    visited = np.zeros(n_nodes, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    return bool(np.all(visited))


def _canonicalize_graph(graph: PrimitiveElasticGraph) -> PrimitiveElasticGraph:
    edges = []
    for i, j in graph.edges:
        if i == j:
            continue
        a, b = sorted((int(i), int(j)))
        edges.append((a, b))
    edges = sorted(set(edges))
    return PrimitiveElasticGraph(
        vertices=np.asarray(graph.vertices, dtype=float).copy(),
        edges=edges,
        lam=float(graph.lam),
        mu=float(graph.mu),
    )


def _deduplicate_graphs(graphs: List[PrimitiveElasticGraph]) -> List[PrimitiveElasticGraph]:
    unique: List[PrimitiveElasticGraph] = []
    seen = set()
    for graph in graphs:
        rounded_vertices = tuple(np.round(np.asarray(graph.vertices, dtype=float).ravel(), 12))
        key = (graph.n_nodes, tuple(graph.edges), rounded_vertices)
        if key not in seen:
            seen.add(key)
            unique.append(graph)
    return unique