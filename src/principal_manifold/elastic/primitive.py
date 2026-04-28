from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .optimizer import Edge, Star

Array = np.ndarray

@dataclass
class PrimitiveElasticGraph:
    vertices: Array
    edges: List[Tuple[int, int]]
    lam: float
    mu: float

    def copy(self) -> "PrimitiveElasticGraph":
        return PrimitiveElasticGraph(
            vertices=np.asarray(self.vertices, dtype=float).copy(),
            edges=[tuple(map(int, e)) for e in self.edges],
            lam=float(self.lam),
            mu=float(self.mu),
        )

    @property
    def n_nodes(self) -> int:
        return int(self.vertices.shape[0])

    def adjacency(self) -> List[List[int]]:
        adj: List[List[int]] = [[] for _ in range(self.n_nodes)]
        for i, j in self.edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def degree(self) -> np.ndarray:
        deg = np.zeros(self.n_nodes, dtype=int)
        for i, j in self.edges:
            deg[i] += 1
            deg[j] += 1
        return deg

    def edge_objects(self, multiplier: float = 1.0) -> List[Edge]:
        return [Edge(i=int(i), j=int(j), lam=float(self.lam * multiplier)) for i, j in self.edges]

    def star_objects(
        self,
        multiplier: float = 1.0,
        allowed_k_stars: Optional[Tuple[int, ...]] = None,
    ) -> List[Star]:
        allowed = None if allowed_k_stars is None else {int(k) for k in allowed_k_stars}
        stars: List[Star] = []
        adj = self.adjacency()
        for center, leaves in enumerate(adj):
            k = len(leaves)
            if k < 2:
                continue
            if allowed is not None and k not in allowed:
                continue
            stars.append(
                Star(
                    center=int(center),
                    leaves=tuple(int(v) for v in leaves),
                    mu=float(self.mu * multiplier),
                )
            )
        return stars
