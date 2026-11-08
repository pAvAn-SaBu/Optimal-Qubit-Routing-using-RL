"""Chip topologies: which physical positions are connected.

A topology is an undirected, connected graph. Positions are numbered 0..n-1.
All-pairs distances and shortest paths are precomputed with BFS.

Shortest-path tie-break rule (needed on rings and grids, where several shortest
paths exist): BFS visits neighbours in ascending index order and keeps the first
parent that reaches a node. The path is therefore always the same for the same
chip, so the simulator stays deterministic.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Topology:
    name: str
    n: int
    edges: tuple[tuple[int, int], ...]
    coords: tuple[tuple[float, float], ...]  # only used for drawing
    adj: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    dist: np.ndarray = field(init=False, repr=False)
    _parent: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        adj: list[set[int]] = [set() for _ in range(self.n)]
        for a, b in self.edges:
            if a == b or not (0 <= a < self.n and 0 <= b < self.n):
                raise ValueError(f"bad edge ({a}, {b}) for {self.n} positions")
            adj[a].add(b)
            adj[b].add(a)
        adj_t = tuple(tuple(sorted(s)) for s in adj)

        dist = np.full((self.n, self.n), -1, dtype=np.int64)
        parent = np.full((self.n, self.n), -1, dtype=np.int64)
        for src in range(self.n):
            dist[src, src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj_t[u]:
                    if dist[src, v] < 0:
                        dist[src, v] = dist[src, u] + 1
                        parent[src, v] = u
                        queue.append(v)
        if (dist < 0).any():
            raise ValueError(f"topology {self.name} is not connected")

        dist.setflags(write=False)
        parent.setflags(write=False)
        object.__setattr__(self, "adj", adj_t)
        object.__setattr__(self, "dist", dist)
        object.__setattr__(self, "_parent", parent)

    @property
    def diameter(self) -> int:
        return int(self.dist.max())

    @property
    def mean_degree(self) -> float:
        return 2 * len(self.edges) / self.n

    def are_adjacent(self, a: int, b: int) -> bool:
        return self.dist[a, b] == 1

    def shortest_path(self, a: int, b: int) -> list[int]:
        """Deterministic shortest path a -> b, endpoints included."""
        path = [b]
        while path[-1] != a:
            path.append(int(self._parent[a, path[-1]]))
        path.reverse()
        return path

    def undirected_edge_set(self) -> set[tuple[int, int]]:
        return {(min(a, b), max(a, b)) for a, b in self.edges}

    def coupling_list(self) -> list[list[int]]:
        """Both directions of every edge, the format Qiskit's CouplingMap expects."""
        out = []
        for a, b in self.edges:
            out.append([a, b])
            out.append([b, a])
        return out


def line(n: int) -> Topology:
    return Topology(f"line{n}", n, tuple((i, i + 1) for i in range(n - 1)),
                    tuple((float(i), 0.0) for i in range(n)))


def ring(n: int) -> Topology:
    if n < 3:
        raise ValueError("ring needs at least 3 positions")
    coords = tuple((math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n))
    return Topology(f"ring{n}", n, tuple((i, (i + 1) % n) for i in range(n)), coords)


def grid(rows: int, cols: int) -> Topology:
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append((i, i + 1))
            if r + 1 < rows:
                edges.append((i, i + cols))
    coords = tuple((float(c), float(-r)) for r in range(rows) for c in range(cols))
    return Topology(f"grid{rows}x{cols}", rows * cols, tuple(edges), coords)


def star(n: int) -> Topology:
    """Position 0 is the hub, connected to every other position."""
    if n < 2:
        raise ValueError("star needs at least 2 positions")
    coords = [(0.0, 0.0)] + [
        (math.cos(2 * math.pi * i / (n - 1)), math.sin(2 * math.pi * i / (n - 1))) for i in range(n - 1)
    ]
    return Topology(f"star{n}", n, tuple((0, i) for i in range(1, n)), tuple(coords))


def from_spec(spec: str) -> Topology:
    """Parse 'line:5', 'ring:6', 'grid:3x3', 'star:5'."""
    try:
        kind, size = spec.lower().split(":")
        if kind == "grid":
            rows, cols = (int(x) for x in size.split("x"))
            return grid(rows, cols)
        builders = {"line": line, "ring": ring, "star": star}
        return builders[kind](int(size))
    except (ValueError, KeyError) as exc:
        raise ValueError(f"bad topology spec {spec!r}; use e.g. line:5, ring:6, grid:3x3, star:5") from exc
