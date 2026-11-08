"""Routing loop shared by every method, plus the non-learned baselines.

A policy is any callable `policy(state) -> (gate_index, Mode)` choosing among
the current waiting gates. `route()` runs it to completion and verifies the
result, so every method (greedy, tabular Q, DQN) is checked the same way.
SABRE is run through Qiskit with the same chip and the same trivial initial
layout, then verified by the same independent checker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from .circuit import Circuit
from .simulator import Event, Mode, RoutingState, cx_depth, events_from_qiskit, verify_routing
from .topology import Topology

Policy = Callable[[RoutingState], tuple[int, Mode]]


@dataclass
class Decision:
    gate: int
    mode: Mode
    swaps: int
    executed: list[int]
    event_start: int  # index into events where this decision's SWAPs begin


@dataclass
class RoutingResult:
    method: str
    swaps: int
    depth: int  # CX depth with SWAP = 3 CX
    events: list[Event]
    decisions: list[Decision] = field(default_factory=list)
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def route(topo: Topology, circuit: Circuit, policy: Policy, method: str = "policy",
          max_steps: int | None = None) -> RoutingResult:
    t0 = time.perf_counter()
    state = RoutingState(topo, circuit)
    decisions: list[Decision] = []
    limit = max_steps or len(circuit) + 1  # every action runs >= 1 gate
    while not state.is_done:
        if len(decisions) >= limit:
            raise RuntimeError("routing did not terminate; this should be impossible")
        g, mode = policy(state)
        start = len(state.events)
        res = state.apply(g, mode)
        decisions.append(Decision(g, Mode(mode), res.swaps, res.executed, start))
    errors = verify_routing(topo, circuit, state.events)
    return RoutingResult(method, state.n_swaps, cx_depth(state.events, topo.n), state.events, decisions,
                         time.perf_counter() - t0, errors)


def greedy_policy(state: RoutingState) -> tuple[int, Mode]:
    """Serve the closest waiting gate (fewest SWAPs now); ties -> earliest gate, MEET."""
    g = min(state.front(), key=lambda x: (state.phys_distance(x), x))
    return g, Mode.MEET


def make_random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)

    def policy(state: RoutingState) -> tuple[int, Mode]:
        front = state.front()
        return front[rng.integers(len(front))], Mode(rng.integers(3))

    return policy


def route_sabre(topo: Topology, circuit: Circuit, seed: int = 0) -> RoutingResult:
    """Qiskit's SABRE router, same chip, same trivial initial layout, no other optimisation."""
    t0 = time.perf_counter()
    qc = circuit.to_qiskit(topo.n)
    routed = transpile(qc, coupling_map=CouplingMap(topo.coupling_list()), layout_method="trivial",
                       routing_method="sabre", optimization_level=0, seed_transpiler=seed)
    events = events_from_qiskit(routed)
    errors = verify_routing(topo, circuit, events)
    swaps = sum(e.kind == "swap" for e in events)
    return RoutingResult("SABRE", swaps, cx_depth(events, topo.n), events, [], time.perf_counter() - t0, errors)
