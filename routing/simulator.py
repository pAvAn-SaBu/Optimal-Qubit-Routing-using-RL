"""Deterministic routing simulator: the fixed rules of the environment.

Terminology (see README):
  * layout       which logical qubit sits on which physical position
  * waiting gate a two-qubit gate that has not run and has no earlier
                 unexecuted gate on either of its qubits ("front layer")

A macro-action is (waiting gate, mode). The simulator walks the deterministic
shortest path between the gate's two qubits, inserts the SWAPs for the chosen
mode, runs the gate, then runs every waiting gate that has become adjacent for
free (repeating until none is left).

Modes, for a gate whose qubits are d apart (d >= 2, so d-1 SWAPs are needed):
  A_TO_B  qubit A walks d-1 steps towards B
  B_TO_A  qubit B walks d-1 steps towards A
  MEET    A walks ceil((d-1)/2) steps, B walks the rest
For d == 2 two of the three modes give the same result. That is harmless; the
agent just sees two equal options.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, NamedTuple

from qiskit import QuantumCircuit

from .circuit import Circuit
from .topology import Topology


class Mode(IntEnum):
    A_TO_B = 0
    B_TO_A = 1
    MEET = 2


MODE_LABELS = {Mode.A_TO_B: "A→B", Mode.B_TO_A: "B→A", Mode.MEET: "meet"}


class Event(NamedTuple):
    kind: str  # "swap" or "cx"; p, q are physical positions
    p: int
    q: int
    gate: int = -1  # circuit gate index for "cx" events (when known)


@dataclass
class StepResult:
    swaps: int
    executed: list[int]  # gate indices run by this action (the chosen one first)


class RoutingState:
    def __init__(self, topo: Topology, circuit: Circuit, initial_layout: Iterable[int] | None = None,
                 record_events: bool = True) -> None:
        if circuit.n_qubits > topo.n:
            raise ValueError(f"circuit has {circuit.n_qubits} qubits but chip {topo.name} has {topo.n} positions")
        self.topo = topo
        self.circuit = circuit
        self.record_events = record_events

        self.l2p = list(initial_layout) if initial_layout is not None else list(range(circuit.n_qubits))
        if len(self.l2p) != circuit.n_qubits or len(set(self.l2p)) != len(self.l2p) \
                or any(not 0 <= p < topo.n for p in self.l2p):
            raise ValueError(f"invalid initial layout {self.l2p}")
        self.initial_layout = tuple(self.l2p)
        self.p2l = [-1] * topo.n  # -1 = empty position
        for q, p in enumerate(self.l2p):
            self.p2l[p] = q

        qubit_gates: list[list[int]] = [[] for _ in range(circuit.n_qubits)]
        for g, (a, b) in enumerate(circuit.gates):
            qubit_gates[a].append(g)
            qubit_gates[b].append(g)
        self._qubit_gates = tuple(tuple(x) for x in qubit_gates)  # shared, never mutated
        self._ptr = [0] * circuit.n_qubits
        self._done = bytearray(len(circuit))
        self._first_pending = 0
        self.n_done = 0
        self.n_swaps = 0
        self.events: list[Event] = []

        self.execute_ready()  # gates that are already adjacent run for free

    # ------------------------------------------------------------------ queries
    @property
    def is_done(self) -> bool:
        return self.n_done == len(self.circuit)

    def phys_distance(self, g: int) -> int:
        a, b = self.circuit.gates[g]
        return int(self.topo.dist[self.l2p[a], self.l2p[b]])

    def front(self) -> list[int]:
        """Waiting gates, in circuit order."""
        out = []
        gates = self.circuit.gates
        for q, ptr in enumerate(self._ptr):
            qg = self._qubit_gates[q]
            if ptr < len(qg):
                g = qg[ptr]
                a, b = gates[g]
                other = b if q == a else a
                if q < other:
                    optr = self._ptr[other]
                    oqg = self._qubit_gates[other]
                    if optr < len(oqg) and oqg[optr] == g:
                        out.append(g)
        out.sort()
        return out

    def upcoming(self, m: int) -> list[int]:
        """The first m unexecuted gates in circuit order (waiting or not)."""
        out = []
        g = self._first_pending
        n = len(self.circuit)
        while g < n and len(out) < m:
            if not self._done[g]:
                out.append(g)
            g += 1
        return out

    # ---------------------------------------------------------------- mutation
    def clone(self, record_events: bool = False) -> "RoutingState":
        new = object.__new__(RoutingState)
        new.topo = self.topo
        new.circuit = self.circuit
        new.record_events = record_events
        new.l2p = self.l2p.copy()
        new.p2l = self.p2l.copy()
        new.initial_layout = self.initial_layout
        new._qubit_gates = self._qubit_gates
        new._ptr = self._ptr.copy()
        new._done = bytearray(self._done)
        new._first_pending = self._first_pending
        new.n_done = self.n_done
        new.n_swaps = self.n_swaps
        new.events = self.events.copy() if record_events else []
        return new

    def _swap(self, p: int, q: int) -> None:
        if not self.topo.are_adjacent(p, q):
            raise AssertionError(f"internal error: SWAP on non-edge ({p}, {q})")
        lp, lq = self.p2l[p], self.p2l[q]
        self.p2l[p], self.p2l[q] = lq, lp
        if lp >= 0:
            self.l2p[lp] = q
        if lq >= 0:
            self.l2p[lq] = p
        self.n_swaps += 1
        if self.record_events:
            self.events.append(Event("swap", p, q))

    def _run_gate(self, g: int) -> None:
        a, b = self.circuit.gates[g]
        pa, pb = self.l2p[a], self.l2p[b]
        if not self.topo.are_adjacent(pa, pb):
            raise AssertionError(f"internal error: gate {g} run on non-adjacent positions")
        self._done[g] = 1
        self.n_done += 1
        self._ptr[a] += 1
        self._ptr[b] += 1
        while self._first_pending < len(self.circuit) and self._done[self._first_pending]:
            self._first_pending += 1
        if self.record_events:
            self.events.append(Event("cx", pa, pb, g))

    def execute_ready(self) -> list[int]:
        """Run every waiting gate whose qubits are adjacent, until none is left."""
        ran = []
        progress = True
        while progress:
            progress = False
            for g in self.front():
                if self.phys_distance(g) == 1:
                    self._run_gate(g)
                    ran.append(g)
                    progress = True
        return ran

    def apply(self, g: int, mode: Mode) -> StepResult:
        """Apply macro-action (gate g, mode). g must be a waiting gate."""
        if g not in self.front():
            raise ValueError(f"gate {g} is not waiting")
        a, b = self.circuit.gates[g]
        path = self.topo.shortest_path(self.l2p[a], self.l2p[b])
        n_swaps = len(path) - 2  # d - 1
        steps_a = {Mode.A_TO_B: n_swaps, Mode.B_TO_A: 0, Mode.MEET: (n_swaps + 1) // 2}[Mode(mode)]
        steps_b = n_swaps - steps_a

        for i in range(steps_a):  # A walks forward along the path
            self._swap(path[i], path[i + 1])
        last = len(path) - 1
        for i in range(steps_b):  # B walks backward along the path
            self._swap(path[last - i], path[last - i - 1])

        self._run_gate(g)
        executed = [g] + self.execute_ready()
        return StepResult(n_swaps, executed)


# ---------------------------------------------------------------- verification
def verify_routing(topo: Topology, circuit: Circuit, events: Iterable[Event],
                   initial_layout: Iterable[int] | None = None) -> list[str]:
    """Independently check a routed event list. Returns a list of errors ([] = valid).

    Checks: every SWAP and CX is on a chip edge; every CX, mapped back to logical
    qubits, is the next gate in program order on *both* of its qubits (so the
    routed circuit is equivalent: only gates on disjoint qubits were reordered);
    all gates ran.
    """
    errors: list[str] = []
    l2p = list(initial_layout) if initial_layout is not None else list(range(circuit.n_qubits))
    p2l = [-1] * topo.n
    for q, p in enumerate(l2p):
        p2l[p] = q
    per_qubit: list[list[int]] = [[] for _ in range(circuit.n_qubits)]
    for g, (a, b) in enumerate(circuit.gates):
        per_qubit[a].append(g)
        per_qubit[b].append(g)
    ptr = [0] * circuit.n_qubits
    executed = 0

    for i, ev in enumerate(events):
        if not topo.are_adjacent(ev.p, ev.q):
            errors.append(f"event {i}: {ev.kind} on non-edge ({ev.p}, {ev.q})")
            continue
        if ev.kind == "swap":
            lp, lq = p2l[ev.p], p2l[ev.q]
            p2l[ev.p], p2l[ev.q] = lq, lp
            if lp >= 0:
                l2p[lp] = ev.q
            if lq >= 0:
                l2p[lq] = ev.p
        elif ev.kind == "cx":
            a, b = p2l[ev.p], p2l[ev.q]
            if a < 0 or b < 0:
                errors.append(f"event {i}: CX touches an empty position")
                continue
            ga = per_qubit[a][ptr[a]] if ptr[a] < len(per_qubit[a]) else None
            gb = per_qubit[b][ptr[b]] if ptr[b] < len(per_qubit[b]) else None
            if ga is None or ga != gb or circuit.gates[ga] != (a, b):
                errors.append(f"event {i}: CX(q{a},q{b}) is out of program order")
                continue
            ptr[a] += 1
            ptr[b] += 1
            executed += 1
        else:
            errors.append(f"event {i}: unknown kind {ev.kind}")
    if executed != len(circuit):
        errors.append(f"only {executed}/{len(circuit)} gates were executed")
    return errors


def events_to_qiskit(events: Iterable[Event], n_physical: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_physical)
    for ev in events:
        if ev.kind == "swap":
            qc.swap(ev.p, ev.q)
        else:
            qc.cx(ev.p, ev.q)
    return qc


def events_from_qiskit(qc: QuantumCircuit) -> list[Event]:
    events = []
    for inst in qc.data:
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if len(qubits) != 2:
            continue
        kind = "swap" if inst.operation.name == "swap" else "cx"
        events.append(Event(kind, qubits[0], qubits[1]))
    return events


def cx_depth(events: Iterable[Event], n_physical: int) -> int:
    """Depth with each SWAP decomposed into 3 CX (the hardware-relevant depth)."""
    level = [0] * n_physical
    for ev in events:
        cost = 3 if ev.kind == "swap" else 1
        start = max(level[ev.p], level[ev.q])
        level[ev.p] = level[ev.q] = start + cost
    return max(level, default=0)
