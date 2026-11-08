import numpy as np
import pytest
from qiskit.quantum_info import Operator

from routing.circuit import Circuit, random_circuit
from routing.simulator import (Event, Mode, RoutingState, events_to_qiskit, verify_routing)
from routing.topology import from_spec, grid, line, ring, star


# ------------------------------------------------------------------ topology
def test_distances_and_paths():
    t = line(4)
    assert t.dist[0, 3] == 3
    assert t.shortest_path(0, 3) == [0, 1, 2, 3]
    r = ring(6)
    assert r.dist[0, 3] == 3 and r.dist[0, 5] == 1
    g = grid(3, 3)
    assert g.dist[0, 8] == 4
    s = star(5)
    assert s.dist[1, 2] == 2 and s.shortest_path(1, 2) == [1, 0, 2]


def test_shortest_path_deterministic_and_valid():
    for spec in ["line:6", "ring:7", "grid:3x3", "star:6"]:
        t = from_spec(spec)
        for a in range(t.n):
            for b in range(t.n):
                p = t.shortest_path(a, b)
                assert p == t.shortest_path(a, b)
                assert len(p) - 1 == t.dist[a, b]
                assert all(t.are_adjacent(x, y) for x, y in zip(p, p[1:]))


def test_bad_spec():
    with pytest.raises(ValueError):
        from_spec("hexagon:4")


# ------------------------------------------------------------- waiting gates
def test_waiting_gates_readme_example():
    # README: 1. CX(q0,q1) 2. CX(q2,q3) 3. CX(q1,q2) -> gates 1 and 2 wait, 3 does not.
    c = Circuit(4, ((0, 1), (2, 3), (1, 2)))
    s = RoutingState(line(4), c, initial_layout=[0, 2, 1, 3], record_events=False)
    # layout puts q0 and q1 apart, q2 and q3 apart, so nothing runs for free
    assert s.n_done == 0
    assert s.front() == [0, 1]


def test_free_gates_run_at_start():
    c = Circuit(4, ((0, 1), (1, 2), (2, 3)))
    s = RoutingState(line(4), c)
    assert s.is_done and s.n_swaps == 0


# ---------------------------------------------------------- macro-actions
def test_modes_on_line_readme_example():
    # README: CX(q0,q3) on P0-P1-P2-P3 needs 2 SWAPs in every mode.
    c = Circuit(4, ((0, 3),))
    for mode, final in [(Mode.A_TO_B, [2, 0, 1, 3]), (Mode.B_TO_A, [0, 2, 3, 1]), (Mode.MEET, [1, 0, 3, 2])]:
        s = RoutingState(line(4), c)
        res = s.apply(0, mode)
        assert res.swaps == 2 and s.is_done
        assert s.l2p == final, (mode, s.l2p)


def test_who_moves_changes_future_cost():
    # Worked example from the design discussion: G1=CX(q0,q2), G2=CX(q1,q3) on a 4-line.
    c = Circuit(4, ((0, 2), (1, 3)))
    s = RoutingState(line(4), c)
    s.apply(0, Mode.B_TO_A)  # q2 moves towards q0 -> q1, q3 become neighbours
    assert s.is_done and s.n_swaps == 1
    s = RoutingState(line(4), c)
    s.apply(0, Mode.A_TO_B)  # q0 moves -> q1, q3 end up 3 apart
    assert not s.is_done
    s.apply(1, Mode.A_TO_B)
    assert s.is_done and s.n_swaps == 3


def test_apply_rejects_blocked_gate():
    c = Circuit(4, ((0, 3), (0, 2)))
    s = RoutingState(line(4), c)
    with pytest.raises(ValueError):
        s.apply(1, Mode.MEET)


def test_clone_is_independent():
    c = Circuit(4, ((0, 3), (1, 2)))
    s = RoutingState(line(4), c)
    t = s.clone()
    t.apply(0, Mode.MEET)
    assert s.n_swaps == 0 and s.l2p == [0, 1, 2, 3]


# ---------------------------------------------------- end-to-end correctness
def _route_randomly(topo, circ, rng):
    s = RoutingState(topo, circ)
    while not s.is_done:
        front = s.front()
        s.apply(front[rng.integers(len(front))], Mode(rng.integers(3)))
    return s


@pytest.mark.parametrize("spec", ["line:5", "ring:6", "grid:2x3", "grid:3x3", "star:6"])
def test_random_routing_always_valid(spec):
    rng = np.random.default_rng(0)
    topo = from_spec(spec)
    for _ in range(30):
        n_q = int(rng.integers(2, topo.n + 1))  # also tests empty positions
        circ = random_circuit(n_q, int(rng.integers(1, 30)), rng)
        s = _route_randomly(topo, circ, rng)
        assert verify_routing(topo, circ, s.events) == []
        assert sum(e.kind == "swap" for e in s.events) == s.n_swaps


def test_verifier_catches_errors():
    topo, circ = line(4), Circuit(4, ((0, 3),))
    assert verify_routing(topo, circ, [Event("cx", 0, 3)])  # non-edge
    assert verify_routing(topo, circ, [])  # gate never ran
    circ2 = Circuit(3, ((0, 1), (1, 2)))
    assert verify_routing(line(3), circ2, [Event("cx", 1, 2), Event("cx", 0, 1)])  # wrong order


def test_routed_circuit_is_unitarily_equivalent():
    """Strongest check: the routed circuit equals the original up to the final qubit permutation."""
    rng = np.random.default_rng(1)
    topo = grid(2, 3)
    circ = random_circuit(6, 15, rng)
    s = _route_randomly(topo, circ, rng)
    routed = events_to_qiskit(s.events, topo.n)
    # undo the final permutation with (unconstrained) swaps, then compare unitaries
    p2l = s.p2l.copy()
    for target in range(topo.n):
        cur = p2l.index(target)
        if cur != target:
            routed.swap(cur, target)
            p2l[cur], p2l[target] = p2l[target], p2l[cur]
    assert Operator(routed).equiv(Operator(circ.to_qiskit(topo.n)))
