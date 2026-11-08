"""State/action encodings shared by the environment and the trained agents.

Action space: 3K discrete actions. Action index = slot * 3 + mode, where slot
k picks the k-th waiting gate (in circuit order) and mode is A→B / B→A / MEET.
Slots beyond the number of waiting gates are masked out. The size depends only
on K, never on the chip, so one agent can run on any topology.

Two encodings of the state:

* tabular_key(): the exact discrete state (layout + next K waiting gates + next
  L later gates), used as a dictionary key by tabular Q-learning.

* action_features(): for the DQN. For every action we run the simulator one step
  on a copy of the state (the transition is deterministic and known) and
  describe the result: SWAPs paid now, gates run, how far apart upcoming
  gate pairs end up, and so on. The network scores each (state, action) pair
  from this description, with weights shared across actions. All features are
  normalised by chip size, so the same network works on lines, rings, grids
  and stars of different sizes. This is how difficulty #3 from the design
  discussion (different chips → different state shapes) is handled.
"""

from __future__ import annotations

import numpy as np

from .simulator import Mode, RoutingState

N_MODES = 3
N_FEATURES = 17
LOOKAHEAD = 30            # gates considered by the lookahead features
DECAYS = (0.5, 0.8, 0.95)  # short / medium / long horizon


def n_actions(k: int) -> int:
    return k * N_MODES


def decode(action: int) -> tuple[int, Mode]:
    return action // N_MODES, Mode(action % N_MODES)


def action_mask(state: RoutingState, k: int) -> np.ndarray:
    mask = np.zeros(n_actions(k), dtype=bool)
    mask[: min(k, len(state.front())) * N_MODES] = True
    return mask


def to_gate(state: RoutingState, action: int, k: int) -> tuple[int, Mode]:
    slot, mode = decode(action)
    front = state.front()
    if slot >= min(k, len(front)):
        raise ValueError(f"action {action} is masked (only {len(front)} waiting gates)")
    return front[slot], mode


def front_cost(state: RoutingState) -> int:
    """Total SWAPs needed if every waiting gate were served on its own: sum of (d-1)."""
    return sum(state.phys_distance(g) - 1 for g in state.front())


def _lookahead(state: RoutingState) -> list[float]:
    dists = [state.phys_distance(g) - 1 for g in state.upcoming(LOOKAHEAD)]
    out = []
    for w in DECAYS:
        weights = w ** np.arange(LOOKAHEAD)
        out.append(float(np.dot(weights[: len(dists)], dists)) / float(weights.sum()))
    return out


def action_features(state: RoutingState, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [3K, N_FEATURES] float32, mask [3K] bool)."""
    topo = state.topo
    diam = float(max(topo.diameter, 1))
    feats = np.zeros((n_actions(k), N_FEATURES), dtype=np.float32)
    mask = action_mask(state, k)
    front = state.front()[:k]
    look_before = _lookahead(state)
    n_total = max(len(state.circuit), 1)

    for slot, g in enumerate(front):
        d_before = state.phys_distance(g)
        for mode in Mode:
            nxt = state.clone()
            res = nxt.apply(g, mode)
            look_after = _lookahead(nxt)
            remaining = len(state.circuit) - nxt.n_done
            f = feats[slot * N_MODES + mode]
            f[0] = res.swaps / diam
            f[1] = len(res.executed) / k
            f[2] = float(nxt.is_done)
            f[3] = front_cost(nxt) / (k * diam)
            f[4] = len(nxt.front()) / k
            f[5:8] = [x / diam for x in look_after]
            f[8:11] = [(b - a) / diam for b, a in zip(look_before, look_after)]
            f[11 + mode] = 1.0
            f[14] = d_before / diam
            f[15] = remaining / 50.0
            f[16] = remaining / n_total
    return feats, mask


def tabular_key(state: RoutingState, k: int, lookahead: int) -> tuple:
    gates = state.circuit.gates
    front = state.front()[:k]
    later = [g for g in state.upcoming(k + lookahead) if g not in front][:lookahead]
    return (tuple(state.p2l), tuple(gates[g] for g in front), tuple(gates[g] for g in later))
