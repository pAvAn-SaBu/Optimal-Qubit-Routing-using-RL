"""Side-by-side replay: trained agent vs Qiskit SABRE on the same circuit and chip.

Each frame is one event (a SWAP or a gate). Nodes are physical positions and
their colour/label is the logical qubit sitting there, so qubit movement is
visible. Orange edge = SWAP, green edge = gate running. For the agent, the
caption also shows the decision behind it: which waiting gate it served and
who moved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .circuit import Circuit
from .policies import RoutingResult
from .simulator import MODE_LABELS
from .topology import Topology

log = logging.getLogger(__name__)


@dataclass
class Frame:
    p2l: list[int]
    highlight: tuple[int, int] | None
    kind: str  # "start", "swap", "cx", "end"
    caption: str
    swaps: int
    gates_done: int


def build_frames(topo: Topology, circuit: Circuit, result: RoutingResult) -> list[Frame]:
    p2l = [-1] * topo.n
    for q in range(circuit.n_qubits):
        p2l[q] = q  # trivial initial layout
    decision_at = {d.event_start: d for d in result.decisions}
    frames = [Frame(p2l.copy(), None, "start", "initial layout", 0, 0)]
    swaps = gates = 0
    current = ""
    for i, ev in enumerate(result.events):
        if i in decision_at:
            d = decision_at[i]
            a, b = circuit.gates[d.gate]
            current = f"decision: serve CX(q{a},q{b}), {MODE_LABELS[d.mode]} → {d.swaps} SWAP(s)"
        if ev.kind == "swap":
            p2l[ev.p], p2l[ev.q] = p2l[ev.q], p2l[ev.p]
            swaps += 1
            text = f"SWAP P{ev.p}↔P{ev.q}"
        else:
            gates += 1
            text = f"CX(q{p2l[ev.p]},q{p2l[ev.q]}) runs on P{ev.p}–P{ev.q}"
        frames.append(Frame(p2l.copy(), (ev.p, ev.q), ev.kind, f"{current}\n{text}" if current else text,
                            swaps, gates))
    frames.append(Frame(p2l.copy(), None, "end", f"done: {swaps} SWAPs, CX-depth {result.depth}", swaps, gates))
    return frames


def narrate(circuit: Circuit, result: RoutingResult) -> str:
    """Text log of the agent's decisions (printed to the console)."""
    lines = [f"{result.method}: {result.swaps} SWAPs, CX-depth {result.depth}, valid={result.valid}"]
    for n, d in enumerate(result.decisions, 1):
        a, b = circuit.gates[d.gate]
        free = d.executed[1:]
        extra = f"; also ran for free: {', '.join(f'CX(q{circuit.gates[g][0]},q{circuit.gates[g][1]})' for g in free)}" if free else ""
        lines.append(f"  step {n:2d}: serve gate #{d.gate} CX(q{a},q{b}) mode {MODE_LABELS[d.mode]:<4} "
                     f"→ {d.swaps} SWAP(s){extra}")
    return "\n".join(lines)


def animate(topo: Topology, circuit: Circuit, results: list[RoutingResult], gif_path: Path | None,
            show: bool, interval_ms: int = 900) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    all_frames = [build_frames(topo, circuit, r) for r in results]
    n_frames = max(len(f) for f in all_frames)
    cmap = plt.get_cmap("tab20" if circuit.n_qubits > 10 else "tab10")
    xs = [c[0] for c in topo.coords]
    ys = [c[1] for c in topo.coords]
    pad = 0.6

    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 6.2))
    axes = [axes] if len(results) == 1 else list(axes)
    gate_text = ", ".join(f"q{a}-q{b}" for a, b in circuit.gates)
    fig.suptitle(f"{topo.name}: {len(circuit)} two-qubit gates\n{gate_text}", fontsize=9)

    def draw(i: int) -> None:
        for ax, frames, res in zip(axes, all_frames, results):
            fr = frames[min(i, len(frames) - 1)]
            ax.clear()
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
            ax.set_aspect("equal")
            ax.axis("off")
            for a, b in topo.edges:
                hl = fr.highlight is not None and {a, b} == set(fr.highlight)
                color = {"swap": "tab:orange", "cx": "tab:green"}.get(fr.kind, "lightgray") if hl else "lightgray"
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=color, lw=6 if hl else 2, zorder=1)
            for p in range(topo.n):
                q = fr.p2l[p]
                face = cmap(q % cmap.N) if q >= 0 else "white"
                ax.scatter(xs[p], ys[p], s=900, color=face, edgecolors="black", zorder=2)
                ax.text(xs[p], ys[p], f"q{q}" if q >= 0 else "–", ha="center", va="center",
                        fontsize=10, fontweight="bold", zorder=3)
                ax.text(xs[p], ys[p] - 0.32, f"P{p}", ha="center", va="top", fontsize=7, color="dimgray")
            done = fr.kind == "end" or i >= len(frames) - 1
            ax.set_title(f"{res.method}\nSWAPs: {fr.swaps}   gates run: {fr.gates_done}/{len(circuit)}"
                         + ("   ✓ finished" if done else ""), fontsize=11)
            ax.text(0.5, -0.02, fr.caption, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=interval_ms, repeat=False)
    if gif_path:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(gif_path, writer=PillowWriter(fps=max(1, round(1000 / interval_ms))))
        log.info("saved animation to %s", gif_path)
    if show:
        plt.show()
    plt.close(fig)


def summary_plot(topo: Topology, results: list[RoutingResult], path: Path) -> None:
    # Figure() without pyplot: saving this must not switch the backend used by animate()'s window
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13, 3.8))
    axes = fig.subplots(1, 3)
    names = [r.method for r in results]
    colors = ["tab:blue" if n not in ("SABRE", "greedy") else ("tab:red" if n == "SABRE" else "tab:gray")
              for n in names]
    axes[0].bar(names, [r.swaps for r in results], color=colors)
    axes[0].set_title("SWAPs inserted (lower is better)")
    axes[1].bar(names, [r.depth for r in results], color=colors)
    axes[1].set_title("CX depth, SWAP = 3 CX (lower is better)")
    for r, c in zip(results, colors):  # cumulative SWAPs over the gates executed
        xs, ys, s, g = [0], [0], 0, 0
        for ev in r.events:
            if ev.kind == "swap":
                s += 1
            else:
                g += 1
                xs.append(g); ys.append(s)
        axes[2].step(xs, ys, where="post", label=r.method, color=c)
    axes[2].set_xlabel("gates executed"); axes[2].set_ylabel("cumulative SWAPs")
    axes[2].set_title("When the SWAPs were spent"); axes[2].legend()
    fig.suptitle(f"{topo.name}: " + ", ".join(f"{r.method} {'valid' if r.valid else 'INVALID'}" for r in results))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    log.info("saved %s", path)

