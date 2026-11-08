"""
Interactive Qubit Routing Visualizer (mid-review demo)

The user chooses:
  - number of qubits
  - hardware topology (line / ring / grid / star / custom edges)
  - the circuit's gates

Qiskit's SABRE router then routes the circuit, and the app replays the result
step by step with a narration of what is happening and why.

Requires:  pip install qiskit matplotlib
           sudo pacman -S tk   (tkinter)
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

ONE_QUBIT_GATES = {"h", "x", "y", "z", "s", "t"}
TWO_QUBIT_GATES = {"cx", "cz"}

DEFAULT_GATES = "cx 0 1\ncx 0 3\ncx 1 3\ncx 0 2"


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def circle_positions(n, radius=None):
    radius = radius or max(1.0, n / 5)
    return {
        i: (radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n))
        for i in range(n)
    }


def is_connected(n, edges):
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {0}, [0]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == n


def build_topology(kind, n, custom_text):
    """Return (positions, edge list) for the chosen topology."""
    if kind == "Line":
        edges = [(i, i + 1) for i in range(n - 1)]
        pos = {i: (i, 0) for i in range(n)}
    elif kind == "Ring":
        edges = [(i, i + 1) for i in range(n - 1)]
        if n > 2:
            edges.append((n - 1, 0))
        pos = circle_positions(n)
    elif kind == "Grid":
        cols = math.ceil(math.sqrt(n))
        edges = []
        for i in range(n):
            if (i % cols) != cols - 1 and i + 1 < n:
                edges.append((i, i + 1))
            if i + cols < n:
                edges.append((i, i + cols))
        pos = {i: (i % cols, -(i // cols)) for i in range(n)}
    elif kind == "Star":
        edges = [(0, i) for i in range(1, n)]
        pos = {0: (0, 0)}
        outer = circle_positions(n - 1, radius=max(1.5, n / 4))
        for i in range(1, n):
            pos[i] = outer[i - 1]
    elif kind == "Custom":
        edges = []
        for chunk in custom_text.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                a, b = (int(x) for x in chunk.split("-"))
            except ValueError:
                raise ValueError(f"Bad edge '{chunk}'. Use the form 0-1, 1-2, 2-3.")
            if not (0 <= a < n and 0 <= b < n) or a == b:
                raise ValueError(f"Edge '{chunk}' is invalid for {n} qubits.")
            edges.append((a, b))
        pos = circle_positions(n)
    else:
        raise ValueError(f"Unknown topology {kind}")

    if not edges:
        raise ValueError("The topology has no connections.")
    if not is_connected(n, edges):
        raise ValueError("The topology must be connected (every qubit reachable from every other).")
    return pos, edges


def parse_gates(text, n):
    """Parse lines like 'cx 0 1' or 'h 2' into [(name, [qubits])]."""
    gates = []
    for raw in text.replace(";", "\n").split("\n"):
        line = raw.replace(",", " ").strip().lower()
        if not line:
            continue
        parts = line.split()
        name = parts[0]
        try:
            qubits = [int(p) for p in parts[1:]]
        except ValueError:
            raise ValueError(f"Bad gate line '{raw.strip()}'. Use e.g. 'cx 0 1'.")
        if name in ONE_QUBIT_GATES:
            expected = 1
        elif name in TWO_QUBIT_GATES:
            expected = 2
        else:
            raise ValueError(
                f"Unsupported gate '{name}'. Supported: "
                + ", ".join(sorted(ONE_QUBIT_GATES | TWO_QUBIT_GATES))
            )
        if len(qubits) != expected:
            raise ValueError(f"Gate '{name}' needs {expected} qubit(s): '{raw.strip()}'")
        if any(q < 0 or q >= n for q in qubits):
            raise ValueError(f"Qubit index out of range (0 to {n - 1}) in '{raw.strip()}'")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Gate '{name}' needs two different qubits: '{raw.strip()}'")
        gates.append((name, qubits))
    if not gates:
        raise ValueError("Enter at least one gate.")
    return gates


# ---------------------------------------------------------------------------
# Routing + narration
# ---------------------------------------------------------------------------
def respects_connectivity(circuit, cmap):
    allowed = set(cmap.get_edges())
    for inst in circuit.data:
        if len(inst.qubits) == 2:
            a = circuit.find_bit(inst.qubits[0]).index
            b = circuit.find_bit(inst.qubits[1]).index
            if (a, b) not in allowed:
                return False
    return True


def fmt_gate(name, qs):
    return f"{name}(" + ",".join(f"q{q}" for q in qs) + ")"


def build_steps(n, gates, cmap, seed):
    qc = QuantumCircuit(n)
    for name, qs in gates:
        getattr(qc, name)(*qs)

    routed = transpile(
        qc,
        coupling_map=cmap,
        layout_method="trivial",
        routing_method="sabre",
        optimization_level=0,
        seed_transpiler=seed,
    )

    remaining = [(name, tuple(qs)) for name, qs in gates if name in TWO_QUBIT_GATES]

    def gaps(mapping):
        """Hops still needed (distance - 1) for each pending gate, given the mapping."""
        l2p = {l: p for p, l in mapping.items()}
        return [(g, max(cmap.distance(l2p[g[1][0]], l2p[g[1][1]]) - 1, 0)) for g in remaining]

    mapping = {p: p for p in range(n)}
    start_gaps = gaps(mapping)
    blocked = [fmt_gate(g[0], g[1]) for g, d in start_gaps if d > 0]
    ready = [fmt_gate(g[0], g[1]) for g, d in start_gaps if d == 0]
    text = (
        "Initial layout: logical qubit q_i starts on physical qubit P_i. "
        f"{len(blocked)} of {len(remaining)} two-qubit gate(s) cannot run directly because their qubits are not adjacent"
        + (f" ({', '.join(blocked)})" if blocked else "")
        + "."
    )
    if ready:
        text += f" Already adjacent: {', '.join(ready)}."
    steps = [dict(kind="start", name="start", phys=[], mapping=dict(mapping),
                  text=text, pending=list(remaining))]

    swaps = 0
    for inst in routed.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        phys = [routed.find_bit(q).index for q in inst.qubits]
        i = len(steps)

        if name == "swap":
            swaps += 1
            before = gaps(mapping)
            a, b = phys
            la, lb = mapping[a], mapping[b]
            mapping[a], mapping[b] = mapping[b], mapping[a]
            after = gaps(mapping)
            total_b = sum(d for _, d in before)
            total_a = sum(d for _, d in after)

            text = (
                f"SWAP on P{a}-P{b}: exchanges logical q{la} and q{lb}. "
                f"Total hops still needed by pending gates: {total_b} -> {total_a}."
            )
            focus = next(((g, d) for g, d in before if d > 0), None)
            if focus:
                g, d_before = focus
                d_after = dict((id(x), d) for x, d in after)[id(g)]
                text += (
                    f" Next blocked gate {fmt_gate(g[0], g[1])}: "
                    f"{d_before + 1} hop(s) apart -> {d_after + 1} hop(s) apart."
                )
            if total_a >= total_b:
                text += (" The total did not shrink: SABRE also weighs upcoming gates "
                         "(lookahead), so this SWAP sets up later interactions.")
            steps.append(dict(kind="swap", name="swap", phys=phys, mapping=dict(mapping),
                              text=text, pending=list(remaining)))

        elif len(phys) == 2:
            logical = tuple(mapping[p] for p in phys)
            for k, (rn, rq) in enumerate(remaining):
                if rn == name and (rq == logical or (name == "cz" and rq == logical[::-1])):
                    remaining.pop(k)
                    break
            text = (
                f"{name.upper()} between logical q{logical[0]} and q{logical[1]}, currently on "
                f"P{phys[0]} and P{phys[1]}. They are adjacent, so the gate runs directly."
            )
            steps.append(dict(kind="gate2", name=name, phys=phys, mapping=dict(mapping),
                              text=text, pending=list(remaining)))

        else:
            logical = mapping[phys[0]]
            text = (
                f"{name.upper()} on logical q{logical} (on P{phys[0]}). "
                "Single-qubit gates need no routing."
            )
            steps.append(dict(kind="gate1", name=name, phys=phys, mapping=dict(mapping),
                              text=text, pending=list(remaining)))

    ok = respects_connectivity(routed, cmap)
    steps.append(dict(
        kind="done", name="done", phys=[], mapping=dict(mapping),
        text=(f"Routing complete. {swaps} SWAP(s) inserted, depth {qc.depth()} -> {routed.depth()}. "
              f"Every two-qubit gate now sits on connected qubits (valid: {ok})."),
        pending=[],
    ))

    stats = (
        f"Original depth: {qc.depth()}   |   Routed depth: {routed.depth()}   |   "
        f"SWAPs inserted: {swaps}   |   Valid on hardware: {ok}   |   Router: Qiskit SABRE"
    )
    return steps, stats


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class RoutingApp:
    def __init__(self, root):
        self.root = root
        root.title("Qubit Routing Visualizer")
        self.steps, self.idx = [], 0
        self.playing, self._after = False, None
        self.pos, self.edges, self.n = {}, set(), 0
        self._build_ui()
        self._draw_placeholder()

    # ---- layout ----
    def _build_ui(self):
        root = self.root
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root, padding=10)
        left.grid(row=0, column=0, sticky="ns")

        ttk.Label(left, text="Number of qubits").pack(anchor="w")
        self.n_var = tk.IntVar(value=4)
        ttk.Spinbox(left, from_=2, to=12, textvariable=self.n_var, width=6).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Hardware topology").pack(anchor="w")
        self.topo_var = tk.StringVar(value="Line")
        ttk.Combobox(left, textvariable=self.topo_var, state="readonly", width=12,
                     values=["Line", "Ring", "Grid", "Star", "Custom"]).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Custom edges (Custom only)\ne.g. 0-1, 1-2, 2-3", justify="left").pack(anchor="w")
        self.custom_var = tk.StringVar(value="0-1, 1-2, 2-3")
        ttk.Entry(left, textvariable=self.custom_var, width=24).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Circuit gates, one per line\nh q  |  x/y/z/s/t q\ncx a b  |  cz a b",
                  justify="left").pack(anchor="w")
        self.gate_text = tk.Text(left, width=24, height=10)
        self.gate_text.insert("1.0", DEFAULT_GATES)
        self.gate_text.pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="SABRE random seed").pack(anchor="w")
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(left, from_=0, to=9999, textvariable=self.seed_var, width=8).pack(anchor="w", pady=(0, 12))

        ttk.Button(left, text="Route & Visualize", command=self.route).pack(fill="x")

        right = ttk.Frame(root, padding=10)
        right.grid(row=0, column=1, sticky="nsew")

        self.fig = Figure(figsize=(7, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        widget.bind("<Button-1>", lambda e: widget.focus_set())
        widget.bind("<Left>", lambda e: self.step(-1))
        widget.bind("<Right>", lambda e: self.step(1))

        self.step_lbl = ttk.Label(right, font=("TkDefaultFont", 11, "bold"))
        self.step_lbl.pack(anchor="w", pady=(6, 2))
        self.narr_lbl = tk.Label(right, justify="left", anchor="w", wraplength=640,
                                 font=("TkDefaultFont", 11), bg="#f1f1f1", padx=8, pady=6)
        self.narr_lbl.pack(fill="x")
        self.pend_lbl = ttk.Label(right, wraplength=640, justify="left")
        self.pend_lbl.pack(anchor="w", pady=(4, 0))
        self.stats_lbl = ttk.Label(right, wraplength=640, justify="left", foreground="#444444")
        self.stats_lbl.pack(anchor="w", pady=(2, 6))

        ctrl = ttk.Frame(right)
        ctrl.pack(anchor="w")
        ttk.Button(ctrl, text="|<  Reset", command=self.reset).pack(side="left", padx=2)
        ttk.Button(ctrl, text="< Prev", command=lambda: self.step(-1)).pack(side="left", padx=2)
        self.play_btn = ttk.Button(ctrl, text="Play", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=2)
        ttk.Button(ctrl, text="Next >", command=lambda: self.step(1)).pack(side="left", padx=2)
        ttk.Label(ctrl, text="   Seconds/step").pack(side="left")
        self.speed = tk.DoubleVar(value=2.0)
        ttk.Scale(ctrl, from_=0.5, to=4.0, variable=self.speed, length=120).pack(side="left", padx=4)

    # ---- actions ----
    def route(self):
        try:
            n = int(self.n_var.get())
            seed = int(self.seed_var.get())
            if not 2 <= n <= 12:
                raise ValueError("Number of qubits must be between 2 and 12.")
            pos, edges = build_topology(self.topo_var.get(), n, self.custom_var.get())
            gates = parse_gates(self.gate_text.get("1.0", "end"), n)
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Input error", str(e))
            return

        both = edges + [(b, a) for a, b in edges]
        cmap = CouplingMap(couplinglist=both)
        try:
            steps, stats = build_steps(n, gates, cmap, seed)
        except Exception as e:  # surface Qiskit errors in the UI
            messagebox.showerror("Routing error", str(e))
            return

        self._stop()
        self.n, self.pos = n, pos
        self.edges = {tuple(sorted(e)) for e in edges}
        self.steps, self.idx = steps, 0
        self.stats_lbl.config(text=stats)
        self.draw()

    def reset(self):
        if self.steps:
            self._stop()
            self.idx = 0
            self.draw()

    def step(self, d):
        if not self.steps:
            return
        self._stop()
        self.idx = max(0, min(len(self.steps) - 1, self.idx + d))
        self.draw()

    def _stop(self):
        self.playing = False
        self.play_btn.config(text="Play")
        if self._after is not None:
            self.root.after_cancel(self._after)
            self._after = None

    def toggle_play(self):
        if not self.steps:
            return
        if self.playing:
            self._stop()
            return
        if self.idx >= len(self.steps) - 1:
            self.idx = 0
            self.draw()
        self.playing = True
        self.play_btn.config(text="Pause")
        self._schedule()

    def _schedule(self):
        self._after = self.root.after(int(self.speed.get() * 1000), self._tick)

    def _tick(self):
        self._after = None
        if not self.playing:
            return
        if self.idx < len(self.steps) - 1:
            self.idx += 1
            self.draw()
            self._schedule()
        else:
            self._stop()

    # ---- drawing ----
    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.axis("off")
        self.ax.text(0.5, 0.5, "Choose qubits, topology and gates,\nthen press 'Route & Visualize'",
                     ha="center", va="center", fontsize=13, color="gray", transform=self.ax.transAxes)
        self.canvas.draw_idle()

    def draw(self):
        step = self.steps[self.idx]
        ax, pos, m = self.ax, self.pos, step["mapping"]
        ax.clear()

        for a, b in self.edges:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color="lightgray", lw=3, zorder=1)

        phys, kind = step["phys"], step["kind"]
        node_color = {p: "white" for p in pos}
        if len(phys) == 2:
            color = "red" if kind == "swap" else "green"
            (x0, y0), (x1, y1) = pos[phys[0]], pos[phys[1]]
            ax.plot([x0, x1], [y0, y1], color=color, lw=6, zorder=2)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, step["name"].upper(),
                    ha="center", fontsize=10, color=color, fontweight="bold", zorder=5)
            for p in phys:
                node_color[p] = "#ffd6d6" if kind == "swap" else "#d6f5d6"
        elif len(phys) == 1:
            node_color[phys[0]] = "#d6e6ff"
            x, y = pos[phys[0]]
            ax.text(x, y + 0.45, step["name"].upper(), ha="center", fontsize=10,
                    color="#2255aa", fontweight="bold", zorder=5)

        size = 1100 if self.n <= 8 else 700
        for p, (x, y) in pos.items():
            ax.scatter(x, y, s=size, color=node_color[p], edgecolor="black", zorder=3)
            ax.text(x, y, f"q{m[p]}", ha="center", va="center", fontsize=13, zorder=4)
            ax.text(x, y - 0.38, f"P{p}", ha="center", fontsize=9, color="gray")

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
        ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Circles = physical qubits (P), label = logical qubit (q) stored there",
                     fontsize=10)
        ax.text(0.5, -0.02, "red = SWAP    green = two-qubit gate    blue = single-qubit gate",
                transform=ax.transAxes, ha="center", va="top", fontsize=9, color="gray")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        last = len(self.steps) - 1
        label = {"start": "Start", "done": "Finished"}.get(kind, f"Operation: {step['name'].upper()}")
        self.step_lbl.config(text=f"Step {self.idx} / {last}  -  {label}")
        self.narr_lbl.config(text=step["text"])
        pend = step["pending"]
        self.pend_lbl.config(
            text="Pending two-qubit gates: " + (", ".join(fmt_gate(n_, q) for n_, q in pend) or "none")
        )


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x720")
    RoutingApp(root)
    root.mainloop()
