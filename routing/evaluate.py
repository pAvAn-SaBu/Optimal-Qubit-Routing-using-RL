"""Benchmark: agents vs greedy vs Qiskit SABRE on unseen circuits.

Every method gets the same circuits, the same chip and the same trivial initial
layout. Every routed result is checked by the independent verifier (edges +
program order); an invalid result is counted and reported, never hidden.

Outputs in the run directory:
  results.csv          one row per (topology, circuit, method)
  summary.csv / .md    per topology × method: mean SWAPs, mean depth, validity,
                       win/tie/loss vs SABRE, Wilcoxon signed-rank p-value
  comparison.png       mean SWAPs per topology (bars) + per-circuit scatter
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .circuit import random_circuit
from .policies import Policy, greedy_policy, route, route_sabre
from .topology import Topology

log = logging.getLogger(__name__)

SEEN = ["line:5", "line:7", "ring:6", "ring:8", "grid:2x3", "grid:3x3", "star:6", "star:8"]
UNSEEN = ["line:10", "ring:10", "grid:3x4", "star:10"]


@dataclass
class Method:
    name: str
    policy: Policy | None          # None = SABRE
    topologies: set[str] | None = None  # restrict (tabular agents only know their own chip)


def run_benchmark(topologies: list[Topology], methods: list[Method], circuits: int, gates: int,
                  seed: int, out_dir: Path, unseen: set[str] = frozenset()) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for topo in topologies:
        for i in range(circuits):
            circ = random_circuit(topo.n, gates, rng)
            for m in methods:
                if m.topologies is not None and topo.name not in m.topologies:
                    continue
                res = route_sabre(topo, circ, seed=i) if m.policy is None else route(topo, circ, m.policy, m.name)
                rows.append({"topology": topo.name, "seen_in_training": topo.name not in unseen,
                             "circuit": i, "method": m.name, "swaps": res.swaps, "depth": res.depth,
                             "valid": res.valid, "seconds": round(res.seconds, 4)})
                if not res.valid:
                    log.error("%s produced an INVALID routing on %s #%d: %s", m.name, topo.name, i, res.errors[:3])
        log.info("finished %s", topo.name)

    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    write_summary(summary, out_dir)
    plot_comparison(rows, summary, out_dir)
    return summary


def summarise(rows: list[dict]) -> list[dict]:
    by = {}
    for r in rows:
        by.setdefault((r["topology"], r["method"]), {})[r["circuit"]] = r
    topos = list(dict.fromkeys(r["topology"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for t in topos:
        sabre = by.get((t, "SABRE"), {})
        for m in methods:
            res = by.get((t, m))
            if not res:
                continue
            ids = sorted(res)
            sw = np.array([res[i]["swaps"] for i in ids])
            row = {"topology": t, "seen_in_training": res[ids[0]]["seen_in_training"], "method": m,
                   "circuits": len(ids), "mean_swaps": round(float(sw.mean()), 2),
                   "mean_depth": round(float(np.mean([res[i]["depth"] for i in ids])), 2),
                   "valid_pct": round(100 * float(np.mean([res[i]["valid"] for i in ids])), 1)}
            if m != "SABRE" and sabre:
                ref = np.array([sabre[i]["swaps"] for i in ids])
                row["swaps_vs_sabre"] = round(float(sw.sum() / max(ref.sum(), 1)), 3)
                row["win/tie/loss"] = f"{(sw < ref).sum()}/{(sw == ref).sum()}/{(sw > ref).sum()}"
                diff = sw - ref
                row["wilcoxon_p"] = round(float(wilcoxon(diff).pvalue), 4) if np.any(diff) else 1.0
            out.append(row)
    return out


def write_summary(summary: list[dict], out_dir: Path) -> None:
    keys = ["topology", "seen_in_training", "method", "circuits", "mean_swaps", "swaps_vs_sabre",
            "win/tie/loss", "wilcoxon_p", "mean_depth", "valid_pct"]
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k, "") for k in keys})
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in summary:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n\nswaps_vs_sabre < 1 means fewer SWAPs than SABRE. "
                                        "win/tie/loss counts circuits vs SABRE. wilcoxon_p < 0.05 means the "
                                        "difference to SABRE is statistically significant.\n")
    print("\n" + table + "\n")


def plot_comparison(rows: list[dict], summary: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topos = list(dict.fromkeys(r["topology"] for r in summary))
    methods = list(dict.fromkeys(r["method"] for r in summary))
    colors = {"SABRE": "tab:red", "greedy": "tab:gray", "DQN": "tab:blue", "tabular-Q": "tab:green"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2.2, 1]})

    width = 0.8 / len(methods)
    for j, m in enumerate(methods):
        xs, ys = [], []
        for i, t in enumerate(topos):
            hit = [r for r in summary if r["topology"] == t and r["method"] == m]
            if hit:
                xs.append(i + (j - (len(methods) - 1) / 2) * width)
                ys.append(hit[0]["mean_swaps"])
        ax1.bar(xs, ys, width, label=m, color=colors.get(m))
    labels = [t + ("" if any(r["seen_in_training"] for r in summary if r["topology"] == t) else "\n(unseen)")
              for t in topos]
    ax1.set_xticks(range(len(topos)), labels, fontsize=8)
    ax1.set_ylabel("mean SWAPs per circuit (lower is better)")
    ax1.set_title("SWAPs inserted, per chip")
    ax1.legend()

    agent = "DQN" if "DQN" in methods else next((m for m in methods if m not in ("SABRE", "greedy")), None)
    if agent:
        sab = {(r["topology"], r["circuit"]): r["swaps"] for r in rows if r["method"] == "SABRE"}
        pts = [(sab[(r["topology"], r["circuit"])], r["swaps"]) for r in rows
               if r["method"] == agent and (r["topology"], r["circuit"]) in sab]
        x, y = np.array(pts).T
        jitter = np.random.default_rng(0).uniform(-0.2, 0.2, (2, len(x)))
        ax2.scatter(x + jitter[0], y + jitter[1], s=10, alpha=0.5)
        lim = max(x.max(), y.max()) + 1
        ax2.plot([0, lim], [0, lim], "k--", lw=1, label="equal to SABRE")
        ax2.set_xlabel("SABRE SWAPs"); ax2.set_ylabel(f"{agent} SWAPs")
        ax2.set_title(f"Per circuit: below the line = {agent} better\n"
                      f"{(y < x).sum()} better / {(y == x).sum()} equal / {(y > x).sum()} worse")
        ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=130)
    plt.close(fig)
    log.info("saved %s", out_dir / "comparison.png")
