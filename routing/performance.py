"""Full performance evaluation of the trained agents (one command, one report).

Sections (all written to the output directory, default runs/performance):

  1. Training analysis    training_dqn.png, training_tabular.png
                          learning curve, return, TD loss, validation SWAPs vs
                          SABRE/greedy, ratio to SABRE, exploration schedule
  2. Benchmark            benchmark/ (results.csv, summary.md, comparison.png via evaluate.py)
                          ratio_vs_sabre.png   SWAP ratio to SABRE with 95% bootstrap CIs
                          swaps_distribution.png  per-circuit box plots
                          depth.png            CX-depth (SWAP = 3 CX)
                          win_tie_loss.png     per-circuit outcome vs SABRE
                          improvement_hist.png per-circuit % SWAPs saved vs SABRE
                          runtime.png          wall-clock time per circuit
                          fidelity.png         estimated success probability, (1 − p)^#CX
  3. Generalisation       scaling_gates.png/.csv   longer circuits than seen in training
                          scaling_chip.png/.csv    chip sizes beyond the training chips
  4. Decision analysis    decisions.png/.csv  how often the DQN picks a gate that is
                          NOT the closest one (non-myopic choices) and which modes it uses
  5. report.md + headline.json  every table and figure in one place

Every routing in every section is checked by the independent verifier; an
invalid routing raises instead of being silently counted.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .circuit import random_circuit
from .evaluate import Method, run_benchmark
from .policies import greedy_policy, route, route_sabre
from .simulator import MODE_LABELS, Mode
from .topology import Topology, from_spec

log = logging.getLogger(__name__)

COLORS = {"SABRE": "tab:red", "greedy": "tab:gray", "DQN": "tab:blue", "tabular-Q": "tab:green"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _color(method: str) -> str | None:
    return COLORS.get(method)


# ------------------------------------------------------------------ csv helpers
def _parse(v: str):
    if v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except ValueError:
        return v


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ statistics
def smooth(y: np.ndarray, win: int) -> np.ndarray:
    win = max(1, min(win, len(y)))
    return np.convolve(y, np.ones(win) / win, mode="valid")


def ratio_ci(agent: np.ndarray, ref: np.ndarray, rng: np.random.Generator, n_boot: int = 2000,
             ) -> tuple[float, float, float]:
    """sum(agent) / sum(ref) and its 95% bootstrap interval (circuits resampled in pairs)."""
    agent, ref = np.asarray(agent, float), np.asarray(ref, float)
    point = agent.sum() / max(ref.sum(), 1.0)
    idx = rng.integers(len(agent), size=(n_boot, len(agent)))
    boots = agent[idx].sum(1) / np.maximum(ref[idx].sum(1), 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def success_probability(n_gates: int, swaps: np.ndarray, cx_error: float) -> np.ndarray:
    """Crude fidelity estimate: every CX fails independently with probability p; a SWAP is 3 CX."""
    return (1.0 - cx_error) ** (n_gates + 3 * np.asarray(swaps, float))


def _paired(rows: list[dict]) -> dict[tuple[str, str], dict[int, dict]]:
    by: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by[(r["topology"], r["method"])][r["circuit"]] = r
    return by


def _ordered(rows: list[dict]) -> tuple[list[str], list[str], set[str]]:
    topos = list(dict.fromkeys(r["topology"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    unseen = {r["topology"] for r in rows if not r["seen_in_training"]}
    return topos, methods, unseen


def _route(topo: Topology, circ, m: Method, seed: int):
    res = route_sabre(topo, circ, seed=seed) if m.policy is None else route(topo, circ, m.policy, m.name)
    if not res.valid:
        raise AssertionError(f"{m.name} produced an invalid routing on {topo.name}: {res.errors[:3]}")
    return res


# ============================================================ 1. training curves
def plot_training(run_dir: Path, out_path: Path, title: str) -> dict | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        log.warning("no %s; skipping training analysis for %s", path, title)
        return None
    rows = read_csv(path)
    train = [r for r in rows if r.get("episode_swaps") is not None]
    val = [r for r in rows if r.get("val_swaps") is not None]
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.ravel()
    win = max(1, len(train) // 100)

    ep = np.array([r["episode"] for r in train])
    sw = np.array([r["episode_swaps"] for r in train], float)
    ax = axes[0]
    ax.plot(ep, sw, alpha=0.12, color="tab:blue")
    ax.plot(ep[win - 1:], smooth(sw, win), color="tab:blue", label=f"moving avg ({win} eps)")
    ax.set(title="SWAPs per training episode (with exploration)", xlabel="episode", ylabel="SWAPs")
    ax.legend()

    ax = axes[1]
    ret = [(r["episode"], r["return"]) for r in train if r.get("return") is not None]
    if ret:
        e, y = np.array(ret).T
        ax.plot(e, y, alpha=0.12, color="tab:purple")
        ax.plot(e[win - 1:], smooth(y, win), color="tab:purple", label=f"moving avg ({win} eps)")
        ax.set(title="Episode return (reward sum)", xlabel="episode", ylabel="return")
        ax.legend()
    else:
        states = [(r["episode"], r["table_states"]) for r in val if r.get("table_states") is not None]
        if states:
            e, y = np.array(states).T
            ax.plot(e, y, marker="o", color="tab:purple")
            ax.set(title="Q-table size (distinct states visited)", xlabel="episode", ylabel="states")

    ax = axes[2]
    loss = [(r["episode"], r["loss"]) for r in train if r.get("loss") is not None]
    if loss:
        e, y = np.array(loss).T
        ax.plot(e, y, alpha=0.15, color="tab:orange")
        ax.plot(e[win - 1:], smooth(y, win), color="tab:orange", label=f"moving avg ({win} eps)")
        ax.set_yscale("log")
        ax.set(title="TD loss (Huber, mean per episode)", xlabel="episode", ylabel="loss (log)")
        ax.legend()
    else:
        fb = [(r["episode"], 100 * r["fallback_rate"]) for r in val if r.get("fallback_rate") is not None]
        if fb:
            e, y = np.array(fb).T
            ax.plot(e, y, marker="o", color="tab:orange")
            ax.set(title="Fallback to greedy on validation (unseen states)", xlabel="episode", ylabel="% decisions")

    ax = axes[3]
    if val:
        e = [r["episode"] for r in val]
        ax.plot(e, [r["val_swaps"] for r in val], marker="o", ms=3, color="tab:blue", label="agent (ε = 0)")
        ax.axhline(val[-1]["sabre_swaps"], color="tab:red", ls="--", label="Qiskit SABRE")
        if any(r.get("greedy_swaps") is not None for r in val):
            g = next(r["greedy_swaps"] for r in val if r.get("greedy_swaps") is not None)
            ax.axhline(g, color="tab:gray", ls=":", label="greedy")
        best = min(val, key=lambda r: r["val_swaps"])
        ax.scatter([best["episode"]], [best["val_swaps"]], s=120, facecolors="none", edgecolors="k",
                   zorder=5, label=f"best checkpoint (ep {best['episode']})")
        ax.set(title="Validation: mean SWAPs per circuit", xlabel="episode", ylabel="SWAPs")
        ax.legend(fontsize=8)

    ax = axes[4]
    if val:
        e = [r["episode"] for r in val]
        ratio = [r["ratio_vs_sabre"] for r in val]
        ax.plot(e, ratio, marker="o", ms=3, color="tab:green")
        ax.axhline(1.0, color="tab:red", ls="--", label="SABRE = 1.0")
        ax.fill_between(e, ratio, 1.0, where=np.array(ratio) < 1.0, color="tab:green", alpha=0.15,
                        label="better than SABRE")
        ax.set(title="Validation SWAPs / SABRE SWAPs (lower is better)", xlabel="episode", ylabel="ratio")
        ax.legend(fontsize=8)

    ax = axes[5]
    eps = [(r["episode"], r["epsilon"]) for r in train if r.get("epsilon") is not None]
    if eps:
        e, y = np.array(eps).T
        ax.plot(e, y, color="k")
        ax.set(title="Exploration schedule ε", xlabel="episode", ylabel="ε")
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            if "alpha" in cfg:
                frac = np.clip(e / max(cfg["episodes"], 1), 0, 1)
                ax2 = ax.twinx()
                ax2.plot(e, cfg["alpha"] + (cfg["alpha_end"] - cfg["alpha"]) * frac, color="tab:brown", ls="--")
                ax2.set_ylabel("learning rate α", color="tab:brown")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("saved %s", out_path)

    if not val:
        return None
    best = min(val, key=lambda r: r["val_swaps"])
    return {"episodes": int(ep.max()) if len(ep) else 0, "best_episode": best["episode"],
            "best_val_swaps": round(best["val_swaps"], 2), "best_ratio_vs_sabre": round(best["ratio_vs_sabre"], 3),
            "final_val_swaps": round(val[-1]["val_swaps"], 2),
            "final_ratio_vs_sabre": round(val[-1]["ratio_vs_sabre"], 3),
            "sabre_val_swaps": round(val[-1]["sabre_swaps"], 2)}


# ============================================================ 2. benchmark plots
def ratio_table(rows: list[dict], seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    by = _paired(rows)
    topos, methods, unseen = _ordered(rows)
    out = []
    for t in topos:
        sab = by.get((t, "SABRE"), {})
        for m in methods:
            if m == "SABRE" or (t, m) not in by:
                continue
            ids = sorted(by[(t, m)])
            a = np.array([by[(t, m)][i]["swaps"] for i in ids])
            r = np.array([sab[i]["swaps"] for i in ids])
            point, lo, hi = ratio_ci(a, r, rng)
            out.append({"topology": t, "method": m, "seen": t not in unseen, "ratio": round(point, 3),
                        "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                        "mean_saved_pct": round(float(np.mean((r - a) / np.maximum(r, 1))) * 100, 1)})
    return out


def plot_ratio(table: list[dict], out_path: Path) -> None:
    plt = _plt()
    topos = list(dict.fromkeys(r["topology"] for r in table))
    methods = list(dict.fromkeys(r["method"] for r in table))
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(topos)), 5))
    unseen_idx = [i for i, t in enumerate(topos) if not next(r["seen"] for r in table if r["topology"] == t)]
    if unseen_idx:
        ax.axvspan(min(unseen_idx) - 0.5, max(unseen_idx) + 0.5, color="gold", alpha=0.12,
                   label="chips never seen in training")
    width = 0.7 / max(len(methods), 1)
    for j, m in enumerate(methods):
        pts = [(topos.index(r["topology"]), r) for r in table if r["method"] == m]
        x = np.array([i for i, _ in pts]) + (j - (len(methods) - 1) / 2) * width
        y = np.array([r["ratio"] for _, r in pts])
        err = np.array([[r["ratio"] - r["ci_low"] for _, r in pts], [r["ci_high"] - r["ratio"] for _, r in pts]])
        ax.errorbar(x, y, yerr=err, fmt="o", capsize=4, color=_color(m), label=m)
    ax.axhline(1.0, color="tab:red", ls="--", label="SABRE (= 1.0)")
    ax.set_xticks(range(len(topos)), topos, rotation=30)
    ax.set_ylabel("total SWAPs / SABRE total SWAPs")
    ax.set_title("SWAP ratio to Qiskit SABRE per chip (95% bootstrap CI). Below 1.0 = fewer SWAPs than SABRE")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_distributions(rows: list[dict], out_path: Path) -> None:
    plt = _plt()
    by = _paired(rows)
    topos, methods, unseen = _ordered(rows)
    cols = 4
    nrows = int(np.ceil(len(topos) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.4 * nrows), squeeze=False)
    for ax, t in zip(axes.ravel(), topos):
        ms = [m for m in methods if (t, m) in by]
        data = [[r["swaps"] for r in by[(t, m)].values()] for m in ms]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanprops={"marker": "^", "markerfacecolor": "w"})
        for patch, m in zip(bp["boxes"], ms):
            patch.set_facecolor(_color(m) or "white")
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(ms) + 1), ms, fontsize=8)
        ax.set_title(t + (" (unseen)" if t in unseen else ""), fontsize=10)
        ax.set_ylabel("SWAPs")
    for ax in axes.ravel()[len(topos):]:
        ax.axis("off")
    fig.suptitle("SWAPs per circuit (box = quartiles, ▲ = mean)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _grouped_bars(ax, topos, methods, value) -> None:
    width = 0.8 / len(methods)
    for j, m in enumerate(methods):
        xs, ys = [], []
        for i, t in enumerate(topos):
            v = value(t, m)
            if v is not None:
                xs.append(i + (j - (len(methods) - 1) / 2) * width)
                ys.append(v)
        ax.bar(xs, ys, width, label=m, color=_color(m))
    ax.set_xticks(range(len(topos)), topos, rotation=30)


def plot_depth(rows: list[dict], out_path: Path) -> None:
    plt = _plt()
    by = _paired(rows)
    topos, methods, _ = _ordered(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    def mean_depth(t, m):
        return float(np.mean([r["depth"] for r in by[(t, m)].values()])) if (t, m) in by else None

    def depth_ratio(t, m):
        if (t, m) not in by or (t, "SABRE") not in by:
            return None
        ids = sorted(by[(t, m)])
        return sum(by[(t, m)][i]["depth"] for i in ids) / max(sum(by[(t, "SABRE")][i]["depth"] for i in ids), 1)

    _grouped_bars(ax1, topos, methods, mean_depth)
    ax1.set(title="Mean CX-depth of routed circuit (SWAP = 3 CX), lower is better", ylabel="CX-depth")
    ax1.legend(fontsize=8)
    _grouped_bars(ax2, topos, [m for m in methods if m != "SABRE"], depth_ratio)
    ax2.axhline(1.0, color="tab:red", ls="--", label="SABRE")
    ax2.set(title="CX-depth relative to SABRE", ylabel="depth / SABRE depth")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_win_tie_loss(rows: list[dict], out_path: Path) -> None:
    plt = _plt()
    by = _paired(rows)
    topos, methods, _ = _ordered(rows)
    others = [m for m in methods if m != "SABRE"]
    fig, axes = plt.subplots(1, len(others), figsize=(5.5 * len(others), 0.45 * len(topos) + 2), squeeze=False)
    for ax, m in zip(axes[0], others):
        ts = [t for t in topos if (t, m) in by and (t, "SABRE") in by]
        w, ti, lo = [], [], []
        for t in ts:
            ids = sorted(by[(t, m)])
            a = np.array([by[(t, m)][i]["swaps"] for i in ids])
            r = np.array([by[(t, "SABRE")][i]["swaps"] for i in ids])
            n = len(ids)
            w.append(100 * (a < r).sum() / n)
            ti.append(100 * (a == r).sum() / n)
            lo.append(100 * (a > r).sum() / n)
        y = np.arange(len(ts))
        ax.barh(y, w, color="tab:green", label="fewer SWAPs (win)")
        ax.barh(y, ti, left=w, color="lightgray", label="equal (tie)")
        ax.barh(y, lo, left=np.add(w, ti), color="tab:red", alpha=0.7, label="more SWAPs (loss)")
        ax.set_yticks(y, ts)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set(title=f"{m} vs SABRE, per circuit", xlabel="% of circuits")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_improvement_hist(rows: list[dict], out_path: Path) -> None:
    plt = _plt()
    by = _paired(rows)
    topos, methods, _ = _ordered(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-80, 80, 33)
    for m in methods:
        if m == "SABRE":
            continue
        vals = []
        for t in topos:
            if (t, m) in by and (t, "SABRE") in by:
                for i, r in by[(t, m)].items():
                    ref = by[(t, "SABRE")][i]["swaps"]
                    vals.append(100 * (ref - r["swaps"]) / max(ref, 1))
        vals = np.clip(vals, bins[0], bins[-1])
        ax.hist(vals, bins=bins, alpha=0.45, color=_color(m), label=f"{m} (median {np.median(vals):+.1f}%)")
        ax.axvline(np.median(vals), color=_color(m), lw=2)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set(title="Per-circuit SWAPs saved vs SABRE (right of 0 = better than SABRE)",
           xlabel="% SWAPs saved relative to SABRE", ylabel="circuits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_runtime(rows: list[dict], out_path: Path) -> None:
    plt = _plt()
    by = _paired(rows)
    topos, methods, _ = _ordered(rows)
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(topos)), 5))
    _grouped_bars(ax, topos, methods, lambda t, m: 1000 * float(np.mean([r["seconds"] for r in by[(t, m)].values()]))
                  if (t, m) in by else None)
    ax.set_yscale("log")
    ax.set(title="Wall-clock routing time per circuit (SABRE includes Qiskit transpile overhead)",
           ylabel="milliseconds (log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_fidelity(rows: list[dict], n_gates: int, cx_error: float, out_path: Path) -> list[dict]:
    plt = _plt()
    by = _paired(rows)
    topos, methods, _ = _ordered(rows)
    table = []
    for t in topos:
        for m in methods:
            if (t, m) in by:
                sw = np.array([r["swaps"] for r in by[(t, m)].values()])
                table.append({"topology": t, "method": m,
                              "success_prob": round(float(success_probability(n_gates, sw, cx_error).mean()), 4)})
    lookup = {(r["topology"], r["method"]): r["success_prob"] for r in table}
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(topos)), 5))
    _grouped_bars(ax, topos, methods, lambda t, m: lookup.get((t, m)))
    ax.set(title=f"Estimated circuit success probability, (1 − p)^#CX with p = {cx_error} per CX "
                 f"({n_gates} gates + 3 CX per SWAP). Higher is better", ylabel="success probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return table


# ============================================================ 3. generalisation
def scaling_gates(methods: list[Method], specs: list[str], gate_counts: list[int], circuits: int,
                  seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for spec in specs:
        topo = from_spec(spec)
        for n_g in gate_counts:
            for i in range(circuits):
                circ = random_circuit(topo.n, n_g, rng)
                for m in methods:
                    if m.topologies is not None and topo.name not in m.topologies:
                        continue
                    res = _route(topo, circ, m, i)
                    rows.append({"topology": topo.name, "gates": n_g, "circuit": i, "method": m.name,
                                 "swaps": res.swaps, "depth": res.depth})
        log.info("scaling (circuit length): finished %s", topo.name)
    return rows


def scaling_chip(methods: list[Method], families: dict[str, list[str]], gates_per_qubit: int, circuits: int,
                 seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for fam, specs in families.items():
        for spec in specs:
            topo = from_spec(spec)
            n_g = gates_per_qubit * topo.n
            for i in range(circuits):
                circ = random_circuit(topo.n, n_g, rng)
                for m in methods:
                    if m.topologies is not None and topo.name not in m.topologies:
                        continue
                    res = _route(topo, circ, m, i)
                    rows.append({"family": fam, "topology": topo.name, "n_qubits": topo.n, "gates": n_g,
                                 "circuit": i, "method": m.name, "swaps": res.swaps, "depth": res.depth})
        log.info("scaling (chip size): finished %s", fam)
    return rows


def _ratio_by(rows: list[dict], key: str, group: str) -> dict:
    """{group_value: {method: [(x, ratio)]}} with ratio = sum swaps / sum SABRE swaps at each x."""
    tot: dict = defaultdict(float)
    for r in rows:
        tot[(r[group], r[key], r["method"])] += r["swaps"]
    out: dict = defaultdict(lambda: defaultdict(list))
    for (g, x, m), s in sorted(tot.items(), key=lambda kv: kv[0][1]):
        if m != "SABRE":
            out[g][m].append((x, s / max(tot[(g, x, "SABRE")], 1)))
    return out


def plot_scaling_gates(rows: list[dict], train_range: tuple[int, int] | None, out_path: Path) -> None:
    plt = _plt()
    topos = list(dict.fromkeys(r["topology"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(2, len(topos), figsize=(5 * len(topos), 8), squeeze=False)
    ratios = _ratio_by(rows, "gates", "topology")
    for j, t in enumerate(topos):
        ax = axes[0, j]
        for m in methods:
            pts = defaultdict(list)
            for r in rows:
                if r["topology"] == t and r["method"] == m:
                    pts[r["gates"]].append(r["swaps"])
            x = sorted(pts)
            mu = np.array([np.mean(pts[g]) for g in x])
            sd = np.array([np.std(pts[g]) for g in x])
            ax.plot(x, mu, marker="o", color=_color(m), label=m)
            ax.fill_between(x, mu - sd, mu + sd, color=_color(m), alpha=0.12)
        ax.set(title=f"{t}: mean SWAPs (± 1 std)", xlabel="two-qubit gates in circuit", ylabel="SWAPs")
        ax.legend(fontsize=8)
        ax = axes[1, j]
        for m, pts in ratios[t].items():
            x, y = zip(*pts)
            ax.plot(x, y, marker="o", color=_color(m), label=m)
        ax.axhline(1.0, color="tab:red", ls="--", label="SABRE")
        if train_range:
            ax.axvspan(*train_range, color="tab:blue", alpha=0.08, label="training circuit lengths")
        ax.set(title=f"{t}: SWAPs / SABRE", xlabel="two-qubit gates in circuit", ylabel="ratio")
        ax.legend(fontsize=8)
    fig.suptitle("Generalisation to longer circuits", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_scaling_chip(rows: list[dict], train_names: set[str], out_path: Path) -> None:
    plt = _plt()
    fams = list(dict.fromkeys(r["family"] for r in rows))
    fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4.5), squeeze=False)
    names = {(r["family"], r["n_qubits"]): r["topology"] for r in rows}
    ratios = _ratio_by(rows, "n_qubits", "family")
    for ax, fam in zip(axes[0], fams):
        for m, pts in ratios[fam].items():
            x, y = zip(*pts)
            ax.plot(x, y, color=_color(m), label=m)
            seen = [names[(fam, n)] in train_names for n in x]
            ax.scatter([a for a, s in zip(x, seen) if s], [b for b, s in zip(y, seen) if s], color=_color(m), zorder=3)
            ax.scatter([a for a, s in zip(x, seen) if not s], [b for b, s in zip(y, seen) if not s],
                       facecolors="white", edgecolors=_color(m), zorder=3)
        ax.axhline(1.0, color="tab:red", ls="--", label="SABRE")
        ax.set(title=f"{fam} chips", xlabel="qubits on chip", ylabel="SWAPs / SABRE SWAPs")
        ax.legend(fontsize=8)
    fig.suptitle("Generalisation to chip size (filled = chip used in DQN training, hollow = unseen)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================ 4. decision analysis
def analyse_decisions(name: str, policy, specs: list[str], circuits: int, gates: int, seed: int) -> list[dict]:
    """One row per agent decision: was the chosen gate the closest one? which mode? greedy agreement?"""
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for spec in specs:
        topo = from_spec(spec)

        def wrapped(state, _topo=topo):
            g, mode = policy(state)
            front = state.front()
            dists = {x: state.phys_distance(x) for x in front}
            gg, _ = greedy_policy(state)
            records.append({"topology": _topo.name, "n_waiting": len(front), "slot": front.index(g),
                            "distance": dists[g], "min_distance": min(dists.values()), "mode": int(mode),
                            "closest": dists[g] == min(dists.values()), "same_gate_as_greedy": g == gg})
            return g, mode

        for _ in range(circuits):
            res = route(topo, random_circuit(topo.n, gates, rng), wrapped, name)
            if not res.valid:
                raise AssertionError(f"{name} invalid on {topo.name}: {res.errors[:3]}")
    return records


def summarise_decisions(records: list[dict]) -> list[dict]:
    out = []
    for t in dict.fromkeys(r["topology"] for r in records):
        rs = [r for r in records if r["topology"] == t]
        choice = [r for r in rs if r["n_waiting"] > 1]
        modes = Counter(r["mode"] for r in rs if r["distance"] > 2)  # d = 2: A→B and MEET are identical
        n_modes = max(sum(modes.values()), 1)
        out.append({
            "topology": t, "decisions": len(rs), "with_choice": len(choice),
            "non_closest_pct": round(100 * np.mean([not r["closest"] for r in choice]), 1) if choice else 0.0,
            "diff_gate_from_greedy_pct": round(100 * np.mean([not r["same_gate_as_greedy"] for r in choice]), 1)
            if choice else 0.0,
            **{f"mode_{MODE_LABELS[m]}_pct": round(100 * modes[int(m)] / n_modes, 1) for m in Mode},
        })
    return out


def plot_decisions(records: list[dict], summary: list[dict], name: str, out_path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ts = [s["topology"] for s in summary]

    ax = axes[0]
    ax.bar(ts, [s["non_closest_pct"] for s in summary], color="tab:blue", label="picked a farther gate")
    ax.bar(ts, [s["diff_gate_from_greedy_pct"] for s in summary], color="none", edgecolor="k", ls="--",
           label="different gate than greedy (incl. equal-distance ties)")
    ax.set(title=f"{name}: non-myopic decisions\n(only decisions with ≥ 2 waiting gates)", ylabel="% decisions")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

    ax = axes[1]
    buckets = ["2", "3", "4", "5+"]
    bucket = lambda d: str(d) if d < 5 else "5+"  # noqa: E731
    counts = {b: Counter() for b in buckets}
    for r in records:
        if r["distance"] >= 2:
            counts[bucket(r["distance"])][r["mode"]] += 1
    bottom = np.zeros(len(buckets))
    for m, c in zip(Mode, ["tab:orange", "tab:purple", "tab:cyan"]):
        frac = np.array([100 * counts[b][int(m)] / max(sum(counts[b].values()), 1) for b in buckets])
        ax.bar(buckets, frac, bottom=bottom, color=c, label=MODE_LABELS[m])
        bottom += frac
    ax.set(title="Mode chosen vs distance of the chosen gate\n(d = 2: A→B and meet are the same move)",
           xlabel="distance d between the gate's qubits", ylabel="% decisions")
    ax.legend(fontsize=8)

    ax = axes[2]
    slots = Counter(r["slot"] for r in records if r["n_waiting"] > 1)
    xs = sorted(slots)
    ax.bar([f"slot {s}" for s in xs], [100 * slots[s] / max(sum(slots.values()), 1) for s in xs], color="tab:green")
    ax.set(title="Which waiting gate is served (0 = earliest in circuit order)", ylabel="% decisions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================================================ 5. report
def _md_table(rows: list[dict], keys: list[str] | None = None) -> str:
    if not rows:
        return "_(no data)_"
    keys = keys or list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def headline(rows: list[dict], seed: int = 0) -> list[dict]:
    """Overall ratio to SABRE per method, split into chips seen / unseen in training."""
    rng = np.random.default_rng(seed)
    by = _paired(rows)
    _, methods, unseen = _ordered(rows)
    out = []
    for m in methods:
        if m == "SABRE":
            continue
        for label, keep in (("seen chips", lambda t: t not in unseen), ("unseen chips", lambda t: t in unseen),
                            ("all chips", lambda t: True)):
            a, r = [], []
            for (t, mm), res in by.items():
                if mm == m and keep(t) and (t, "SABRE") in by:
                    for i, row in res.items():
                        a.append(row["swaps"])
                        r.append(by[(t, "SABRE")][i]["swaps"])
            if not a:
                continue
            a, r = np.array(a), np.array(r)
            point, lo, hi = ratio_ci(a, r, rng)
            out.append({"method": m, "subset": label, "circuits": len(a), "ratio_vs_sabre": round(point, 3),
                        "95% CI": f"[{lo:.3f}, {hi:.3f}]", "swaps_saved_pct": round(100 * (1 - point), 1),
                        "win/tie/loss": f"{(a < r).sum()}/{(a == r).sum()}/{(a > r).sum()}"})
    return out


def write_report(out_dir: Path, sections: dict) -> None:
    p = []
    p.append("# Performance evaluation: RL qubit routing\n")
    p.append("All methods route the same random circuits on the same chip from the same trivial initial "
             "layout. Every routed circuit passed the independent verifier (chip edges + program order).\n")
    if sections.get("headline"):
        p.append("## Headline: SWAPs relative to Qiskit SABRE\n")
        p.append("`ratio_vs_sabre` = total SWAPs / SABRE total SWAPs (below 1 = better). The CI is a 95% "
                 "paired bootstrap over circuits.\n")
        p.append(_md_table(sections["headline"]) + "\n")
    p.append("## 1. Training\n")
    for key, title in (("training_dqn", "Double DQN"), ("training_tabular", "Tabular Q-learning")):
        if sections.get(key):
            p.append(f"### {title}\n\n![{title}]({key}.png)\n")
            p.append(_md_table([sections[key]]) + "\n")
    if sections.get("benchmark_summary"):
        p.append("## 2. Benchmark on held-out circuits\n")
        p.append(f"Test set: {sections['circuits']} circuits × {sections['gates']} CX gates per chip, "
                 f"seed {sections['seed']} (different from training and validation).\n")
        p.append(_md_table(sections["benchmark_summary"]) + "\n")
        p.append("### Ratio to SABRE with confidence intervals\n\n![ratio](ratio_vs_sabre.png)\n")
        p.append(_md_table(sections["ratio_table"]) + "\n")
        for f, t in (("benchmark/comparison.png", "Mean SWAPs and per-circuit scatter"),
                     ("swaps_distribution.png", "Per-circuit distribution"),
                     ("win_tie_loss.png", "Win / tie / loss vs SABRE"),
                     ("improvement_hist.png", "Per-circuit improvement"),
                     ("depth.png", "Circuit depth"), ("runtime.png", "Runtime")):
            p.append(f"### {t}\n\n![{t}]({f})\n")
        p.append(f"### Estimated fidelity (p = {sections['cx_error']} per CX)\n\n![fidelity](fidelity.png)\n")
        p.append("Fewer SWAPs means fewer noisy CX gates. This is a simple uniform-error model meant to show "
                 "the practical effect, not a hardware-calibrated simulation.\n")
    if sections.get("scaling_gates"):
        p.append("## 3. Generalisation\n\n### Longer circuits\n\n![scaling gates](scaling_gates.png)\n")
    if sections.get("scaling_chip"):
        p.append("### Larger / different chips\n\n![scaling chip](scaling_chip.png)\n")
    if sections.get("decisions"):
        p.append("## 4. What the agent learned (decision analysis)\n")
        p.append("A greedy router always serves the closest waiting gate. `non_closest_pct` is how often the "
                 "agent deliberately serves a farther one, paying more SWAPs now to save more later. "
                 "Mode shares only count gates with d > 2, where the three modes really differ.\n")
        p.append("![decisions](decisions.png)\n")
        p.append(_md_table(sections["decisions"]) + "\n")
    (out_dir / "report.md").write_text("\n".join(p))
    log.info("saved %s", out_dir / "report.md")


# ============================================================ driver
def run_performance(out_dir: Path, methods: list[Method], benchmark_specs: list[str], unseen: set[str],
                    dqn_run: Path | None, tabular_run: Path | None, dqn_policy=None, dqn_train_specs=None,
                    dqn_train_gates: tuple[int, int] | None = None, circuits: int = 30, gates: int = 30,
                    seed: int = 12345, cx_error: float = 0.01, scaling: bool = True, quick: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sections: dict = {"circuits": circuits, "gates": gates, "seed": seed, "cx_error": cx_error}

    log.info("== 1/4 training analysis")
    if dqn_run:
        sections["training_dqn"] = plot_training(dqn_run, out_dir / "training_dqn.png", "Double DQN training")
    if tabular_run:
        sections["training_tabular"] = plot_training(tabular_run, out_dir / "training_tabular.png",
                                                     "Tabular Q-learning training")

    log.info("== 2/4 benchmark (%d chips × %d circuits)", len(benchmark_specs), circuits)
    topos = [from_spec(s) for s in benchmark_specs]
    sections["benchmark_summary"] = run_benchmark(topos, methods, circuits, gates, seed, out_dir / "benchmark", unseen)
    rows = read_csv(out_dir / "benchmark" / "results.csv")
    sections["ratio_table"] = ratio_table(rows)
    write_csv(sections["ratio_table"], out_dir / "ratio_vs_sabre.csv")
    sections["headline"] = headline(rows)
    plot_ratio(sections["ratio_table"], out_dir / "ratio_vs_sabre.png")
    plot_distributions(rows, out_dir / "swaps_distribution.png")
    plot_depth(rows, out_dir / "depth.png")
    plot_win_tie_loss(rows, out_dir / "win_tie_loss.png")
    plot_improvement_hist(rows, out_dir / "improvement_hist.png")
    plot_runtime(rows, out_dir / "runtime.png")
    write_csv(plot_fidelity(rows, gates, cx_error, out_dir / "fidelity.png"), out_dir / "fidelity.csv")

    n_scale = 5 if quick else 15
    if scaling:
        log.info("== 3/4 generalisation")
        general = [m for m in methods if m.topologies is None]  # the tabular agent only knows one chip
        gate_counts = [10, 20, 40, 60] if quick else [10, 20, 30, 40, 60, 80]
        sg = scaling_gates(general, ["line:7", "ring:8", "grid:3x3"], gate_counts, n_scale, seed + 1)
        write_csv(sg, out_dir / "scaling_gates.csv")
        plot_scaling_gates(sg, dqn_train_gates, out_dir / "scaling_gates.png")
        sections["scaling_gates"] = True
        families = {"line": ["line:4", "line:5", "line:7", "line:10", "line:13", "line:16"],
                    "ring": ["ring:6", "ring:8", "ring:10", "ring:13", "ring:16"],
                    "grid": ["grid:2x2", "grid:2x3", "grid:3x3", "grid:3x4", "grid:4x4", "grid:4x5"],
                    "star": ["star:6", "star:8", "star:10", "star:13", "star:16"]}
        if quick:
            families = {k: v[:4] for k, v in families.items()}
        sc = scaling_chip(general, families, 3, n_scale, seed + 2)
        write_csv(sc, out_dir / "scaling_chip.csv")
        plot_scaling_chip(sc, {from_spec(s).name for s in (dqn_train_specs or [])}, out_dir / "scaling_chip.png")
        sections["scaling_chip"] = True

    if dqn_policy is not None:
        log.info("== 4/4 decision analysis")
        rec = analyse_decisions("DQN", dqn_policy, benchmark_specs, 5 if quick else 10, gates, seed + 3)
        sections["decisions"] = summarise_decisions(rec)
        write_csv(sections["decisions"], out_dir / "decisions.csv")
        plot_decisions(rec, sections["decisions"], "DQN", out_dir / "decisions.png")

    write_report(out_dir, sections)
    (out_dir / "headline.json").write_text(json.dumps(
        {"headline": sections["headline"], "training_dqn": sections.get("training_dqn"),
         "training_tabular": sections.get("training_tabular")}, indent=2, default=str))
    print("\n" + _md_table(sections["headline"]) + "\n")
    return sections
