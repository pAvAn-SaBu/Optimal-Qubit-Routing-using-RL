"""Training utilities shared by the tabular and DQN agents."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .circuit import Circuit, random_circuit
from .policies import Policy, route, route_sabre
from .topology import Topology

log = logging.getLogger(__name__)


def linear_schedule(start: float, end: float, frac_done: float) -> float:
    return start + (end - start) * min(max(frac_done, 0.0), 1.0)


@dataclass
class ValidationSet:
    """Fixed circuits (fixed seed) used to pick the best checkpoint. SABRE is run once up front."""
    items: list[tuple[Topology, Circuit]]
    sabre_swaps: list[int]

    @classmethod
    def build(cls, topologies: list[Topology], per_topology: int, n_gates: int, seed: int) -> "ValidationSet":
        rng = np.random.default_rng(seed)
        items = [(t, random_circuit(t.n, n_gates, rng)) for t in topologies for _ in range(per_topology)]
        sabre = [route_sabre(t, c, seed=i).swaps for i, (t, c) in enumerate(items)]
        return cls(items, sabre)

    def evaluate(self, policy: Policy) -> dict:
        swaps = []
        for topo, circ in self.items:
            res = route(topo, circ, policy)
            if not res.valid:
                raise AssertionError(f"invalid routing during validation: {res.errors[:3]}")
            swaps.append(res.swaps)
        total, sabre_total = sum(swaps), sum(self.sabre_swaps)
        return {"val_swaps": total / len(swaps), "sabre_swaps": sabre_total / len(swaps),
                "ratio_vs_sabre": total / max(sabre_total, 1)}


class MetricsLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def log(self, row: dict) -> None:
        self.rows.append(row)
        keys = sorted({k for r in self.rows for k in r})
        with open(self.run_dir / "metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)

    def save_config(self, config: dict) -> None:
        with open(self.run_dir / "config.json", "w") as fh:
            json.dump(config, fh, indent=2, default=str)

    def plot(self, title: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train = [r for r in self.rows if "episode_swaps" in r]
        val = [r for r in self.rows if "val_swaps" in r]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        if train:
            ep = np.array([r["episode"] for r in train])
            sw = np.array([r["episode_swaps"] for r in train], dtype=float)
            win = max(1, len(sw) // 50)
            smooth = np.convolve(sw, np.ones(win) / win, mode="valid")
            ax1.plot(ep, sw, alpha=0.15, color="tab:blue", label="per episode")
            ax1.plot(ep[win - 1:], smooth, color="tab:blue", label=f"moving avg ({win})")
            ax1.set_xlabel("episode"); ax1.set_ylabel("SWAPs"); ax1.set_title("Training episodes (with exploration)")
            ax1.legend()
        if val:
            ep = [r["episode"] for r in val]
            ax2.plot(ep, [r["val_swaps"] for r in val], marker="o", label="agent (greedy w.r.t. Q)")
            ax2.axhline(val[0]["sabre_swaps"], color="tab:red", ls="--", label="Qiskit SABRE")
            if "greedy_swaps" in val[0]:
                ax2.axhline(val[0]["greedy_swaps"], color="gray", ls=":", label="greedy baseline")
            ax2.set_xlabel("episode"); ax2.set_ylabel("mean SWAPs per circuit"); ax2.set_title("Validation set")
            ax2.legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(self.run_dir / "training_curve.png", dpi=130)
        plt.close(fig)
        log.info("saved %s", self.run_dir / "training_curve.png")
