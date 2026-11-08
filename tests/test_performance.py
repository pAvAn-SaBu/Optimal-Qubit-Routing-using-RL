import numpy as np

from routing.evaluate import Method
from routing.performance import headline, ratio_ci, scaling_gates, success_probability, summarise_decisions, \
    analyse_decisions
from routing.policies import greedy_policy


def test_ratio_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    agent = np.array([8, 9, 10, 7, 12])
    ref = np.array([10, 10, 11, 9, 12])
    point, lo, hi = ratio_ci(agent, ref, rng)
    assert point == agent.sum() / ref.sum()
    assert lo <= point <= hi


def test_success_probability_drops_with_swaps():
    p = success_probability(30, np.array([0, 10]), 0.01)
    assert p[0] == (0.99 ** 30) and p[1] < p[0]


def test_headline_counts_wins():
    rows = [{"topology": "t", "seen_in_training": True, "circuit": i, "method": m, "swaps": s}
            for i, (a, b) in enumerate([(5, 6), (6, 6), (7, 6)]) for m, s in (("SABRE", b), ("X", a))]
    (h,) = [r for r in headline(rows) if r["subset"] == "all chips"]
    assert h["win/tie/loss"] == "1/1/1" and h["ratio_vs_sabre"] == 1.0


def test_scaling_and_decisions_run_and_are_valid():
    rows = scaling_gates([Method("SABRE", None), Method("greedy", greedy_policy)], ["line:5"], [8], 2, seed=0)
    assert {r["method"] for r in rows} == {"SABRE", "greedy"}
    rec = analyse_decisions("greedy", greedy_policy, ["ring:6"], 2, 12, seed=0)
    (s,) = summarise_decisions(rec)
    assert s["non_closest_pct"] == 0.0  # greedy always serves the closest gate
