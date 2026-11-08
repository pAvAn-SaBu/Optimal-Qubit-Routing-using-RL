"""Command-line entry point. Run `python main.py -h` or `python main.py <command> -h`.

  train-tabular  Stage 1: tabular Q-learning on one small chip
  train-dqn      Stage 2: Double DQN on a mix of chips
  evaluate       benchmark agents vs greedy vs Qiskit SABRE (seen + unseen chips)
  visualize      animate the agent's routing next to SABRE's on one circuit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("routing")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("routing").setLevel(logging.DEBUG if verbose else logging.INFO)


def load_agent(path: str):
    """Return (method name, policy, agent) for a .pt (DQN) or .pkl (tabular) checkpoint."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"model file not found: {p} (train one first, see README 'How to run')")
    if p.suffix == ".pt":
        from routing.dqn import DQNAgent
        agent = DQNAgent.load(p)
        return "DQN", agent.policy, agent
    from routing.q_learning import TabularQAgent
    agent = TabularQAgent.load(p)
    return "tabular-Q", agent.policy, agent


def cmd_train_tabular(args) -> None:
    from routing.q_learning import TabularConfig, train_tabular
    from routing.topology import from_spec

    cfg = TabularConfig(topology=args.topology, episodes=args.episodes, k=args.k, lookahead=args.lookahead,
                        seed=args.seed)
    out = Path(args.out)
    train_tabular(cfg, out, from_spec(cfg.topology))
    log.info("done. best model: %s, curve: %s", out / "best.pkl", out / "training_curve.png")


def cmd_train_dqn(args) -> None:
    import torch
    from routing.dqn import DQNConfig, train_dqn

    torch.set_num_threads(args.threads)
    cfg = DQNConfig(episodes=args.episodes, seed=args.seed)
    if args.topologies:
        cfg.train_topologies = args.topologies
    out = Path(args.out)
    train_dqn(cfg, out)
    log.info("done. best model: %s, curve: %s", out / "best.pt", out / "training_curve.png")


def cmd_evaluate(args) -> None:
    from routing.evaluate import SEEN, UNSEEN, Method, run_benchmark
    from routing.policies import greedy_policy
    from routing.topology import from_spec

    specs = args.topologies or (SEEN + UNSEEN)
    methods = [Method("SABRE", None), Method("greedy", greedy_policy)]
    if args.dqn:
        name, policy, _ = load_agent(args.dqn)
        methods.append(Method(name, policy))
    if args.tabular:
        name, policy, agent = load_agent(args.tabular)
        tab_topo = from_spec(agent.cfg.topology)
        methods.append(Method(name, policy, {tab_topo.name}))
        if agent.cfg.topology not in specs:
            specs = [agent.cfg.topology] + specs
    if len(methods) == 2:
        log.warning("no --dqn or --tabular model given; comparing greedy vs SABRE only")
    topos = [from_spec(s) for s in specs]
    unseen = {from_spec(s).name for s in UNSEEN}
    run_benchmark(topos, methods, args.circuits, args.gates, args.seed, Path(args.out), unseen)
    if args.tabular:
        log.info("tabular agent fell back to greedy on %.1f%% of decisions (unseen states)",
                 100 * agent.fallback_rate)
    log.info("results in %s (summary.md, summary.csv, results.csv, comparison.png)", args.out)


def cmd_performance(args) -> None:
    from routing.evaluate import SEEN, UNSEEN, Method
    from routing.performance import run_performance
    from routing.policies import greedy_policy
    from routing.topology import from_spec

    specs = args.topologies or (SEEN + UNSEEN)
    methods = [Method("SABRE", None), Method("greedy", greedy_policy)]
    dqn_policy = dqn_run = tab_run = None
    dqn_specs, dqn_gates = SEEN, None
    if args.dqn and Path(args.dqn).exists():
        name, dqn_policy, agent = load_agent(args.dqn)
        methods.append(Method(name, dqn_policy))
        dqn_run = Path(args.dqn).parent
        dqn_specs, dqn_gates = agent.cfg.train_topologies, (agent.cfg.min_gates, agent.cfg.max_gates)
    elif args.dqn:
        log.warning("DQN checkpoint %s not found; skipping DQN", args.dqn)
    if args.tabular and Path(args.tabular).exists():
        name, policy, agent = load_agent(args.tabular)
        methods.append(Method(name, policy, {from_spec(agent.cfg.topology).name}))
        tab_run = Path(args.tabular).parent
        if agent.cfg.topology not in specs:
            specs = [agent.cfg.topology] + specs
    elif args.tabular:
        log.warning("tabular checkpoint %s not found; skipping tabular-Q", args.tabular)

    circuits = 10 if args.quick else args.circuits
    unseen = {from_spec(s).name for s in UNSEEN}
    run_performance(Path(args.out), methods, specs, unseen, dqn_run, tab_run, dqn_policy, dqn_specs, dqn_gates,
                    circuits, args.gates, args.seed, args.cx_error, scaling=not args.no_scaling, quick=args.quick)
    log.info("full report: %s", Path(args.out) / "report.md")


def cmd_visualize(args) -> None:
    from routing.circuit import Circuit, random_circuit
    from routing.policies import route, route_sabre
    from routing.topology import from_spec
    from routing.visualize import animate, narrate, summary_plot

    topo = from_spec(args.topology)
    if args.circuit:
        circ = Circuit.parse(args.circuit, args.qubits or topo.n)
    else:
        circ = random_circuit(args.qubits or topo.n, args.gates, np.random.default_rng(args.seed))
    name, policy, _ = load_agent(args.model)
    results = [route(topo, circ, policy, name), route_sabre(topo, circ, seed=args.seed)]
    if not results[0].decisions:
        log.warning("this circuit needs no SWAPs at all (all gates already adjacent); try another seed")

    print(narrate(circ, results[0]), flush=True)
    for r in results:
        print(f"{r.method:>10}: {r.swaps:3d} SWAPs | CX-depth {r.depth:3d} | valid: {r.valid}"
              + ("" if r.valid else f"  ERRORS: {r.errors[:3]}"), flush=True)
    out = Path(args.out)
    stem = f"{topo.name}_seed{args.seed}" if not args.circuit else f"{topo.name}_custom"
    summary_plot(topo, results, out / f"{stem}_summary.png")
    animate(topo, circ, results, None if args.no_gif else out / f"{stem}.gif", show=not args.no_show,
            interval_ms=args.interval)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RL qubit routing: train, evaluate and visualise.")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train-tabular", help="Stage 1: tabular Q-learning on one small chip")
    p.add_argument("--topology", default="line:4", help="e.g. line:4, ring:4, star:4 (keep it small)")
    p.add_argument("--episodes", type=int, default=300_000)
    p.add_argument("--k", type=int, default=3, help="waiting gates the agent can choose from")
    p.add_argument("--lookahead", type=int, default=1, help="later gates included in the state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/tabular")
    p.set_defaults(func=cmd_train_tabular)

    p = sub.add_parser("train-dqn", help="Stage 2: Double DQN on a mix of chips")
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--topologies", nargs="+", help="training chips (default: 8 chips, see routing/dqn.py)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4, help="CPU threads for torch")
    p.add_argument("--out", default="runs/dqn")
    p.set_defaults(func=cmd_train_dqn)

    p = sub.add_parser("evaluate", help="benchmark vs greedy and Qiskit SABRE")
    p.add_argument("--dqn", help="path to a DQN checkpoint (.pt)")
    p.add_argument("--tabular", help="path to a tabular checkpoint (.pkl)")
    p.add_argument("--topologies", nargs="+", help="chips to test (default: training chips + 4 unseen)")
    p.add_argument("--circuits", type=int, default=30, help="circuits per chip")
    p.add_argument("--gates", type=int, default=30, help="two-qubit gates per circuit")
    p.add_argument("--seed", type=int, default=12345, help="test-set seed (differs from training/validation)")
    p.add_argument("--out", default="runs/eval")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("performance", help="full evaluation: training curves, benchmark plots, scaling, report")
    p.add_argument("--dqn", default="runs/dqn/best.pt", help="DQN checkpoint (.pt); skipped if missing")
    p.add_argument("--tabular", default="runs/tabular/best.pkl", help="tabular checkpoint (.pkl); skipped if missing")
    p.add_argument("--topologies", nargs="+", help="benchmark chips (default: training chips + 4 unseen)")
    p.add_argument("--circuits", type=int, default=30, help="benchmark circuits per chip")
    p.add_argument("--gates", type=int, default=30, help="two-qubit gates per benchmark circuit")
    p.add_argument("--seed", type=int, default=12345, help="test-set seed (differs from training/validation)")
    p.add_argument("--cx-error", type=float, default=0.01, help="per-CX error rate for the fidelity estimate")
    p.add_argument("--no-scaling", action="store_true", help="skip the circuit-length / chip-size studies")
    p.add_argument("--quick", action="store_true", help="fewer circuits everywhere (smoke test, ~1 min)")
    p.add_argument("--out", default="runs/performance")
    p.set_defaults(func=cmd_performance)

    p = sub.add_parser("visualize", help="animate agent vs SABRE on one circuit")
    p.add_argument("--model", default="runs/dqn/best.pt", help=".pt (DQN) or .pkl (tabular)")
    p.add_argument("--topology", default="grid:3x3")
    p.add_argument("--gates", type=int, default=15)
    p.add_argument("--qubits", type=int, help="logical qubits (default: fill the chip)")
    p.add_argument("--circuit", help="your own circuit, e.g. '0-3,1-2,0-2' = CX(q0,q3), CX(q1,q2), CX(q0,q2)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--interval", type=int, default=900, help="ms per frame")
    p.add_argument("--no-show", action="store_true", help="don't open a window (just save files)")
    p.add_argument("--no-gif", action="store_true", help="don't save a GIF")
    p.add_argument("--out", default="runs/visuals")
    p.set_defaults(func=cmd_visualize)

    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
