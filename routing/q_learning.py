"""Stage 1: tabular Q-learning (small chips).

Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]

The table is a dict from the exact discrete state (features.tabular_key) to an
array of 3K Q-values. A table only covers the chip it was trained on, and only
states it has visited. At routing time an unseen state falls back to the greedy
baseline, and the fallback rate is reported so this limitation is visible.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import features as F
from .env import CircuitSampler, RewardConfig, RoutingEnv
from .policies import greedy_policy
from .simulator import Mode, RoutingState
from .topology import Topology
from .training import MetricsLogger, ValidationSet, linear_schedule

log = logging.getLogger(__name__)


@dataclass
class TabularConfig:
    topology: str = "line:4"
    k: int = 3                # waiting gates the agent can choose from
    lookahead: int = 1        # later gates included in the state
    episodes: int = 300000
    min_gates: int = 6
    max_gates: int = 14
    alpha: float = 0.2
    alpha_end: float = 0.01   # α decays linearly so Q-values settle (targets are noisy:
                              # the state shows only part of the remaining circuit)
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.7  # fraction of episodes over which ε decays
    val_every: int = 25000
    val_circuits: int = 100
    seed: int = 0


class TabularQAgent:
    def __init__(self, cfg: TabularConfig) -> None:
        self.cfg = cfg
        self.q: dict[tuple, np.ndarray] = {}
        self.lookups = 0
        self.misses = 0

    def key(self, state: RoutingState) -> tuple:
        return F.tabular_key(state, self.cfg.k, self.cfg.lookahead)

    def values(self, key: tuple) -> np.ndarray:
        if key not in self.q:
            # zeros are optimistic here (every reward sum is ≲ 0), which encourages exploration
            self.q[key] = np.zeros(F.n_actions(self.cfg.k))
        return self.q[key]

    def act(self, state: RoutingState, mask: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        valid = np.flatnonzero(mask)
        if rng.random() < epsilon:
            return int(rng.choice(valid))
        q = self.values(self.key(state))
        return int(valid[np.argmax(q[valid])])

    def update(self, key, action, reward, next_key, next_mask, done, alpha: float) -> None:
        q = self.values(key)
        target = reward
        if not done:
            nq = self.values(next_key)
            target += self.cfg.gamma * nq[next_mask].max()
        q[action] += alpha * (target - q[action])

    # ------------------------------------------------------------- inference
    def policy(self, state: RoutingState) -> tuple[int, Mode]:
        self.lookups += 1
        key = self.key(state)
        if key not in self.q:
            self.misses += 1
            return greedy_policy(state)
        mask = F.action_mask(state, self.cfg.k)
        q = np.where(mask, self.q[key], -np.inf)
        return F.to_gate(state, int(np.argmax(q)), self.cfg.k)

    @property
    def fallback_rate(self) -> float:
        return self.misses / max(self.lookups, 1)

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"config": asdict(self.cfg), "q": self.q}, fh)

    @classmethod
    def load(cls, path: Path) -> "TabularQAgent":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        agent = cls(TabularConfig(**data["config"]))
        agent.q = data["q"]
        return agent


def train_tabular(cfg: TabularConfig, run_dir: Path, topo: Topology) -> TabularQAgent:
    from .policies import route

    rng = np.random.default_rng(cfg.seed)
    sampler = CircuitSampler([topo], cfg.min_gates, cfg.max_gates)
    env = RoutingEnv(sampler, k=cfg.k, reward=RewardConfig(gamma=cfg.gamma), compute_features=False)
    agent = TabularQAgent(cfg)
    logger = MetricsLogger(run_dir)
    logger.save_config(asdict(cfg))
    val = ValidationSet.build([topo], cfg.val_circuits, n_gates=cfg.max_gates, seed=cfg.seed + 10_000)
    greedy_val = val.evaluate(greedy_policy)["val_swaps"]
    log.info("validation set: SABRE %.2f, greedy %.2f mean SWAPs",
             sum(val.sabre_swaps) / len(val.sabre_swaps), greedy_val)

    best = float("inf")
    t0 = time.time()
    _, info = env.reset(seed=cfg.seed)
    for ep in range(1, cfg.episodes + 1):
        eps = linear_schedule(cfg.eps_start, cfg.eps_end, ep / (cfg.episodes * cfg.eps_decay_frac))
        alpha = linear_schedule(cfg.alpha, cfg.alpha_end, ep / cfg.episodes)
        if ep > 1:
            _, info = env.reset()
        done = False
        while not done:
            key = agent.key(env.state)
            a = agent.act(env.state, info["action_mask"], eps, rng)
            _, r, done, _, info = env.step(a)
            next_key = None if done else agent.key(env.state)
            agent.update(key, a, r, next_key, info["action_mask"], done, alpha)
        logger.rows.append({"episode": ep, "episode_swaps": env.state.n_swaps, "epsilon": eps})

        if ep % cfg.val_every == 0 or ep == cfg.episodes:
            agent.lookups = agent.misses = 0
            res = val.evaluate(agent.policy)
            res.update(episode=ep, greedy_swaps=greedy_val, fallback_rate=agent.fallback_rate,
                       table_states=len(agent.q))
            logger.log(res)
            if res["val_swaps"] < best:
                best = res["val_swaps"]
                agent.save(run_dir / "best.pkl")
            log.info("ep %6d | ε %.2f | val %.2f (SABRE %.2f, greedy %.2f) | fallback %.1f%% | %d states | %.0fs",
                     ep, eps, res["val_swaps"], res["sabre_swaps"], greedy_val, 100 * agent.fallback_rate,
                     len(agent.q), time.time() - t0)
    agent.save(run_dir / "last.pkl")
    logger.plot(f"Tabular Q-learning on {topo.name}")
    return TabularQAgent.load(run_dir / "best.pkl")
