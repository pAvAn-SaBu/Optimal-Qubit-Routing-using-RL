"""Stage 2: Double DQN (larger chips, several topologies, one network).

Network: a small MLP that scores ONE (state, action) pair from its feature
vector (features.action_features); the same weights score all 3K actions.
Masked actions get −inf. Because inputs are normalised per chip, a single
network is trained on a mix of lines, rings, grids and stars.

Double DQN target:
    a* = argmax_a' Q_online(s', a')          (online net chooses)
    y  = r + γ · Q_target(s', a*) · (1 − done)  (target net scores)
Plus: replay buffer, target network (hard sync), Huber loss, gradient clipping,
ε-greedy exploration with linear decay, best checkpoint by validation SWAPs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import features as F
from .env import CircuitSampler, RewardConfig, RoutingEnv
from .policies import greedy_policy
from .simulator import Mode, RoutingState
from .topology import from_spec
from .training import MetricsLogger, ValidationSet, linear_schedule

log = logging.getLogger(__name__)


@dataclass
class DQNConfig:
    train_topologies: list[str] = field(default_factory=lambda: [
        "line:5", "line:7", "ring:6", "ring:8", "grid:2x3", "grid:3x3", "star:6", "star:8"])
    k: int = 4
    episodes: int = 6000
    min_gates: int = 10
    max_gates: int = 40
    hidden: int = 128
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 50_000
    warmup_steps: int = 2_000
    target_sync: int = 1_000   # env steps between target-network syncs
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_frac: float = 0.6
    val_every: int = 250
    val_per_topology: int = 10
    val_gates: int = 30
    seed: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)


class QNet(nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """feats [B, A, F], mask [B, A] bool -> Q [B, A] (masked = −inf)."""
        q = self.net(feats).squeeze(-1)
        return q.masked_fill(~mask, float("-inf"))


class ReplayBuffer:
    def __init__(self, size: int, n_actions: int, n_features: int) -> None:
        self.size, self.ptr, self.full = size, 0, False
        self.feats = np.zeros((size, n_actions, n_features), np.float32)
        self.next_feats = np.zeros((size, n_actions, n_features), np.float32)
        self.next_mask = np.zeros((size, n_actions), bool)
        self.actions = np.zeros(size, np.int64)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)

    def __len__(self) -> int:
        return self.size if self.full else self.ptr

    def add(self, feats, action, reward, next_feats, next_mask, done) -> None:
        i = self.ptr
        self.feats[i], self.actions[i], self.rewards[i] = feats, action, reward
        self.next_feats[i], self.next_mask[i], self.dones[i] = next_feats, next_mask, done
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(len(self), size=n)
        return (torch.from_numpy(self.feats[idx]), torch.from_numpy(self.actions[idx]),
                torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.next_feats[idx]),
                torch.from_numpy(self.next_mask[idx]), torch.from_numpy(self.dones[idx]))


class DQNAgent:
    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.online = QNet(F.N_FEATURES, cfg.hidden)
        self.target = QNet(F.N_FEATURES, cfg.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    @torch.no_grad()
    def q_values(self, feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        q = self.online(torch.from_numpy(feats)[None], torch.from_numpy(mask.astype(bool))[None])
        return q[0].numpy()

    def act(self, feats, mask, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.choice(np.flatnonzero(mask)))
        return int(np.argmax(self.q_values(feats, mask)))

    def policy(self, state: RoutingState) -> tuple[int, Mode]:
        feats, mask = F.action_features(state, self.cfg.k)
        return F.to_gate(state, int(np.argmax(self.q_values(feats, mask))), self.cfg.k)

    def save(self, path: Path) -> None:
        cfg = asdict(self.cfg)
        torch.save({"config": cfg, "state_dict": self.online.state_dict()}, path)

    @classmethod
    def load(cls, path: Path) -> "DQNAgent":
        data = torch.load(path, weights_only=False)
        cfg = data["config"]
        cfg["reward"] = RewardConfig(**cfg["reward"])
        agent = cls(DQNConfig(**cfg))
        agent.online.load_state_dict(data["state_dict"])
        agent.target.load_state_dict(data["state_dict"])
        agent.online.eval()
        return agent


def train_dqn(cfg: DQNConfig, run_dir: Path) -> DQNAgent:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    topos = [from_spec(s) for s in cfg.train_topologies]
    cfg.reward.gamma = cfg.gamma  # shaping must use the same γ as the learner
    env = RoutingEnv(CircuitSampler(topos, cfg.min_gates, cfg.max_gates), k=cfg.k, reward=cfg.reward)
    agent = DQNAgent(cfg)
    optim = torch.optim.Adam(agent.online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, F.n_actions(cfg.k), F.N_FEATURES)
    logger = MetricsLogger(run_dir)
    logger.save_config(asdict(cfg))

    val = ValidationSet.build(topos, cfg.val_per_topology, cfg.val_gates, seed=cfg.seed + 10_000)
    greedy_val = val.evaluate(greedy_policy)["val_swaps"]
    log.info("validation set (%d circuits): SABRE %.2f, greedy %.2f mean SWAPs",
             len(val.items), sum(val.sabre_swaps) / len(val.items), greedy_val)

    best, steps, t0 = float("inf"), 0, time.time()
    res = val.evaluate(agent.policy)  # untrained network, for reference
    logger.log({**res, "episode": 0, "greedy_swaps": greedy_val, "steps": 0})
    log.info("untrained network: val %.2f", res["val_swaps"])
    obs, _ = env.reset(seed=cfg.seed)
    for ep in range(1, cfg.episodes + 1):
        eps = linear_schedule(cfg.eps_start, cfg.eps_end, ep / (cfg.episodes * cfg.eps_decay_frac))
        if ep > 1:
            obs, _ = env.reset()
        done, ep_return, losses = False, 0.0, []
        while not done:
            mask = obs["mask"].astype(bool)
            a = agent.act(obs["features"], mask, eps, rng)
            next_obs, r, done, _, _ = env.step(a)
            buffer.add(obs["features"], a, r, next_obs["features"], next_obs["mask"].astype(bool), done)
            obs, ep_return, steps = next_obs, ep_return + r, steps + 1

            if steps >= cfg.warmup_steps:
                f, act, rew, nf, nmask, dn = buffer.sample(cfg.batch_size, rng)
                q = agent.online(f, torch.ones_like(nmask)).gather(1, act[:, None]).squeeze(1)
                with torch.no_grad():
                    # terminal next-states have an all-false mask; give them a dummy valid action
                    safe_mask = nmask | (dn[:, None] > 0)
                    a_star = agent.online(nf, safe_mask).argmax(1, keepdim=True)
                    q_next = agent.target(nf, safe_mask).gather(1, a_star).squeeze(1)
                    y = rew + cfg.gamma * q_next * (1 - dn)
                loss = nn.functional.smooth_l1_loss(q, y)
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.online.parameters(), cfg.grad_clip)
                optim.step()
                losses.append(loss.item())
            if steps % cfg.target_sync == 0:
                agent.target.load_state_dict(agent.online.state_dict())

        logger.rows.append({"episode": ep, "episode_swaps": env.state.n_swaps, "return": ep_return,
                            "epsilon": eps, "loss": float(np.mean(losses)) if losses else None})

        if ep % cfg.val_every == 0 or ep == cfg.episodes:
            agent.online.eval()
            res = val.evaluate(agent.policy)
            agent.online.train()
            res.update(episode=ep, greedy_swaps=greedy_val, steps=steps)
            logger.log(res)
            if res["val_swaps"] < best:
                best = res["val_swaps"]
                agent.save(run_dir / "best.pt")
            log.info("ep %5d | steps %6d | ε %.2f | val %.2f (SABRE %.2f, greedy %.2f, ratio %.3f) | %.0fs",
                     ep, steps, eps, res["val_swaps"], res["sabre_swaps"], greedy_val,
                     res["ratio_vs_sabre"], time.time() - t0)
    agent.save(run_dir / "last.pt")
    logger.plot("Double DQN on mixed topologies")
    return DQNAgent.load(run_dir / "best.pt")
