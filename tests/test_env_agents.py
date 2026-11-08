import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from routing import features as F
from routing.circuit import Circuit, random_circuit
from routing.dqn import DQNAgent, DQNConfig, train_dqn
from routing.env import CircuitSampler, RewardConfig, RoutingEnv
from routing.policies import greedy_policy, route, route_sabre
from routing.q_learning import TabularConfig, TabularQAgent, train_tabular
from routing.simulator import RoutingState
from routing.topology import from_spec, line


def make_env(**kw):
    topos = [from_spec(s) for s in ["line:5", "ring:6", "grid:2x3", "star:5"]]
    return RoutingEnv(CircuitSampler(topos, 5, 20), **kw)


class _MaskTolerant(RoutingEnv):
    """check_env samples actions without the mask; map those onto a valid action for this test only."""
    def step(self, action):
        mask = F.action_mask(self.state, self.k)
        if not mask[action]:
            action = int(np.flatnonzero(mask)[action % mask.sum()])
        return super().step(action)


@pytest.mark.filterwarnings("ignore:.*Box observation space.*")
def test_gymnasium_api_compliance():
    topos = [from_spec(s) for s in ["line:5", "grid:2x3"]]
    check_env(_MaskTolerant(CircuitSampler(topos, 5, 20), k=4), skip_render_check=True)


def test_episodes_terminate_with_masked_random_actions():
    env = make_env(k=4)
    rng = np.random.default_rng(0)
    for ep in range(50):
        obs, info = env.reset(seed=ep)
        assert obs["features"].shape == (12, F.N_FEATURES)
        for _ in range(100):
            a = int(rng.choice(np.flatnonzero(info["action_mask"])))
            obs, r, done, trunc, info = env.step(a)
            if done:
                break
        assert done, "every action runs >= 1 gate, so episodes must end"


def test_masked_action_rejected():
    env = make_env(k=4)
    _, info = env.reset(seed=1)
    invalid = np.flatnonzero(~info["action_mask"])
    if len(invalid):
        with pytest.raises(ValueError):
            env.step(int(invalid[0]))


def test_reward_telescopes_with_gamma_one():
    """With γ = 1 the shaping terms cancel: return = −SWAPs + gate_reward·gates + bonus − Φ(s0)."""
    cfg = RewardConfig(gamma=1.0)
    env = make_env(k=4, reward=cfg)
    rng = np.random.default_rng(3)
    for ep in range(20):
        _, info = env.reset(seed=100 + ep)
        phi0 = -cfg.shaping_beta * F.front_cost(env.state)
        gates_before = env.state.n_done
        total, done = 0.0, False
        while not done:
            _, r, done, _, info = env.step(int(rng.choice(np.flatnonzero(info["action_mask"]))))
            total += r
        gates = env.state.n_done - gates_before
        expected = -env.state.n_swaps + cfg.gate_reward * gates + cfg.finish_bonus - phi0
        assert total == pytest.approx(expected)


def test_action_features_match_simulator():
    c = Circuit(4, ((0, 3), (1, 2)))
    s = RoutingState(line(4), c)
    feats, mask = F.action_features(s, k=4)
    assert mask.sum() == 3  # one waiting gate (0,3); (1,2) is adjacent and already ran
    # every mode of CX(q0,q3) on a 4-line costs 2 SWAPs; diameter is 3
    assert np.allclose(feats[:3, 0], 2 / 3)
    assert not feats[3:].any()


def test_tabular_q_update_rule():
    agent = TabularQAgent(TabularConfig(k=1))
    mask = np.array([True, True, True])
    agent.update("s", 0, -1.0, "s2", mask, False, alpha=0.5)
    assert agent.q["s"][0] == pytest.approx(-0.5)  # 0 + 0.5 * (-1 + γ·0 − 0)
    agent.q["s2"][:] = [-2.0, -1.0, -3.0]
    agent.update("s", 1, 0.0, "s2", mask, False, alpha=1.0)
    assert agent.q["s"][1] == pytest.approx(-agent.cfg.gamma)  # max over s2 is -1


def test_tabular_and_dqn_train_save_load_and_route(tmp_path):
    tab = train_tabular(TabularConfig(episodes=500, val_every=500, val_circuits=5), tmp_path / "tab", line(4))
    dqn = train_dqn(DQNConfig(train_topologies=["line:5", "grid:2x3"], episodes=20, warmup_steps=50,
                              batch_size=16, val_every=20, val_per_topology=2), tmp_path / "dqn")
    assert (tmp_path / "dqn" / "training_curve.png").exists()
    reloaded = DQNAgent.load(tmp_path / "dqn" / "best.pt")
    rng = np.random.default_rng(0)
    for spec in ["line:5", "grid:3x3", "star:7"]:  # DQN also runs on chips it never saw
        topo = from_spec(spec)
        circ = random_circuit(topo.n, 20, rng)
        r1, r2 = route(topo, circ, dqn.policy), route(topo, circ, reloaded.policy)
        assert r1.valid and r1.swaps == r2.swaps
    circ = random_circuit(4, 12, rng)
    assert route(line(4), circ, tab.policy).valid


def test_baselines_valid_on_partially_filled_chip():
    rng = np.random.default_rng(5)
    topo = from_spec("grid:3x3")
    circ = random_circuit(5, 25, rng)  # 5 logical qubits on 9 positions
    assert route(topo, circ, greedy_policy).valid
    assert route_sabre(topo, circ).valid
