"""Environment interface and whole-chain checks (§8, test 12 placeholder)."""
import math

import numpy as np
import pytest

from control.trajectories import Throw
from sim.env import OBS_NAMES, PRIV_NAMES, YoyoEnv, predict_time_to_bottom
from sim.log import load_run, save_run
from sim.params import Params
from sim.simulator import Simulator
from yoyo.modes import Mode


def test_reset_step_shapes():
    env = YoyoEnv(randomize=True)
    st = env.reset(seed=0)
    st, obs, priv, r, done, info = env.step(st, np.zeros(2))
    assert obs.shape == (len(OBS_NAMES),)
    assert priv.shape == (len(PRIV_NAMES),)
    assert st.sim.s.t == pytest.approx(st.sim.p.substeps * st.sim.p.dt)
    assert not done and isinstance(info["events"], list)


def test_action_clipping():
    env = YoyoEnv(max_delta=0.01)
    st = env.reset(seed=0)
    q0 = st.q_cmd.copy()
    st, *_ = env.step(st, np.array([1.0, -1.0]))
    assert np.allclose(st.q_cmd - q0, [0.01, -0.01])


def test_start_after_bottom():
    env = YoyoEnv()
    st = env.reset(seed=0, start="after_bottom", k_bottom=1)
    assert st.sim.s.n_bottom == 1 and st.sim.s.mode == Mode.TAUT


def test_throw_and_bounce_chain():
    """Scripted throw: release, bottom with a bounded hook drag (< 2 cm, stage 2 gate)."""
    p = Params.load()
    sim = Simulator(p, rng=np.random.default_rng(0))
    sim.reset_stowed()
    sim.run(2.2, Throw(p.l1))
    types = [e["type"] for e in sim.events]
    assert "release" in types and "bottom" in types
    d = sim.log.arrays()
    tb = next(e["t"] for e in sim.events if e["type"] == "bottom")
    k = np.searchsorted(d["t"], tb)
    w = (d["t"] > tb) & (d["t"] < tb + 0.15)
    drag = d["Hy"][k] - d["Hy"][w].min()
    assert 0.0 < drag < 0.02


def test_time_to_bottom_prediction():
    p = Params.load()
    a = p.kappa() * p.g
    assert predict_time_to_bottom(p, 0.0, 0.0) == pytest.approx(math.sqrt(2 * p.L / a))
    t = predict_time_to_bottom(p, 0.4, -0.5)
    assert t > 0.5 / a


def test_log_roundtrip(tmp_path):
    p = Params.load()
    sim = Simulator(p, sensor_noise=False)
    sim.reset_stowed()
    sim.run(0.05)
    save_run(tmp_path, sim.log, sim.events, p)
    d, ev, pd = load_run(tmp_path)
    assert len(d["t"]) == len(sim.log.rows)
    assert Params(**pd) == p
