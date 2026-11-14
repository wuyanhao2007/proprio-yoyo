"""Roadmap stage 1 step 9 / PHYSICS_SPEC §9 test 12: true-state pulse baseline."""
import numpy as np
import pytest

from control.baseline import BaselineConfig, ZlajpahBaseline
from sim.params import Params
from sim.simulator import Simulator
from yoyo.modes import Mode


def _run(seed=0, duration=11.0, log=True, **cfg):
    p = Params.load(i_peak_ratio=2.0)
    sim = Simulator(p, rng=np.random.default_rng(seed), log=log)
    sim.reset_stowed()
    ctrl = ZlajpahBaseline(p, BaselineConfig(**cfg))
    sim.run(duration, ctrl)
    return p, sim, ctrl


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_flow_throw_three_bounces_catch(seed):
    """Stage-1 gate: phase sequence fall -> bounce -> climb, 3 cycles, then caught."""
    p, sim, ctrl = _run(seed)
    main = [e["type"] for e in sim.events if e["type"] in ("release", "bottom", "stowed", "fail")]
    assert main == ["release", "bottom", "bottom", "bottom", "stowed"], main
    assert sim.s.mode == Mode.STOWED and sim.s.caught
    # the two normal bounces peak within +-5 cm of the target, below the funnel mouth
    for r in ctrl.pulses[:2]:
        assert abs(r.apex - ctrl.cfg.apex_target) < 0.05
        assert r.apex > p.ell_mouth
    # hook drag at every bottom stays below 2 cm (stage-2 gate)
    d = sim.log.arrays()
    for tb in [e["t"] for e in sim.events if e["type"] == "bottom"]:
        k = np.searchsorted(d["t"], tb)
        w = (d["t"] > tb) & (d["t"] < tb + 0.03)
        assert d["Hy"][k] - d["Hy"][w].min() < 0.02


def test_zlajpah_sustains_20_bounces():
    """Test 12: the pulse baseline keeps >= 20 bounces without decay."""
    p, sim, ctrl = _run(duration=60.0, log=False, n_bounces=0)
    assert sim.s.mode != Mode.FAIL, sim.s.fail_reason
    assert sim.s.n_bottom >= 20
    apex = np.array([r.apex for r in ctrl.pulses if not np.isnan(r.apex)])
    assert len(apex) >= 19
    assert apex[-5:].mean() < apex[:5].mean() + 0.02          # no decay (l grows if it decays)
    assert np.all(np.abs(apex - ctrl.cfg.apex_target) < 0.06)
