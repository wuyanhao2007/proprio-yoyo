"""Roadmap stage 1 step 9 / PHYSICS_SPEC §9 test 12: true-state pulse baseline."""
import numpy as np
import pytest

from control.baseline import BaselineConfig, ZlajpahBaseline
from control.full_flow import catch_report, make_full_flow, reset_full_flow
from control.trajectories import ThrowFlip
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


# ------------------------------------------------------------- flip hand (§4.4)
MAIN = ("unseat", "release", "bottom", "in_cup", "flip_start", "drop", "seat_rebound", "stowed", "fail")
FLIP_FLOW = ["unseat", "release"] + ["bottom"] * 4 + ["in_cup", "flip_start", "stowed"]


def _run_flip(seed=0, duration=13.0, **params):
    p = Params.load(hand="flip", i_peak_ratio=2.0, **params)
    sim = Simulator(p, rng=np.random.default_rng(seed))
    ctrl, base = make_full_flow(p)
    reset_full_flow(sim)
    sim.run(duration, ctrl)
    return p, sim, base


def _check_flip_flow(p, sim, base):
    main = [e["type"] for e in sim.events if e["type"] in MAIN]
    assert main == FLIP_FLOW, (main, sim.s.fail_reason)
    assert sim.s.mode == Mode.STOWED and sim.s.caught
    assert np.sin(sim.s.phi2) > np.cos(p.stow_tilt_max)            # stowed mouth up
    r = catch_report(sim)
    assert 0.0 < r["v_arrive"] < p.v_cap_max                       # no magnet: it arrives by itself
    assert abs(r["theta_arrive_deg"]) < 15.0
    assert r["i2_peak"] <= p.i_max2 + 1e-9
    sigma = [e for e in sim.events if e["type"] == "bottom"][-1]["sigma"]
    assert r["flip_direction"] == sigma                            # A9: turn with sign sigma
    assert r["flip_direction"] == -base.cfg.throw_direction        # even bottoms: back the way it came
    phi2 = sim.log.arrays()["phi2"]
    assert abs(phi2[-1] - phi2[0]) < p.stow_tilt_max               # no net wrist turn per cycle
    assert (np.cos(phi2) > -0.5).all()                             # the mouth never turns to the link-1 (-x) side
    for rec in base.pulses[:base.cfg.n_bounces - 1]:               # normal bounces stay out of the seat
        assert rec.apex > p.ell_min + p.s_max


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flip_full_flow_throw_four_bounces_catch_stow(seed):
    """Flip hand: turn-down throw -> 4 bottoms -> seat (IN_CUP) -> flip back -> STOWED."""
    _check_flip_flow(*_run_flip(seed))


def test_flip_full_flow_with_cogging():
    """Seeds only change sensor noise, which the PD never sees, so seeds 0-2 above
    follow one physical trajectory. With cogging torque (random phases) the seed
    changes the arm motion itself: two really different runs."""
    runs = [_run_flip(seed, cogging_amp=0.02) for seed in (0, 1)]
    for run in runs:
        _check_flip_flow(*run)
    q1a, q1b = (sim.log.arrays()["q1"] for _, sim, _ in runs)
    assert not np.array_equal(q1a, q1b)


def test_flip_baseline_config_comes_from_the_hand_overlay():
    """hand_flip.yaml ``baseline`` block overrides the defaults; magnet has no block."""
    assert BaselineConfig.load("magnet") == BaselineConfig() == BaselineConfig.load()
    c = BaselineConfig.load("flip")
    p = Params.load(hand="flip")
    assert c.apex_target > p.ell_min + p.s_max                     # apex in the flare at most, not the seat
    assert c.a_down < Params.load().g                              # slack limit
    assert c.n_bounces % 2 == 0                                    # stow flip returns on the throw side
    assert BaselineConfig.load("flip", n_bounces=5).n_bounces == 5
    with pytest.raises(ValueError):
        BaselineConfig.load("flip", no_such_setting=1)


def test_throw_flip_profile_ends_mouth_down():
    th = ThrowFlip(t0=0.2, t_turn=0.3, direction=-1)
    sim = type("S", (), {"s": type("s", (), {"t": 0.0})()})()
    assert th(sim) == (0.0, pytest.approx(np.pi / 2))              # starts mouth up
    sim.s.t = 1.0
    assert th(sim)[1] == pytest.approx(-np.pi / 2) and th.phi_down == pytest.approx(-np.pi / 2)
    assert np.sin(ThrowFlip(direction=1).phi_down) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        ThrowFlip(direction=0)
