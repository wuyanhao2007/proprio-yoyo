"""PHYSICS_SPEC §9 tests 1-6 (yo-yo, string, events) with a prescribed hook."""
import math

import numpy as np
import pytest

from sim.params import Params
from sim.simulator import KinematicHook, Simulator
from yoyo.modes import Mode


def make(hook=None, **over):
    over.setdefault("capture_model", "none")
    p = Params.load(**over)
    return p, Simulator(p, hook=hook or KinematicHook.fixed(), sensor_noise=False)


def test_derived_defaults():
    """§1 numbers quoted in the spec."""
    p = Params.load()
    assert p.kappa() == pytest.approx(0.071, abs=5e-4)
    assert p.zeta == pytest.approx(0.858, abs=5e-4)
    assert p.ell_min == pytest.approx(0.031)
    assert p.ell_mouth == pytest.approx(0.073)
    v = math.sqrt(2 * p.kappa() * p.g * p.L)
    assert v == pytest.approx(1.06, abs=0.01)
    assert v / p.r0 == pytest.approx(176, abs=1)
    assert math.sqrt(2 * p.L / (p.kappa() * p.g)) == pytest.approx(1.5, abs=0.02)


def test_free_fall():
    """Test 1 (§3.3): l(t) = kappa g t^2 / 2 within 1e-6 m."""
    p, sim = make(planar=False)
    sim.reset_taut(ell=0.0)
    for t_check in (0.5, 1.0, 1.4):
        sim.run(t_check - sim.s.t)
        assert sim.s.ell == pytest.approx(0.5 * p.kappa() * p.g * sim.s.t**2, abs=1e-6)


@pytest.mark.parametrize("a", [0.0, 3.0, 12.0])
def test_tension_formula(a):
    """Test 2 (§3.2): hook accelerating up at a -> T = m (g + a)(1 - kappa)."""
    p, sim = make(hook=KinematicHook.const_accel(a), planar=False)
    sim.reset_taut(ell=0.1)
    sim.run(0.2)
    assert sim.s.mode == Mode.TAUT
    assert sim.diag.T == pytest.approx(p.m * (p.g + a) * (1 - p.kappa()), rel=1e-9)
    # and l_dd = kappa (g + a): check via a finite run
    assert sim.s.ell == pytest.approx(0.1 + 0.5 * p.kappa() * (p.g + a) * sim.s.t**2, abs=1e-6)


def _energy(p, s):
    return 0.5 * p.m * (s.vx**2 + s.vy**2) + 0.5 * p.I * s.omega**2 + p.m * p.g * s.py


@pytest.mark.parametrize("variable_radius", [False, True])
def test_energy_conservation(variable_radius):
    """Test 3 (§3.2, §3.4): fixed hook, no damping, swing + r(l): drift < 0.1 % in 2 s."""
    p, sim = make(L=5.0, variable_radius=variable_radius, c_l=0.0)
    sim.reset_taut(ell=0.3, elld=0.0, theta=0.4, thetad=0.0)
    E0 = _energy(p, sim.s)
    E = [E0]
    for _ in range(10):
        sim.run(0.2)
        assert sim.s.mode == Mode.TAUT
        E.append(_energy(p, sim.s))
    drift = max(abs(e - E0) for e in E)
    scale = p.m * p.g * 1.0  # reference energy scale (1 m of drop)
    assert drift / scale < 1e-3


def test_bottom_event():
    """Test 4 (§3.6): E+/E- = zeta_e^2, sigma flips, omega keeps sign, int F_b = J."""
    p, sim = make(planar=False, c_loss=0.95)
    sim.reset_taut(ell=0.7, elld=0.5, sigma=1)
    sim.run(0.5)
    ev = next(e for e in sim.events if e["type"] == "bottom")
    assert ev["E_plus"] / ev["E_minus"] == pytest.approx(p.zeta_e**2, abs=1e-6)
    assert ev["sigma"] == -1
    assert np.sign(ev["omega_plus"]) == np.sign(ev["omega_minus"])
    d = sim.log.arrays()
    fb = np.hypot(d["F_bx"], d["F_by"])
    impulse = np.sum(0.5 * (fb[1:] + fb[:-1]) * np.diff(d["t"]))
    assert impulse == pytest.approx(ev["J"], rel=1e-3)
    assert ev["pulse_dt"] == pytest.approx(math.pi * p.r0 / ev["v_minus"])


def test_passive_bounce():
    """Test 5 (§3.7): apex ratio zeta_e^2 and h = kappa g T_b^2 / 8 within 1 %."""
    p, sim = make(planar=False)
    sim.reset_taut(ell=0.0)
    sim.run(7.0)
    d = sim.log.arrays()
    bt = [e["t"] for e in sim.events if e["type"] == "bottom"]
    assert len(bt) >= 3
    apex = []
    for a, b in zip(bt[:-1], bt[1:]):
        w = (d["t"] > a) & (d["t"] < b)
        apex.append(p.L - d["ell"][w].min())
        Tb = b - a
        assert apex[-1] == pytest.approx(p.kappa() * p.g * Tb**2 / 8, rel=1e-2)
    for h0, h1 in zip(apex[:-1], apex[1:]):
        assert h1 / h0 == pytest.approx(p.zeta_e**2, rel=1e-3)


def test_slack_and_snap():
    """Test 6 (§3.5): hook accelerates down at 1.5 g -> FREE; stops -> snaps taut;
    with e_taut = 0 the post-impulse state satisfies v_r = sigma r omega."""
    g = 9.81
    t_stop = 0.3

    def hook(t):
        if t < t_stop:
            return 0.0, -0.75 * g * t * t, 0.0, -1.5 * g * t, 0.0, -1.5 * g, -math.pi / 2, 0.0
        y = -0.75 * g * t_stop**2
        return 0.0, y, 0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0

    p, sim = make(hook=KinematicHook(hook), planar=False, e_taut=0.0)
    sim.reset_taut(ell=0.2, elld=0.3, sigma=1)
    sim.run(1.0)
    types = [e["type"] for e in sim.events]
    assert "slack" in types
    i_slack = types.index("slack")
    snaps = [e for e in sim.events[i_slack:] if e["type"] == "snap" and e["P"] > 0]
    assert snaps, types
    ev = snaps[0]
    assert ev["result"] == "taut"
    assert ev["v_r_plus"] == pytest.approx(ev["sigma"] * ev["r"] * ev["omega_plus"], abs=1e-9)


def test_snap_momentum_with_arm():
    """Test 6b (§3.5, §5.3): with the arm attached, the snap impulse is equal and
    opposite: m dp = -P u and M dq_dot = Jac^T P u."""
    from arm import dynamics as ad
    from arm import kinematics as ak
    p = Params.load(e_taut=0.0)
    sim = Simulator(p, sensor_noise=False)
    sim.reset_taut(ell=0.3, elld=0.0)
    s = sim.s
    # put the yo-yo into FREE moving away from the hook with no spin
    s.mode = int(Mode.FREE)
    s.ell_s = s.ell
    s.vy -= 1.0
    s.omega = 0.0
    sim._event_snap()
    ev = sim.events[-1]
    P, ux, uy = ev["P"], ev["ux"], ev["uy"]
    assert P > 0
    assert p.m * (ev["vx_plus"] - ev["vx_minus"]) == pytest.approx(-P * ux, abs=1e-12)
    assert p.m * (ev["vy_plus"] - ev["vy_minus"]) == pytest.approx(-P * uy, abs=1e-12)
    m11, m12, m22 = ad.mass_matrix(sim.c_free, s.q1, s.phi2)
    dq1 = ev["q1d_plus"] - ev["q1d_minus"]
    dq2 = ev["phi2d_plus"] - ev["phi2d_minus"]
    (j00, j01), (j10, j11) = ak.hook_jacobian(p.l1, p.l2, s.q1, s.phi2)
    assert m11 * dq1 + m12 * dq2 == pytest.approx(j00 * P * ux + j10 * P * uy, abs=1e-12)
    assert m12 * dq1 + m22 * dq2 == pytest.approx(j01 * P * ux + j11 * P * uy, abs=1e-12)
    # e = 0: relative radial speed equals the paid-out string speed afterwards
    assert ev["v_r_plus"] == pytest.approx(ev["sigma"] * ev["r"] * ev["omega_plus"], abs=1e-9)
