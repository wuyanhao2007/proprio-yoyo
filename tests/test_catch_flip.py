"""PHYSICS_SPEC §4.4 flip-catch tests (catch_mode = "flip", amendment A9).

The four §4.4 tests:
  1. mouth up at rest: the yo-yo stays at s = 0;
  2. flip 180 deg from the seat (raised-cosine velocity, wrist axis fixed,
     kappa = 0.32): regression against the estimate; with A9 the outcome depends
     on the flip direction, the first-written bead model gives ~3.5 / 1.6 cm;
  3. mouth up -> mouth down: the yo-yo falls out and switches to TAUT;
  4. catch_mode = "magnet": unchanged (tests 1-12 run in the other files).
plus consistency checks of the coupled IN_CUP equations.
"""
import math

import numpy as np
import pytest

from arm import dynamics as ad
from control.stow_flip import StowFlip, StowFlipConfig
from sim.params import Params
from sim.simulator import Simulator
from yoyo.catch_flip import cosine_flip, cup_system, simulate_prescribed
from yoyo.catch_magnet import MagnetCatch
from yoyo.modes import Mode

DOWN, UP = -math.pi / 2, math.pi / 2


def flip_params(**kw):
    return Params.load(hand="flip", **kw)


def _types(sim):
    return [e["type"] for e in sim.events]


# ------------------------------------------------------------------ config
def test_flip_overlay_values():
    """§4.4 table: hand_flip.yaml sets the flip column, everything from config."""
    p = flip_params()
    assert p.catch_mode == "flip"
    assert p.l2 == pytest.approx(-p.ell_min, abs=5e-4)      # yo-yo centre on the wrist axis
    assert p.d_cup == pytest.approx(0.088)
    assert p.s_max == pytest.approx(0.03)
    assert p.F_mag0 == 0.0
    assert p.lc == pytest.approx(0.010)
    assert p.Ic == pytest.approx(p.mc * 0.035**2)
    assert p.kappa(p.r_full) == pytest.approx(0.32, abs=0.01)
    assert StowFlipConfig.load().t_flip == pytest.approx(0.12)


# ------------------------------------------------- test 1: mouth up at rest
def test_mouth_up_at_rest_stays_seated():
    p = flip_params()
    r = simulate_prescribed(p, lambda t: (UP, 0.0, 0.0), 0.5, dt=1e-4)
    assert r["fell_at"] is None and np.all(r["s"] == 0.0)
    sim = Simulator(p, rng=np.random.default_rng(0))
    sim.reset_in_cup(q2_ref=UP)
    s_seen = []
    sim.run(0.45, lambda sim: s_seen.append(sim.s.cup_s))
    assert sim.s.mode == Mode.IN_CUP and max(s_seen) == 0.0
    sim.run(0.1)                                             # seated mouth up for 0.5 s
    assert sim.s.mode == Mode.STOWED and sim.s.caught
    assert _types(sim)[-1] == "stowed"


# ------------------------------------------- test 2: flip from the seat (regression)
@pytest.mark.parametrize("t_flip,s_ref", [(0.12, 0.036), (0.08, 0.016)])
def test_flip_regression_first_written_bead_model(t_flip, s_ref):
    """§4.4 numbers (~3.5 cm at 0.12 s, ~1.6 cm at 0.08 s): s_dd = kappa A_out with the
    string bilateral (spin_coupling false), s_max lifted so the maximum is visible."""
    p = flip_params(spin_coupling=False, s_max=1.0)
    r = simulate_prescribed(p, cosine_flip(t_flip, phi0=DOWN), t_flip + 0.4, dt=1e-4)
    assert r["s"].max() == pytest.approx(s_ref, rel=0.03)


@pytest.mark.parametrize("sigma", [1, -1])
@pytest.mark.parametrize("t_flip", [0.08, 0.12, 0.15])
def test_flip_direction_decides_with_spin_coupling(t_flip, sigma):
    """A9: turning with sign sigma winds the string first and keeps the yo-yo in the
    seat; turning the other way lets the string go slack and the yo-yo falls out."""
    p = flip_params()
    good = simulate_prescribed(p, cosine_flip(t_flip, phi0=DOWN, direction=sigma),
                               t_flip + 0.3, sigma=sigma, dt=1e-4)
    assert good["fell_at"] is None
    assert good["s"].max() < 1e-3
    bad = simulate_prescribed(p, cosine_flip(t_flip, phi0=DOWN, direction=-sigma),
                              t_flip + 0.3, sigma=sigma, dt=1e-4)
    if t_flip >= 0.08:
        assert bad["fell_at"] is not None and bad["fell_at"] < t_flip


def test_spin_coupling_formula():
    """Seated -> sliding taut: s_dd = kappa A_out - sigma (1 - kappa) r phi2_dd and
    T = (I kappa / r^2)(A_out + sigma r phi2_dd), at s = 0 with the wrist axis fixed."""
    p = flip_params()
    r, kap = p.r_full, p.kappa(p.r_full)
    from yoyo.catch_flip import cup_accel
    for sigma in (1, -1):
        for phi, phdd in ((DOWN, -150.0), (DOWN + 0.4, 80.0)):
            M, f, _, _ = cup_system(p, 0.0, phi, 0.0, 0.0, 1e-4, 0.0, (1.0, 0.0, 1.0), (0.0, 0.0))
            v = np.zeros(4)
            presc = ((np.array([1.0, 0, 0, 0]), 0.0), (np.array([0, 1.0, 0, 0]), phdd))
            z, N, T, taut, _ = cup_accel(p, sigma, M, f, 1e-4, 0.0, 0.0, v, presc)
            A = -p.g * math.sin(phi) * 1.0          # gravity only (x ~ 0, at rest)
            T_pred = p.I * kap / r**2 * (A + sigma * r * phdd)
            if T_pred > 0:
                assert taut and z[2] == pytest.approx(kap * A - sigma * (1 - kap) * r * phdd, rel=1e-3)
                assert T == pytest.approx(T_pred, rel=1e-3)
            else:                                  # string cannot push: slack, no kappa
                assert not taut and z[2] == pytest.approx(A, rel=1e-3)


@pytest.mark.parametrize("sigma", [1, -1])
def test_stow_flip_controller_full_sim(sigma):
    """control/stow_flip.py in the coupled simulator: auto direction stows, the
    opposite direction drops the yo-yo out of the seat."""
    p = flip_params(i_peak_ratio=2.0)
    sim = Simulator(p, rng=np.random.default_rng(0))
    sim.reset_in_cup(q2_ref=DOWN, sigma=sigma)
    sim.run(0.8, StowFlip(p, StowFlipConfig.load()))
    ev = _types(sim)
    assert ev[0] == "flip_start" and "drop" not in ev
    assert sim.s.mode == Mode.STOWED and sim.s.caught
    assert math.sin(sim.s.phi2) > math.cos(p.stow_tilt_max)
    sim = Simulator(p, rng=np.random.default_rng(0))
    sim.reset_in_cup(q2_ref=DOWN, sigma=sigma)
    sim.run(0.3, StowFlip(p, StowFlipConfig.load(direction=-sigma)))
    assert "drop" in _types(sim)


# --------------------------------------------- test 3: mouth up -> mouth down
def test_mouth_up_to_down_falls_out_to_taut():
    p = flip_params(i_peak_ratio=2.0)
    sim = Simulator(p, rng=np.random.default_rng(0))
    sim.reset_stowed(q2_ref=UP)
    assert not sim.s.released
    prof = cosine_flip(0.3, phi0=UP, direction=1, t0=0.05)

    def ctrl(sim):
        return 0.0, prof(sim.s.t + p.cmd_delay)[0]
    sim.run(0.8, ctrl, stop_on_fail=False)
    main = [e for e in sim.events if e["type"] in ("unseat", "release", "drop", "stowed", "fail")]
    assert [e["type"] for e in main][:2] == ["unseat", "release"], _types(sim)
    rel = main[1]
    assert rel["s"] >= p.s_max and rel["to"] == Mode.TAUT
    assert sim.s.released
    a = sim.log.arrays()
    k = np.searchsorted(a["t"], rel["t"])
    assert a["mode"][k - 1] == Mode.IN_CUP and a["mode"][k + 1] == Mode.TAUT


def test_mouth_down_drop_time():
    """Seated mouth down, nothing holds it: s_dd = kappa(r_full) g until s_max."""
    p = flip_params()
    sim = Simulator(p, rng=np.random.default_rng(0))
    sim.reset_in_cup(q2_ref=DOWN)
    sim.run(0.3)
    drop = next(e for e in sim.events if e["type"] == "drop")
    t_pred = math.sqrt(2 * p.s_max / (p.kappa(p.r_full) * p.g))
    assert drop["t"] == pytest.approx(t_pred, rel=0.02)
    assert drop["to"] == Mode.TAUT
    assert sim.s.mode == Mode.TAUT and sim.s.ell > p.ell_mouth      # left through the mouth


# ---------------------------------------------- coupled-model consistency
def test_seated_mass_matrix_matches_stowed_constants():
    """Seated rigid (s = 0, theta_y_d = phi2_d) reduction of M4 = c_stow with spin."""
    p = flip_params()
    c_free = ad.link_constants(p, stowed=False)
    c_stow = ad.link_constants(p, stowed=True, yoyo_spin=True)
    A = np.array([[1.0, 0], [0, 1.0], [0, 0], [0, 1.0]])
    for q1, phi2 in ((0.0, DOWN), (0.4, 1.0), (-0.7, 2.6)):
        Ma = ad.mass_matrix(c_free, q1, phi2)
        M, _, _, _ = cup_system(p, q1, phi2, 0.0, 0.0, 0.0, 0.0, Ma, (0.0, 0.0))
        Mr = A.T @ M @ A
        m11, m12, m22 = ad.mass_matrix(c_stow, q1, phi2)
        assert np.allclose(Mr, [[m11, m12], [m12, m22]], rtol=1e-12)


def test_in_cup_energy_conserved_while_sliding():
    """No motor torque, no friction: arm + sliding, spinning yo-yo conserve energy."""
    p = flip_params(b1=0.0, b2=0.0, tc1=0.0, tc2=0.0, cogging_amp=0.0, s_max=0.2)
    sim = Simulator(p, rng=np.random.default_rng(0), sensor_noise=False)
    sim.reset_in_cup(q2_ref=DOWN + 0.3)
    sim.p = p = p.replace(Kp1=0.0, Kd1=0.0, Kp2=0.0, Kd2=0.0)   # motors off after placing
    sim.motor.i1 = sim.motor.i2 = 0.0
    c = sim.c_free
    r = p.r_full

    def energy():
        s = sim.s
        E = ad.energy(c, s.q1, s.phi2, s.q1d, s.phi2d)
        th_d = s.omega + s.phi2d
        return E + 0.5 * p.m * (s.vx**2 + s.vy**2) + 0.5 * p.I * th_d**2 + p.m * p.g * s.py
    E0 = energy()
    Es = []
    for _ in range(600):                       # 0.12 s, sliding out on a taut string
        sim.step()
        assert sim.s.mode == Mode.IN_CUP
        Es.append(energy())
    assert sim.s.cup_s > 0.005 and sim.s.cup_gap == 0.0          # arm falls too (motors off)
    assert sim.s.cup_sd == pytest.approx(sim.s.sigma * r * sim.s.omega, rel=1e-9)
    assert max(abs(e - E0) for e in Es) < 1e-7


# ----------------------------------------------------- test 4: magnet unchanged
def test_magnet_mode_is_the_default_and_unchanged():
    assert Params.load(hand="magnet") == Params.load()
    p = Params.load()
    assert p.catch_mode == "magnet"
    sim = Simulator(p, rng=np.random.default_rng(0))
    assert isinstance(sim.catch, MagnetCatch)
    assert sim.c_stow == ad.link_constants(p, stowed=True)      # no spin inertia added
    with pytest.raises(ValueError):
        sim.reset_in_cup()
    with pytest.raises(ValueError):
        Params.load(catch_mode="glue")
