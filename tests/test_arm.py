"""PHYSICS_SPEC §9 tests 9-11 (arm statics, Jacobian, torque mapping, energy)."""
import math

import numpy as np
import pytest

from arm import dynamics as ad
from arm import kinematics as ak
from sim.params import Params
from sim.simulator import Simulator
from yoyo.modes import Mode

POSES = [(0.0, -math.pi / 2), (0.4, -1.2), (-0.7, 0.3), (1.2, 2.5)]


@pytest.mark.parametrize("stowed", [False, True])
@pytest.mark.parametrize("q1,phi2", POSES)
def test_gravity_by_hand(q1, phi2, stowed):
    """Test 9a: G(q) equals the sum of point-mass gravity moments."""
    p = Params.load()
    c = ad.link_constants(p, stowed)
    g = p.g
    E = np.array(ak.elbow(p.l1, q1))
    a = np.array([math.cos(phi2), math.sin(phi2)])
    link1 = [(p.m1, E / 2), (p.m_e, E)]
    link2 = [(p.m2, E + a * p.l2 / 2), (p.mc, E + a * p.lc)]
    if stowed:
        link2.append((p.m, E + a * (p.l2 + p.ell_min)))
    # generalized gravity force = sum m g dy/dq
    # link-1 bodies: dy/dq1 = x ; link-2 bodies ride on E: dy/dq1 = E_x, dy/dphi2 = x - E_x
    G1 = sum(m * g * pos[0] for m, pos in link1) + sum(m * g * E[0] for m, _ in link2)
    G2 = sum(m * g * (pos[0] - E[0]) for m, pos in link2)
    g1, g2 = ad.gravity(c, q1, phi2)
    assert g1 == pytest.approx(G1, rel=1e-12)
    assert g2 == pytest.approx(G2, rel=1e-12)


@pytest.mark.parametrize("q1,phi2", POSES)
def test_jacobian_numeric(q1, phi2):
    """Test 9b: Jac matches finite differences of H."""
    p = Params.load()
    J = np.array(ak.hook_jacobian(p.l1, p.l2, q1, phi2))
    eps = 1e-7
    H0 = np.array(ak.hook(p.l1, p.l2, q1, phi2))
    Jn = np.column_stack([
        (np.array(ak.hook(p.l1, p.l2, q1 + eps, phi2)) - H0) / eps,
        (np.array(ak.hook(p.l1, p.l2, q1, phi2 + eps)) - H0) / eps,
    ])
    assert np.allclose(J, Jn, atol=1e-6)


@pytest.mark.parametrize("stowed", [False, True])
def test_matches_sympy(stowed):
    """Closed-form M and C qd + G agree with the symbolic Lagrangian derivation."""
    from arm.derive_sympy import numeric_functions
    f = numeric_functions()
    p = Params.load()
    c = ad.link_constants(p, stowed)
    vals = dict(l1=p.l1, l2=p.l2, m1=p.m1, m2=p.m2, mc=p.mc, lc=p.lc, Ic=p.Ic, m_e=p.m_e,
                N1=p.N1, Jr1=p.Jr1, Jr2=p.Jr2, g=p.g,
                m_y=p.m if stowed else 0.0, d_y=p.l2 + p.ell_min)
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-3, 3, 2)
        qd = rng.uniform(-10, 10, 2)
        Ms, hs = f(vals, q, qd)
        m11, m12, m22 = ad.mass_matrix(c, *q)
        h1, h2 = ad.bias_forces(c, *q, *qd)
        assert np.allclose(np.array(Ms, dtype=float), [[m11, m12], [m12, m22]], rtol=1e-10, atol=1e-14)
        assert np.allclose(np.array(hs, dtype=float).ravel(), [h1, h2], rtol=1e-10, atol=1e-12)


def test_relative_torque_mapping():
    """Test 10: tau_m2 q2_dot = Q_drive . (q1_dot, phi2_dot); with q1 locked only
    phi2 accelerates and link 1 receives -tau_m2."""
    p = Params.load(g=0.0)
    rng = np.random.default_rng(1)
    for _ in range(5):
        tau2 = rng.uniform(-0.1, 0.1)
        q1d, phi2d = rng.uniform(-5, 5, 2)
        Q = ad.drive_forces(p, 0.0, tau2)
        assert tau2 * (phi2d - q1d) == pytest.approx(Q[0] * q1d + Q[1] * phi2d)
        assert Q[0] == pytest.approx(-tau2)
    c = ad.link_constants(p, stowed=False)
    tau2 = 0.05
    m11, m12, m22 = ad.mass_matrix(c, 0.3, -1.0)
    Q1, Q2 = ad.drive_forces(p, 0.0, tau2)
    phi2dd = Q2 / m22                       # q1 locked, at rest, no gravity
    lock = m12 * phi2dd - Q1                # torque joint 1 must supply to stay locked
    assert phi2dd == pytest.approx(tau2 / c.J2)
    assert Q1 == pytest.approx(-tau2)
    assert lock == pytest.approx(m12 * phi2dd + tau2)


def test_arm_energy_conservation():
    """Test 11: no friction, no input, no yo-yo -> energy conserved."""
    p = Params.load(b1=0.0, b2=0.0, tc1=0.0, tc2=0.0)
    sim = Simulator(p, sensor_noise=False, log=False)
    sim.reset_stowed()
    sim.s.mode = int(Mode.FAIL)       # arm alone
    sim._tau = (0.0, 0.0)
    y = [0.3, 0.5, 0.0, 0.0]
    c = sim.c_free
    E0 = ad.energy(c, *y)
    t = 0.0
    E = []
    for _ in range(10000):           # 2 s
        y = sim._rk4(t, y, p.dt, Mode.FAIL)
        t += p.dt
        E.append(ad.energy(c, *y))
    swing = max(abs(e - E0) for e in E)
    assert swing / (c.G1 + abs(c.g * c.S2)) < 1e-8


def test_static_hold_stowed():
    """The arm holds the stowed yo-yo at rest with currents within limits."""
    p = Params.load()
    sim = Simulator(p, sensor_noise=False)
    sim.reset_stowed()
    q0 = (sim.s.q1, sim.s.phi2)
    sim.run(0.5)
    assert sim.s.mode == Mode.STOWED
    assert abs(sim.s.q1 - q0[0]) < 2e-3 and abs(sim.s.phi2 - q0[1]) < 2e-3
    assert abs(sim.motor.i1) < p.i_max1


# ------------------------------------------------------------ flip hand, l2 < 0 (§4.4)
@pytest.mark.parametrize("stowed", [False, True])
@pytest.mark.parametrize("q1,phi2", POSES)
def test_flip_hand_l2_negative_gravity_and_jacobian(q1, phi2, stowed):
    """§4.4: the §5 formulas are unchanged for the flip hand, l2 = -ell_min < 0.
    Gravity = sum of point-mass moments (= dV/dq), Jacobian = finite differences."""
    p = Params.load(hand="flip")
    assert p.l2 < 0
    c = ad.link_constants(p, stowed, yoyo_spin=True)
    g = p.g
    E = np.array(ak.elbow(p.l1, q1))
    a = np.array([math.cos(phi2), math.sin(phi2)])
    link1 = [(p.m1, E / 2), (p.m_e, E)]
    link2 = [(p.m2, E + a * p.l2 / 2), (p.mc, E + a * p.lc)]
    if stowed:
        link2.append((p.m, E + a * (p.l2 + p.ell_min)))
    G1 = sum(m * g * pos[0] for m, pos in link1) + sum(m * g * E[0] for m, _ in link2)
    G2 = sum(m * g * (pos[0] - E[0]) for m, pos in link2)
    g1, g2 = ad.gravity(c, q1, phi2)
    assert g1 == pytest.approx(G1, rel=1e-12)
    assert g2 == pytest.approx(G2, rel=1e-12, abs=1e-15)
    eps = 1e-6
    pe = lambda q1_, f2_: ad.energy(c, q1_, f2_, 0.0, 0.0)
    assert g1 == pytest.approx((pe(q1 + eps, phi2) - pe(q1 - eps, phi2)) / (2 * eps), rel=1e-7)
    assert g2 == pytest.approx((pe(q1, phi2 + eps) - pe(q1, phi2 - eps)) / (2 * eps), rel=1e-6, abs=1e-12)
    J = np.array(ak.hook_jacobian(p.l1, p.l2, q1, phi2))
    H0 = np.array(ak.hook(p.l1, p.l2, q1, phi2))
    Jn = np.column_stack([
        (np.array(ak.hook(p.l1, p.l2, q1 + 1e-7, phi2)) - H0) / 1e-7,
        (np.array(ak.hook(p.l1, p.l2, q1, phi2 + 1e-7)) - H0) / 1e-7,
    ])
    assert np.allclose(J, Jn, atol=1e-6)
    # H sits on the far side of the wrist axis from the funnel mouth (+a)
    assert (H0 - E) @ a == pytest.approx(p.l2)


@pytest.mark.parametrize("stowed", [False, True])
def test_flip_hand_matches_sympy(stowed):
    """The symbolic derivation declares l2 positive; check it still agrees for l2 < 0."""
    from arm.derive_sympy import numeric_functions
    f = numeric_functions()
    p = Params.load(hand="flip")
    c = ad.link_constants(p, stowed)
    vals = dict(l1=p.l1, l2=p.l2, m1=p.m1, m2=p.m2, mc=p.mc, lc=p.lc, Ic=p.Ic, m_e=p.m_e,
                N1=p.N1, Jr1=p.Jr1, Jr2=p.Jr2, g=p.g,
                m_y=p.m if stowed else 0.0, d_y=p.l2 + p.ell_min)
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.uniform(-3, 3, 2)
        qd = rng.uniform(-10, 10, 2)
        Ms, hs = f(vals, q, qd)
        m11, m12, m22 = ad.mass_matrix(c, *q)
        h1, h2 = ad.bias_forces(c, *q, *qd)
        assert np.allclose(np.array(Ms, dtype=float), [[m11, m12], [m12, m22]], rtol=1e-10, atol=1e-14)
        assert np.allclose(np.array(hs, dtype=float).ravel(), [h1, h2], rtol=1e-10, atol=1e-12)


def test_flip_hand_arm_energy_conservation():
    """Test 11 with l2 < 0 (arm alone, no friction, no input)."""
    p = Params.load(hand="flip", b1=0.0, b2=0.0, tc1=0.0, tc2=0.0)
    sim = Simulator(p, sensor_noise=False, log=False)
    sim.reset_stowed()
    sim.s.mode = int(Mode.FAIL)
    sim._tau = (0.0, 0.0)
    y = [0.3, 0.5, 0.0, 2.0]
    c = sim.c_free
    E0 = ad.energy(c, *y)
    t, swing = 0.0, 0.0
    for _ in range(10000):
        y = sim._rk4(t, y, p.dt, Mode.FAIL)
        t += p.dt
        swing = max(swing, abs(ad.energy(c, *y) - E0))
    assert swing / (c.G1 + abs(c.g * c.S2)) < 1e-8
