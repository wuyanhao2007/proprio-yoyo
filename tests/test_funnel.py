"""PHYSICS_SPEC §9 tests 7-8 (throw and capture) + funnel geometry (§4.3)."""
import math

import numpy as np
import pytest

from sim.params import Params
from sim.simulator import KinematicHook, Simulator
from yoyo.funnel import FunnelGeometry, magnet_force
from yoyo.modes import Mode


# ------------------------------------------------------------------ geometry
def test_profile_shape():
    p = Params.load()
    f = FunnelGeometry(p)
    assert f.half_width(0.0) == pytest.approx(p.seat_width / 2)
    assert f.half_width(p.seat_depth) == pytest.approx(p.seat_width / 2)
    assert f.half_width(p.d_cup) == pytest.approx(p.mouth_width / 2)
    assert f.clearance == pytest.approx(p.seat_width / 2 - p.R)
    # trumpet: tangent to the seat wall, widening monotonically
    assert f.wall_slope(p.seat_depth + 1e-6) == pytest.approx(0.0, abs=1e-3)
    assert np.all(np.diff(f.w) >= 0)
    cone = FunnelGeometry(p.replace(flare_exponent=1.0))
    assert cone.wall_slope(p.seat_depth + 1e-6) > 0.3  # a cone has a corner at the seat


def test_contact_normals():
    p = Params.load()
    f = FunnelGeometry(p)
    assert f.contacts(0.0, p.ell_min) == []                    # seated, centred: no contact
    c = f.contacts(f.ws - p.R + 0.001, 0.02)                   # pushed 1 mm into the seat wall
    assert len(c) == 1 and c[0].pen == pytest.approx(0.001, abs=1e-9)
    assert c[0].nx == pytest.approx(-1.0) and c[0].nz == pytest.approx(0.0, abs=1e-9)
    z = 0.5 * (p.seat_depth + p.d_cup) + p.R * 0.3              # on the flare: normal also points out
    x = f.half_width(z) - p.R * 0.95
    c = f.contacts(x, z)
    assert c and c[0].nx < 0 and c[0].nz > 0
    assert f.outside_hit(f.wm + 0.002, p.d_cup + p.R - 0.001)
    assert not f.outside_hit(0.0, p.d_cup + p.R - 0.001)


def test_magnet_profile():
    p = Params.load()
    assert magnet_force(p, p.ell_min) == pytest.approx(p.F_mag0)
    assert magnet_force(p, p.ell_min + p.d_mag) == pytest.approx(p.F_mag0 / 4)


# ------------------------------------------------------------------- throw §4.1
@pytest.mark.parametrize("a_up,released", [(0.0, False), (9.5, False), (11.0, True)])
def test_release_threshold(a_up, released):
    """Test 7: funnel down, release iff upward accel > F_mag0/m - g (~10.2 m/s^2)."""
    p = Params.load()
    assert p.F_mag0 / p.m - p.g == pytest.approx(10.19, abs=0.01)
    sim = Simulator(p, hook=KinematicHook.const_accel(a_up), sensor_noise=False)
    sim.reset_stowed()
    sim.run(0.05)
    types = [e["type"] for e in sim.events]
    assert ("release" in types) == released
    if not released:
        assert sim.s.mode == Mode.STOWED


# ---------------------------------------------------------- capture §4.2 gate
def _climb(model, theta0=0.0, v_seat=0.2, ell0=0.06, **over):
    """Yo-yo climbing on the funnel axis (fixed hook), arriving at the seat at
    about v_seat (ignoring the magnet)."""
    p = Params.load(capture_model=model, **over)
    a = p.kappa() * p.g
    v0 = math.sqrt(v_seat**2 + 2 * a * (ell0 - p.ell_min))
    sim = Simulator(p, hook=KinematicHook.fixed(), sensor_noise=False)
    sim.reset_taut(ell=ell0, elld=-v0, theta=theta0, sigma=-1)
    sim.s.inside = ell0 < p.ell_mouth
    return p, sim


def test_gate_slow_stows():
    p, sim = _climb("gate", v_seat=0.2)
    sim.run(0.3)
    assert sim.s.mode == Mode.STOWED
    assert any(e["type"] == "stowed" for e in sim.events)


def test_gate_fast_rebounds():
    p, sim = _climb("gate", v_seat=0.9, F_mag0=1e-9)
    sim.run(0.1)
    ev = next(e for e in sim.events if e["type"] == "seat_rebound")
    assert ev["v"] > p.v_cap_max
    # rebound speed = e_cup * v
    k = next(i for i, e in enumerate(sim.events) if e["type"] == "seat_rebound")
    assert sim.events[k]["e_cup"] == p.e_cup


def test_gate_misaligned_fails():
    p = Params.load(capture_model="gate")
    sim = Simulator(p, hook=KinematicHook.fixed(), sensor_noise=False)
    sim.reset_taut(ell=p.ell_mouth + 0.005, elld=-0.6, theta=math.radians(15), sigma=-1)
    sim.s.inside = False
    sim.run(0.1)
    assert sim.s.mode == Mode.FAIL and sim.s.fail_reason == "misaligned"


def test_magnet_only_inside():
    """After a throw the yo-yo leaves with inside = False: no magnet pull."""
    p = Params.load()
    sim = Simulator(p, hook=KinematicHook.fixed(), sensor_noise=False)
    sim.reset_taut(ell=p.ell_min + 0.002, elld=0.3)
    sim.s.inside = False
    sim.step()
    assert sim.diag.F_mag == 0.0
    sim.s.inside = True
    sim.step()
    assert sim.diag.F_mag > 0.0


# ------------------------------------------------------ capture §4.3 geometry
def test_geometry_centred_slow_stows():
    p, sim = _climb("geometry", v_seat=0.2)
    sim.run(0.3)
    assert sim.s.mode == Mode.STOWED


def test_geometry_wall_guides_offset_yoyo():
    """Off-axis entry inside the mouth: the wall pushes it to the axis and it stows."""
    p = Params.load(capture_model="geometry", mu_wall=0.1)
    sim = Simulator(p, hook=KinematicHook.fixed(), sensor_noise=False)
    ell0 = p.ell_mouth + 0.01
    th0 = math.atan2(0.012, ell0)   # 12 mm lateral offset at the mouth
    sim.reset_taut(ell=ell0, elld=-0.55, theta=th0, sigma=-1)
    sim.s.inside = False
    sim.run(0.4)
    d = sim.log.arrays()
    assert d["contact"].sum() > 0, "expected wall contact"
    assert sim.s.mode == Mode.STOWED, [e["type"] for e in sim.events]


def test_geometry_outside_mouth_fails():
    p = Params.load(capture_model="geometry")
    sim = Simulator(p, hook=KinematicHook.fixed(), sensor_noise=False)
    x0, z0 = 0.045, p.d_cup + p.R + 0.003   # 45 mm off-axis (mouth half-width 40 mm),
    ell0 = math.hypot(x0, z0)                # disc top 3 mm below the mouth plane
    sim.reset_taut(ell=ell0, elld=-1.0, theta=math.atan2(x0, z0), sigma=-1)
    sim.s.inside = False
    sim.run(0.2)
    assert sim.s.mode == Mode.FAIL and sim.s.fail_reason == "hit_rim"
