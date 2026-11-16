"""Rigid-body dynamics of the 2-DOF arm (PHYSICS_SPEC §5.2).

    M(q) qdd + C(q, qd) qd + G(q) = Q_drive + Q_fric + Jac^T F_H,   q = (q1, phi2)

Closed-form Lagrangian terms; ``arm/derive_sympy.py`` re-derives them
symbolically and tests/test_arm.py checks both agree.

Bodies (§5.2):
  link 1  : uniform rod m1 (COM l1/2), elbow point mass m_e at E,
            reflected joint-1 rotor N1^2 Jr1 on q1.
  link 2  : adapter rod m2 (COM l2/2), funnel point mass mc at lc plus Ic,
            GM3506 rotor Jr2 (direct drive, absolute angle = phi2),
            and when STOWED the yo-yo as a point mass m at l2 + ell_min.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sim.params import Params


@dataclass(frozen=True)
class LinkConstants:
    J1: float   # inertia of the link-1 group about joint 1 (incl. N1^2 Jr1)
    M2: float   # total mass carried by link 2
    S2: float   # first moment of link 2 about the wrist axis
    J2: float   # inertia of link 2 about the wrist axis (incl. Jr2)
    G1: float   # gravity coefficient: G1(q) = G1 * cos q1
    l1: float
    g: float


def link_constants(p: Params, stowed: bool, yoyo_spin: bool = False) -> LinkConstants:
    """``yoyo_spin``: the stowed yo-yo also turns with link 2 (flip catch, §4.3),
    so its spin inertia I adds to J2. The magnet model leaves it out."""
    l1, l2 = p.l1, p.l2
    M2 = p.m2 + p.mc
    S2 = p.m2 * l2 / 2 + p.mc * p.lc
    J2 = p.m2 * l2**2 / 3 + p.mc * p.lc**2 + p.Ic + p.Jr2
    if stowed:  # §4.1 the stowed yo-yo is a point mass on link 2
        d = l2 + p.ell_min
        M2 += p.m
        S2 += p.m * d
        J2 += p.m * d * d
        if yoyo_spin:
            J2 += p.I
    J1 = p.m1 * l1**2 / 3 + p.m_e * l1**2 + p.N1**2 * p.Jr1
    G1 = p.g * (p.m1 * l1 / 2 + p.m_e * l1 + M2 * l1)
    return LinkConstants(J1=J1, M2=M2, S2=S2, J2=J2, G1=G1, l1=l1, g=p.g)


def mass_matrix(c: LinkConstants, q1: float, phi2: float):
    """M(q) as (M11, M12, M22)."""
    m12 = c.S2 * c.l1 * math.cos(q1 - phi2)
    return c.J1 + c.M2 * c.l1**2, m12, c.J2


def bias_forces(c: LinkConstants, q1, phi2, q1d, phi2d):
    """C(q, qd) qd + G(q) as (h1, h2)."""
    k = c.S2 * c.l1 * math.sin(q1 - phi2)
    h1 = k * phi2d**2 + c.G1 * math.cos(q1)
    h2 = -k * q1d**2 + c.g * c.S2 * math.cos(phi2)
    return h1, h2


def gravity(c: LinkConstants, q1, phi2):
    return c.G1 * math.cos(q1), c.g * c.S2 * math.cos(phi2)


def energy(c: LinkConstants, q1, phi2, q1d, phi2d) -> float:
    """Kinetic + potential energy of the arm (zero potential at y = 0)."""
    m11, m12, m22 = mass_matrix(c, q1, phi2)
    ke = 0.5 * (m11 * q1d**2 + 2 * m12 * q1d * phi2d + m22 * phi2d**2)
    pe = c.G1 * math.sin(q1) + c.g * c.S2 * math.sin(phi2)
    return ke + pe


def drive_forces(p: Params, tau_m1: float, tau_m2: float):
    """Q_drive (§5.2): the wrist motor acts between link 1 and link 2."""
    return p.N1 * p.eta1 * tau_m1 - tau_m2, tau_m2


def smooth_friction(b: float, tc: float, v: float) -> float:
    """f(v) = b v + tc tanh(v / 0.01)  (§5.2)."""
    return b * v + tc * math.tanh(v / 0.01)


def friction_forces(p: Params, q1d: float, phi2d: float):
    """Q_fric (§5.2) in (q1, phi2) coordinates; wrist friction acts on q2."""
    q2d = phi2d - q1d
    f1 = smooth_friction(p.b1, p.tc1, q1d)
    f2 = smooth_friction(p.b2, p.tc2, q2d)
    return -f1 + f2, -f2


def solve2(m11, m12, m22, b1, b2):
    """Solve the symmetric 2x2 system M x = b."""
    det = m11 * m22 - m12 * m12
    return (m22 * b1 - m12 * b2) / det, (m11 * b2 - m12 * b1) / det


def inv2(m11, m12, m22):
    det = m11 * m22 - m12 * m12
    return m22 / det, -m12 / det, m11 / det
