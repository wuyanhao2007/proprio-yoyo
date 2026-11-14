"""Yo-yo equations of motion and event maps (PHYSICS_SPEC §3).

Conventions (§0): u = (sin th, -cos th) points from H to the yo-yo centre,
u_th = (cos th, sin th); p = H + l u. Rolling constraint when taut:
l_dot = sigma r omega, sigma in {+1, -1} flips at every bottom.

These are pure functions of scalars; the arm coupling (simultaneous solve of
the string tension with the arm acceleration, §5.4) is in sim/simulator.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sim.params import Params


def string_dirs(theta: float):
    s, c = math.sin(theta), math.cos(theta)
    return (s, -c), (c, s)


def taut_tension_free_term(p: Params, ell, elld, theta, thetad, r, drdl,
                           F_mag, Fc_u, tau_c, sigma):
    """§3.2 split of the string tension:  T = T0 - mu * (Hdd . u),  mu = m (1 - kappa).

    A   = g cos th - Hdd.u + l thd^2 - F_mag/m + Fc.u/m      (Fc: wall contact on the yo-yo)
    rho = (r_dot / r) l_dot
    l_dd = kappa A + (1 - kappa) rho + sigma r tau_c / (I + m r^2) - c_l l_dot
    T   = m (A - l_dd)

    Derivation: m p_dd = -(T + F_mag) u + m g_vec + Fc ;  I w_dot = sigma T r + tau_c ;
    differentiate l_dot = sigma r w twice and eliminate w_dot. sigma drops out,
    so paying out and winding up use the same equation.
    Returns (A0, rho, kappa, mu, T0) where A0 = A + Hdd.u.
    """
    m, I = p.m, p.I
    mr2 = m * r * r
    kappa = mr2 / (I + mr2)
    rho = (drdl * elld / r) * elld
    A0 = p.g * math.cos(theta) + ell * thetad**2 - F_mag / m + Fc_u / m
    spin = sigma * r * tau_c / (I + mr2)
    mu = m * (1.0 - kappa)
    T0 = m * ((1.0 - kappa) * (A0 - rho) - spin + p.c_l * elld)
    return A0, rho, kappa, mu, T0


def taut_accels(p: Params, ell, elld, theta, thetad, r, sigma, tau_c,
                A0, rho, kappa, Hdd_u, Hdd_t, Fc_t):
    """§3.2 string and swing accelerations given the hook acceleration."""
    A = A0 - Hdd_u
    elldd = kappa * A + (1.0 - kappa) * rho + sigma * r * tau_c / (p.I + p.m * r * r) - p.c_l * elld
    if p.planar:
        thetadd = (-p.g * math.sin(theta) - Hdd_t - 2.0 * elld * thetad + Fc_t / p.m) / ell
    else:
        thetadd = 0.0
    return elldd, thetadd


@dataclass
class BottomResult:
    elld: float
    sigma: int
    omega: float
    J: float
    pulse_dt: float
    E_ratio: float


def bottom_event(p: Params, elld_minus: float, sigma: int, omega_minus: float) -> BottomResult:
    """§3.6 bottom: reverse along the string, flip sigma, keep spin direction."""
    v = elld_minus
    ze = p.zeta_e
    J = p.m * (1.0 + ze) * v           # total string impulse on the hook
    pulse_dt = math.pi * p.r0 / v       # half a turn around the axle
    return BottomResult(elld=-ze * v, sigma=-sigma, omega=ze * omega_minus,
                        J=J, pulse_dt=pulse_dt, E_ratio=ze * ze)


def bottom_pulse(J: float, pulse_dt: float, t: float) -> float:
    """§3.6 raised-cosine force magnitude on the hook; integrates to J."""
    if t < 0.0 or t > pulse_dt:
        return 0.0
    return (J / pulse_dt) * (1.0 - math.cos(2.0 * math.pi * t / pulse_dt))


def snap_impulse(p: Params, v_r: float, omega: float, sigma: int, r: float,
                 lam_uu: float):
    """§3.5 re-tensioning impulse. Returns (P, e_used) or (0, 0) when no impulse.

    Delta = v_r - sigma r omega (yo-yo outruns the paid-out string).
    m_eff = 1 / (1/m + r^2/I + u.Lambda_H^-1.u), P = (1 + e) m_eff Delta.
    """
    delta = v_r - sigma * r * omega
    if delta <= 0.0:
        return 0.0, 0.0
    # Below v_snap_min treat the snap as fully inelastic (spec: 1e-3 m/s). A larger
    # threshold stops wall contact from producing an endless train of micro-rebounds.
    e = p.e_taut if delta >= p.v_snap_min else 0.0
    m_eff = 1.0 / (1.0 / p.m + r * r / p.I + lam_uu)
    return (1.0 + e) * m_eff * delta, e
