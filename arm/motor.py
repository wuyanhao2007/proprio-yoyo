"""Motors and drivers (PHYSICS_SPEC §6).

Both motors run FOC current control (SimpleFOC Mini v2 / DRV8316); the current
loop is a first-order lag. Position mode runs a PD in the driver firmware at the
physics rate, on *measured* joint angles (§7).
Belt compliance (v4) is not implemented yet: the belt is rigid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim.params import Params


@dataclass
class MotorState:
    i1: float = 0.0
    i2: float = 0.0
    # cogging residual phases (domain randomization), one pair per motor
    cog_phase: tuple = (0.0, 0.0, 0.0, 0.0)
    # time spent above the continuous current (for heating estimates)
    t_over1: float = 0.0
    t_over2: float = 0.0


def pd_current_command(p: Params, q1_ref, q2_ref, q1_meas, q2_meas, q1d_meas, q2d_meas):
    """§6 position mode. Joint 1 gains are at the joint; wrist uses relative q2."""
    i1 = (p.Kp1 * (q1_ref - q1_meas) - p.Kd1 * q1d_meas) / (p.Kt1 * p.N1 * p.eta1)
    i2 = (p.Kp2 * (q2_ref - q2_meas) - p.Kd2 * q2d_meas) / p.Kt2
    return i1, i2


def current_limits(p: Params, q1d: float, q2d: float):
    """Current limits per motor as ((lo1, hi1), (lo2, hi2)). §6"""
    lo1, hi1 = -p.i_max1, p.i_max1
    lo2, hi2 = -p.i_max2, p.i_max2
    if p.use_back_emf:  # v4: |i| <= (V - Ke w_m) / R, Ke = Kt in SI units
        w1 = p.N1 * q1d
        w2 = q2d
        hi1 = min(hi1, (p.V_bus - p.Kt1 * w1) / p.R1)
        lo1 = max(lo1, (-p.V_bus - p.Kt1 * w1) / p.R1)
        hi2 = min(hi2, (p.V_lim2 - p.Kt2 * w2) / p.R2)
        lo2 = max(lo2, (-p.V_lim2 - p.Kt2 * w2) / p.R2)
    return (lo1, hi1), (lo2, hi2)


def step_currents(p: Params, s: MotorState, i_cmd1, i_cmd2, q1d, q2d, dt):
    """Exact discretization of  di/dt = (clip(i_cmd) - i) / tau_i."""
    (lo1, hi1), (lo2, hi2) = current_limits(p, q1d, q2d)
    c1 = min(max(i_cmd1, lo1), hi1)
    c2 = min(max(i_cmd2, lo2), hi2)
    k = 1.0 - math.exp(-dt / p.tau_i)
    s.i1 += k * (c1 - s.i1)
    s.i2 += k * (c2 - s.i2)
    if abs(s.i1) > p.tau_max1 / p.Kt1:
        s.t_over1 += dt
    if abs(s.i2) > p.tau_max2 / p.Kt2:
        s.t_over2 += dt


def motor_torques(p: Params, s: MotorState, theta_m1: float, theta_m2: float):
    """tau_m = Kt i, plus the cogging residual used for domain randomization (§7)."""
    t1 = p.Kt1 * s.i1
    t2 = p.Kt2 * s.i2
    if p.cogging_amp > 0.0:
        a, b, c, d = s.cog_phase
        t1 += p.cogging_amp * p.tau_max1 * (math.sin(11 * theta_m1 + a) + 0.5 * math.sin(22 * theta_m1 + b))
        t2 += p.cogging_amp * p.tau_max2 * (math.sin(7 * theta_m2 + c) + 0.5 * math.sin(14 * theta_m2 + d))
    return t1, t2


def random_cogging_phases(rng: np.random.Generator):
    return tuple(float(x) for x in rng.uniform(0, 2 * math.pi, size=4))
