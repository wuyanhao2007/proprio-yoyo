"""Coupled arm + yo-yo hybrid simulator (numpy reference implementation).

PHYSICS_SPEC §2 (modes/events), §3 (yo-yo), §4 (funnel), §5 (arm), §6 (motors),
§7 (sensors). One ``step()`` = one physics step dt (default 5 kHz), RK4 inside a
mode, events located by linear interpolation of the event function (at most one
interpolated event per step, §2), followed by step-end checks.

Coupling (§5.4): the string tension and the hook acceleration are solved
simultaneously every derivative evaluation (they are linearly coupled through
the arm's operational-space inertia), instead of lagging F_H by one step. This
is the "solve jointly" option of §5.4 and avoids the lag oscillation.

The hook can instead follow a prescribed trajectory (``KinematicHook``), which
is what the v0 unit tests use.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from arm import dynamics as ad
from arm import kinematics as ak
from arm import motor as am
from arm.sensors import Sensors
from sim.log import SimLog
from sim.params import Params
from yoyo import dynamics as yd
from yoyo.funnel import FunnelGeometry, magnet_force, wall_damping
from yoyo.modes import Mode

TWO_PI = 2.0 * math.pi
SLACK_T = -0.01          # §2 slack hysteresis (N)


# ----------------------------------------------------------------- hook source
class KinematicHook:
    """Prescribed hook motion: fn(t) -> (Hx, Hy, Hdx, Hdy, Hddx, Hddy, phi2, phi2d)."""

    def __init__(self, fn: Callable[[float], tuple]):
        self.fn = fn

    def eval(self, t: float):
        return self.fn(t)

    @staticmethod
    def fixed(Hx=0.0, Hy=0.0, phi2=-math.pi / 2):
        return KinematicHook(lambda t: (Hx, Hy, 0.0, 0.0, 0.0, 0.0, phi2, 0.0))

    @staticmethod
    def const_accel(ay: float, Hy0=0.0, vy0=0.0, phi2=-math.pi / 2):
        return KinematicHook(lambda t: (0.0, Hy0 + vy0 * t + 0.5 * ay * t * t, 0.0,
                                        vy0 + ay * t, 0.0, ay, phi2, 0.0))


# ----------------------------------------------------------------------- state
@dataclass
class SimState:
    t: float = 0.0
    mode: int = int(Mode.STOWED)
    # arm, dynamics coordinates (q1, phi2); q2 = phi2 - q1
    q1: float = 0.0
    phi2: float = -math.pi / 2
    q1d: float = 0.0
    phi2d: float = 0.0
    # yo-yo, TAUT coordinates
    ell: float = 0.0
    elld: float = 0.0
    theta: float = 0.0
    thetad: float = 0.0
    # yo-yo, Cartesian (always kept up to date)
    px: float = 0.0
    py: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    spin_angle: float = 0.0
    sigma: int = 1
    ell_s: float = 0.0
    inside: bool = False
    # bottom force pulse on the hook (§3.6)
    pulse_t0: float = -1.0
    pulse_J: float = 0.0
    pulse_dt: float = 0.0
    pulse_ux: float = 0.0
    pulse_uy: float = 0.0
    # references currently applied by the driver firmware
    q1_ref: float = 0.0
    q2_ref: float = -math.pi / 2
    # bookkeeping
    n_bottom: int = 0
    released: bool = False
    caught: bool = False
    t_caught: float = -1.0
    fail_reason: str = ""

    @property
    def q2(self) -> float:
        return self.phi2 - self.q1

    @property
    def q2d(self) -> float:
        return self.phi2d - self.q1d


@dataclass
class Diag:
    """Quantities evaluated at the end of a step (for logs, events and info)."""
    T: float = 0.0
    F_mag: float = 0.0
    Fhx: float = 0.0
    Fhy: float = 0.0
    Fbx: float = 0.0
    Fby: float = 0.0
    Fcx: float = 0.0
    Fcy: float = 0.0
    Hx: float = 0.0
    Hy: float = 0.0
    Hdx: float = 0.0
    Hdy: float = 0.0
    Hddx: float = 0.0
    Hddy: float = 0.0
    q1dd: float = 0.0
    phi2dd: float = 0.0
    tau_m1: float = 0.0
    tau_m2: float = 0.0
    tau_wrist_rope: float = 0.0
    delta: float = 0.0
    F_in: float = 0.0
    contact: bool = False


# ------------------------------------------------------------------- simulator
class Simulator:
    def __init__(self, params: Params, hook: KinematicHook | None = None,
                 rng: np.random.Generator | None = None, sensor_noise: bool = True,
                 log: bool = True):
        self.p = params
        self.hook = hook
        self.kin = hook is not None
        self.rng = rng if rng is not None else np.random.default_rng()
        self.funnel = FunnelGeometry(params) if params.capture_model != "none" else None
        self.c_free = ad.link_constants(params, stowed=False)
        self.c_stow = ad.link_constants(params, stowed=True)
        self.c_wall = wall_damping(params, params.m)
        self.s = SimState()
        self.motor = am.MotorState(cog_phase=am.random_cogging_phases(self.rng))
        self.sensors = Sensors(params, self.rng, noise=sensor_noise)
        self.cmd_queue: deque = deque()
        self.events: list[dict] = []
        self.log = SimLog() if log else None
        self.diag = Diag()
        self._tau = (0.0, 0.0)
        self._prev_d = 0.0          # last H-yo-yo distance seen by the mouth check

    # ======================================================= initial conditions
    def reset_stowed(self, q1_ref: float = 0.0, q2_ref: float = -math.pi / 2):
        """Arm holding the stowed yo-yo at rest at (approximately) the reference."""
        s = self.s = SimState(mode=int(Mode.STOWED), q1_ref=q1_ref, q2_ref=q2_ref)
        self._place_arm_static(q1_ref, q2_ref, hanging=False)
        self._sync_stowed_yoyo()
        self._finish_reset()
        return s

    def reset_taut(self, ell: float, elld: float = 0.0, theta: float = 0.0,
                   thetad: float = 0.0, sigma: int = 1,
                   q1_ref: float = 0.0, q2_ref: float = -math.pi / 2):
        """Yo-yo on a taut string (e.g. mid-flight), arm holding still."""
        s = self.s = SimState(mode=int(Mode.TAUT), q1_ref=q1_ref, q2_ref=q2_ref)
        self._place_arm_static(q1_ref, q2_ref, hanging=True)
        s.ell, s.elld, s.theta, s.thetad, s.sigma = ell, elld, theta, thetad, sigma
        s.released = True
        H, Hd, _ = self._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
        self._sync_taut_cartesian(H, Hd)
        if self.funnel is not None and ell < self.p.ell_mouth:
            s.inside = True
        self._finish_reset()
        return s

    def _place_arm_static(self, q1_ref, q2_ref, hanging: bool):
        s, p = self.s, self.p
        if self.kin:
            H = self.hook.eval(0.0)
            s.phi2, s.phi2d = H[6], H[7]
            return
        # static load torques; PD has no integrator, so offset q by tau / Kp
        c = self.c_free if hanging else self.c_stow
        q1, phi2 = q1_ref, q1_ref + q2_ref
        for _ in range(3):
            g1, g2 = ad.gravity(c, q1, phi2)
            if hanging:  # string tension ~ m g (1 - kappa) straight down at H
                T = p.m * p.g * (1 - p.kappa())
                (j00, j01), (j10, j11) = ak.hook_jacobian(p.l1, p.l2, q1, phi2)
                g1 += T * j10
                g2 += T * j11
            # wrist: tau2 = g2 ; joint 1: N eta tau1 - tau2 = g1
            tau2 = g2
            tau1 = (g1 + tau2) / (p.N1 * p.eta1)
            q1 = q1_ref - p.N1 * p.eta1 * tau1 / p.Kp1
            q2 = q2_ref - tau2 / p.Kp2
            phi2 = q1 + q2
        s.q1, s.phi2 = q1, phi2
        self.motor.i1 = tau1 / p.Kt1
        self.motor.i2 = tau2 / p.Kt2

    def _finish_reset(self):
        s = self.s
        self._prev_d = s.ell if s.mode == Mode.TAUT else self.p.ell_min
        self.events = []
        self.cmd_queue.clear()
        if self.log is not None:
            self.log.clear()
        self.sensors.reset(s.q1, s.q2, self.motor.i1, self.motor.i2)
        self.diag = self._diagnostics(s.t, self._pack(), s.mode)
        if self.log is not None:
            self.log.record(s, self.diag, self.motor, self.sensors)

    # ================================================================= commands
    def set_reference(self, q1_ref: float, q2_ref: float, delay: float | None = None):
        """Queue a joint reference; it reaches the firmware after cmd_delay."""
        d = self.p.cmd_delay if delay is None else delay
        self.cmd_queue.append((self.s.t + d - 1e-12, q1_ref, q2_ref))

    # ================================================================ kinematics
    def _hook_kin(self, t, q1, phi2, q1d, phi2d):
        p = self.p
        if self.kin:
            Hx, Hy, Hdx, Hdy, _, _, f2, f2d = self.hook.eval(t)
            return (Hx, Hy), (Hdx, Hdy), (f2, f2d)
        Hx, Hy = ak.hook(p.l1, p.l2, q1, phi2)
        (j00, j01), (j10, j11) = ak.hook_jacobian(p.l1, p.l2, q1, phi2)
        return (Hx, Hy), (j00 * q1d + j01 * phi2d, j10 * q1d + j11 * phi2d), (phi2, phi2d)

    def _sync_taut_cartesian(self, H, Hd):
        s = self.s
        (ux, uy), (tx, ty) = yd.string_dirs(s.theta)
        s.px, s.py = H[0] + s.ell * ux, H[1] + s.ell * uy
        s.vx = Hd[0] + s.elld * ux + s.ell * s.thetad * tx
        s.vy = Hd[1] + s.elld * uy + s.ell * s.thetad * ty
        r, _ = self.p.radius(s.ell)
        s.omega = s.sigma * s.elld / r

    def _stow_point(self, q1, phi2):
        """Seat position of the yo-yo centre: H + ell_min a (§4.1)."""
        p = self.p
        if self.kin:
            Hx, Hy, *_ , f2, _ = self.hook.eval(self.s.t)
            return Hx + p.ell_min * math.cos(f2), Hy + p.ell_min * math.sin(f2)
        return ak.point_on_link2(p.l1, q1, phi2, p.l2 + p.ell_min)

    def _sync_stowed_yoyo(self):
        s, p = self.s, self.p
        s.px, s.py = self._stow_point(s.q1, s.phi2)
        if self.kin:
            _, Hd, (f2, f2d) = self._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
            s.vx, s.vy = Hd
        else:
            s.vx, s.vy = ak.world_point_velocity(p.l1, s.q1, s.phi2, s.q1d, s.phi2d, s.px, s.py)
        s.ell, s.elld, s.omega = p.ell_min, 0.0, 0.0
        s.theta = s.phi2 + math.pi / 2
        s.thetad = s.phi2d

    # ================================================================ packing
    def _pack(self):
        s = self.s
        arm = [s.q1, s.phi2, s.q1d, s.phi2d]
        if s.mode == Mode.TAUT:
            return arm + [s.ell, s.elld, s.theta, s.thetad]
        if s.mode == Mode.FREE:
            return arm + [s.px, s.py, s.vx, s.vy, s.omega, s.ell_s]
        return arm

    def _unpack(self, y, mode, t):
        s = self.s
        s.t = t
        if not self.kin:
            s.q1, s.phi2, s.q1d, s.phi2d = y[0], y[1], y[2], y[3]
        else:
            _, _, (s.phi2, s.phi2d) = self._hook_kin(t, 0, 0, 0, 0)
        if mode == Mode.TAUT:
            s.ell, s.elld, s.theta, s.thetad = y[4], y[5], y[6], y[7]
            H, Hd, _ = self._hook_kin(t, s.q1, s.phi2, s.q1d, s.phi2d)
            self._sync_taut_cartesian(H, Hd)
        elif mode == Mode.FREE:
            s.px, s.py, s.vx, s.vy, s.omega, s.ell_s = y[4], y[5], y[6], y[7], y[8], y[9]
            # keep the TAUT coordinates meaningful for logs/plots while slack
            H, Hd, _ = self._hook_kin(t, s.q1, s.phi2, s.q1d, s.phi2d)
            rx, ry = s.px - H[0], s.py - H[1]
            d = math.hypot(rx, ry)
            if d > 1e-9:
                ux, uy = rx / d, ry / d
                s.ell, s.theta = d, math.atan2(ux, -uy)
                s.elld = (s.vx - Hd[0]) * ux + (s.vy - Hd[1]) * uy
                s.thetad = ((s.vx - Hd[0]) * -uy + (s.vy - Hd[1]) * ux) / d
        elif mode == Mode.STOWED:
            self._sync_stowed_yoyo()

    # ================================================================ dynamics
    def _pulse(self, t):
        s = self.s
        if s.pulse_t0 < 0.0:
            return 0.0, 0.0
        f = yd.bottom_pulse(s.pulse_J, s.pulse_dt, t - s.pulse_t0)
        return f * s.pulse_ux, f * s.pulse_uy

    def _slack_winding(self, ell_s: float, omega: float) -> float:
        """[A7] d(ell_s)/dt in FREE: the string is tied to the axle, so a spinning
        yo-yo keeps winding / paying out string while it is slack (no slip)."""
        p = self.p
        r, _ = p.radius(ell_s)
        rate = self.s.sigma * r * omega
        if (ell_s <= p.ell_min and rate < 0.0) or (ell_s >= p.L and rate > 0.0):
            return 0.0
        return rate

    def _wall_forces(self, H, Hd, phi2, phi2d, arm, px, py, vx, vy, omega):
        """§4.3 penalty contact of the yo-yo rim with the funnel wall.
        Returns force on the yo-yo, spin torque, generalized force on the arm,
        and the magnitude of the normal force."""
        f = self.funnel
        if f is None or self.p.capture_model != "geometry":
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        p = self.p
        ax, ay = math.cos(phi2), math.sin(phi2)
        nx_, ny_ = -ay, ax
        rx, ry = px - H[0], py - H[1]
        xl, zl = rx * nx_ + ry * ny_, rx * ax + ry * ay
        cons = f.contacts(xl, zl)
        if not cons:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        Fx = Fy = tau = Q1 = Q2 = 0.0
        q1, _, q1d, _ = arm
        for c in cons:
            Nx, Ny = c.nx * nx_ + c.nz * ax, c.nx * ny_ + c.nz * ay
            Cx, Cy = H[0] + c.cx * nx_ + c.cz * ax, H[1] + c.cx * ny_ + c.cz * ay
            if self.kin:
                wvx, wvy = Hd[0] - phi2d * (Cy - H[1]), Hd[1] + phi2d * (Cx - H[0])
            else:
                wvx, wvy = ak.world_point_velocity(p.l1, q1, phi2, q1d, phi2d, Cx, Cy)
            rpx, rpy = -p.R * Nx, -p.R * Ny                  # centre -> rim contact point
            rvx, rvy = vx - omega * rpy, vy + omega * rpx    # rim point velocity
            relx, rely = rvx - wvx, rvy - wvy
            vn = relx * Nx + rely * Ny
            Fn = p.k_wall * c.pen - self.c_wall * vn
            if Fn <= 0.0:
                continue
            tx, ty = -Ny, Nx
            vt = relx * tx + rely * ty
            Ff = -p.mu_wall * Fn * math.tanh(vt / 0.01)
            fx, fy = Fn * Nx + Ff * tx, Fn * Ny + Ff * ty
            Fx += fx
            Fy += fy
            tau += rpx * (Ff * ty) - rpy * (Ff * tx)
            if not self.kin:
                (j00, j01), (j10, j11) = ak.world_point_jacobian(p.l1, q1, phi2, Cx, Cy)
                Q1 -= j00 * fx + j10 * fy
                Q2 -= j01 * fx + j11 * fy
        return Fx, Fy, tau, Q1, Q2, 1.0

    def _arm_solve_terms(self, t, arm, stowed, Qc1, Qc2):
        """Returns (H, Hd, phi2, phi2d, a0, Lam, Minv, Jac, rhs) where
        Hdd = a0 + Lam F_hook and qdd = Minv (rhs + Jac^T F_hook)."""
        p = self.p
        if self.kin:
            Hx, Hy, Hdx, Hdy, Hddx, Hddy, f2, f2d = self.hook.eval(t)
            return ((Hx, Hy), (Hdx, Hdy), f2, f2d, (Hddx, Hddy), (0.0, 0.0, 0.0),
                    None, None, None)
        q1, f2, q1d, f2d = arm
        c = self.c_stow if stowed else self.c_free
        m11, m12, m22 = ad.mass_matrix(c, q1, f2)
        h1, h2 = ad.bias_forces(c, q1, f2, q1d, f2d)
        d1, d2 = ad.drive_forces(p, *self._tau)
        fr1, fr2 = ad.friction_forces(p, q1d, f2d)
        r1 = d1 + fr1 - h1 + Qc1
        r2 = d2 + fr2 - h2 + Qc2
        i11, i12, i22 = ad.inv2(m11, m12, m22)
        (j00, j01), (j10, j11) = ak.hook_jacobian(p.l1, p.l2, q1, f2)
        # qdd0 = Minv rhs ; a0 = Jac qdd0 + Jdot qd
        qa = i11 * r1 + i12 * r2
        qb = i12 * r1 + i22 * r2
        bx = -p.l1 * math.cos(q1) * q1d**2 - p.l2 * math.cos(f2) * f2d**2
        by = -p.l1 * math.sin(q1) * q1d**2 - p.l2 * math.sin(f2) * f2d**2
        a0 = (j00 * qa + j01 * qb + bx, j10 * qa + j11 * qb + by)
        # Lam = Jac Minv Jac^T  (symmetric: Lxx, Lxy, Lyy)
        A00 = j00 * i11 + j01 * i12
        A01 = j00 * i12 + j01 * i22
        A10 = j10 * i11 + j11 * i12
        A11 = j10 * i12 + j11 * i22
        Lam = (A00 * j00 + A01 * j01, A00 * j10 + A01 * j11, A10 * j10 + A11 * j11)
        H = ak.hook(p.l1, p.l2, q1, f2)
        Hd = (j00 * q1d + j01 * f2d, j10 * q1d + j11 * f2d)
        return H, Hd, f2, f2d, a0, Lam, (i11, i12, i22), ((j00, j01), (j10, j11)), (r1, r2)

    @staticmethod
    def _arm_qdd(Minv, Jac, rhs, Fx, Fy):
        if Minv is None:
            return 0.0, 0.0
        (i11, i12, i22), ((j00, j01), (j10, j11)), (r1, r2) = Minv, Jac, rhs
        b1 = r1 + j00 * Fx + j10 * Fy
        b2 = r2 + j01 * Fx + j11 * Fy
        return i11 * b1 + i12 * b2, i12 * b1 + i22 * b2

    def _deriv(self, t, y, mode, want_diag=False):
        p, s = self.p, self.s
        arm = y[0:4]
        Fbx, Fby = self._pulse(t)
        dg = Diag() if want_diag else None

        if mode == Mode.TAUT:
            ell, elld, th, thd = y[4], y[5], y[6], y[7]
            (ux, uy), (tx, ty) = yd.string_dirs(th)
            r, drdl = p.radius(ell)
            omega = s.sigma * elld / r
            H, Hd, (f2, f2d) = self._hook_kin(t, *arm)
            px, py = H[0] + ell * ux, H[1] + ell * uy
            vx = Hd[0] + elld * ux + ell * thd * tx
            vy = Hd[1] + elld * uy + ell * thd * ty
            Fcx, Fcy, tau_c, Qc1, Qc2, hit = self._wall_forces(H, Hd, f2, f2d, arm,
                                                               px, py, vx, vy, omega)
            Fm = magnet_force(p, ell) if s.inside else 0.0
            A0, rho, kappa, mu, T0 = yd.taut_tension_free_term(
                p, ell, elld, th, thd, r, drdl, Fm, Fcx * ux + Fcy * uy, tau_c, s.sigma)
            H, Hd, f2, f2d, a0, Lam, Minv, Jac, rhs = self._arm_solve_terms(
                t, arm, False, Qc1, Qc2)
            Lxx, Lxy, Lyy = Lam
            Lu = (Lxx * ux + Lxy * uy, Lxy * ux + Lyy * uy)
            lam_uu = ux * Lu[0] + uy * Lu[1]
            # Simultaneous solve (§5.4): T = T0 - mu Hdd_y.u, Hdd_y = a0 + Lam (T + Fm) u.
            # Hdd_y excludes the hook acceleration caused by the bottom pulse F_b: the
            # yo-yo has already been reversed instantaneously by that same impulse
            # (§3.6 amendment), so it must not feel it a second time (otherwise the
            # string goes "slack" ~3 ms after every bottom).
            T = (T0 - mu * (a0[0] * ux + a0[1] * uy + lam_uu * Fm)) / (1.0 + mu * lam_uu)
            Fhx, Fhy = (T + Fm) * ux + Fbx, (T + Fm) * uy + Fby
            Hddx = a0[0] + Lxx * Fhx + Lxy * Fhy
            Hddy = a0[1] + Lxy * Fhx + Lyy * Fhy
            Hyx = Hddx - (Lxx * Fbx + Lxy * Fby)
            Hyy = Hddy - (Lxy * Fbx + Lyy * Fby)
            q1dd, f2dd = self._arm_qdd(Minv, Jac, rhs, Fhx, Fhy)
            elldd, thdd = yd.taut_accels(p, ell, elld, th, thd, r, s.sigma, tau_c, A0, rho,
                                         kappa, Hyx * ux + Hyy * uy, Hyx * tx + Hyy * ty,
                                         Fcx * tx + Fcy * ty)
            if p.capture_model == "gate" and s.inside:
                thdd = f2dd  # §4.2 funnel wall holds the string on the funnel axis
            dy = [arm[2], arm[3], q1dd, f2dd, elld, elldd, thd, thdd]
            if self.kin:
                dy[0:4] = [0.0, 0.0, 0.0, 0.0]
            if want_diag:
                dg.T, dg.F_mag = T, Fm
                dg.Fhx, dg.Fhy, dg.Fcx, dg.Fcy, dg.contact = Fhx, Fhy, Fcx, Fcy, bool(hit)
                self._fill_diag(dg, H, Hd, Hddx, Hddy, q1dd, f2dd, Fbx, Fby, f2, ux, uy, Jac)
            return (dy, dg) if want_diag else dy

        if mode == Mode.FREE:
            px, py, vx, vy, om, ell_s = y[4], y[5], y[6], y[7], y[8], y[9]
            H, Hd, (f2, f2d) = self._hook_kin(t, *arm)
            Fcx, Fcy, tau_c, Qc1, Qc2, hit = self._wall_forces(H, Hd, f2, f2d, arm,
                                                               px, py, vx, vy, om)
            rx, ry = px - H[0], py - H[1]
            rn = math.hypot(rx, ry) or 1.0
            urx, ury = rx / rn, ry / rn
            # §4.2 amendment: the magnet keeps pulling a slack yo-yo that is inside
            Fm = magnet_force(p, rn) if s.inside else 0.0
            Fkx, Fky = Fbx + Fm * urx, Fby + Fm * ury       # force on the hook
            H, Hd, f2, f2d, a0, Lam, Minv, Jac, rhs = self._arm_solve_terms(
                t, arm, False, Qc1, Qc2)
            q1dd, f2dd = self._arm_qdd(Minv, Jac, rhs, Fkx, Fky)
            Lxx, Lxy, Lyy = Lam
            Hddx = a0[0] + Lxx * Fkx + Lxy * Fky
            Hddy = a0[1] + Lxy * Fkx + Lyy * Fky
            dy = [arm[2], arm[3], q1dd, f2dd, vx, vy,
                  (Fcx - Fm * urx) / p.m, (Fcy - Fm * ury) / p.m - p.g, tau_c / p.I,
                  self._slack_winding(ell_s, om)]
            if self.kin:
                dy[0:4] = [0.0, 0.0, 0.0, 0.0]
            if want_diag:
                dg.F_mag = Fm
                dg.Fhx, dg.Fhy, dg.Fcx, dg.Fcy, dg.contact = Fkx, Fky, Fcx, Fcy, bool(hit)
                self._fill_diag(dg, H, Hd, Hddx, Hddy, q1dd, f2dd, Fbx, Fby, f2, urx, ury, Jac)
            return (dy, dg) if want_diag else dy

        # STOWED (yo-yo merged into link 2) or FAIL (arm alone)
        stowed = mode == Mode.STOWED
        H, Hd, f2, f2d, a0, Lam, Minv, Jac, rhs = self._arm_solve_terms(t, arm, stowed, 0.0, 0.0)
        q1dd, f2dd = self._arm_qdd(Minv, Jac, rhs, Fbx, Fby)
        Lxx, Lxy, Lyy = Lam
        Hddx = a0[0] + Lxx * Fbx + Lxy * Fby
        Hddy = a0[1] + Lxy * Fbx + Lyy * Fby
        dy = [arm[2], arm[3], q1dd, f2dd] if not self.kin else [0.0, 0.0, 0.0, 0.0]
        if want_diag:
            dg.Fhx, dg.Fhy = Fbx, Fby
            self._fill_diag(dg, H, Hd, Hddx, Hddy, q1dd, f2dd, Fbx, Fby, f2,
                            math.cos(f2), math.sin(f2), Jac)
            if stowed:
                dg.F_in = self._stowed_pull(t, arm, q1dd, f2dd, f2, Hddx, Hddy)
        return (dy, dg) if want_diag else dy

    def _fill_diag(self, dg, H, Hd, Hddx, Hddy, q1dd, f2dd, Fbx, Fby, f2, ux, uy, Jac):
        dg.Hx, dg.Hy, dg.Hdx, dg.Hdy, dg.Hddx, dg.Hddy = H[0], H[1], Hd[0], Hd[1], Hddx, Hddy
        dg.q1dd, dg.phi2dd = q1dd, f2dd
        dg.Fbx, dg.Fby = Fbx, Fby
        dg.tau_m1, dg.tau_m2 = self._tau
        ax, ay = math.cos(f2), math.sin(f2)
        dg.delta = math.atan2(ax * uy - ay * ux, ax * ux + ay * uy)  # link 2 vs string
        if Jac is not None:  # rope torque about the wrist axis = Jac[:, 1] . F_H
            dg.tau_wrist_rope = Jac[0][1] * dg.Fhx + Jac[1][1] * dg.Fhy

    def _stowed_pull(self, t, arm, q1dd, f2dd, f2, Hddx, Hddy):
        """§4.1 inward pull the magnet must supply to keep the yo-yo seated."""
        p = self.p
        ax, ay = math.cos(f2), math.sin(f2)
        if self.kin:
            pax, pay = Hddx, Hddy
        else:
            q1, f2_, q1d, f2d = arm
            px, py = ak.point_on_link2(p.l1, q1, f2_, p.l2 + p.ell_min)
            (j00, j01), (j10, j11) = ak.world_point_jacobian(p.l1, q1, f2_, px, py)
            bx, by = ak.world_point_bias_accel(p.l1, q1, f2_, q1d, f2d, px, py)
            pax = j00 * q1dd + j01 * f2dd + bx
            pay = j10 * q1dd + j11 * f2dd + by
        Freq_x, Freq_y = p.m * pax, p.m * (pay + p.g)   # F_req = m (p_dd - g_vec)
        return -(Freq_x * ax + Freq_y * ay)

    def _rk4(self, t, y, h, mode):
        k1 = self._deriv(t, y, mode)
        y2 = [a + 0.5 * h * b for a, b in zip(y, k1)]
        k2 = self._deriv(t + 0.5 * h, y2, mode)
        y3 = [a + 0.5 * h * b for a, b in zip(y, k2)]
        k3 = self._deriv(t + 0.5 * h, y3, mode)
        y4 = [a + h * b for a, b in zip(y, k3)]
        k4 = self._deriv(t + h, y4, mode)
        return [a + h / 6.0 * (b + 2 * c + 2 * d + e)
                for a, b, c, d, e in zip(y, k1, k2, k3, k4)]

    def _diagnostics(self, t, y, mode) -> Diag:
        return self._deriv(t, y, mode, want_diag=True)[1]

    # ================================================================ events
    def _event_fn(self, t, y, mode):
        """Interpolated events (§2). Returns dict name -> g; event when g >= 0."""
        p, s = self.p, self.s
        out = {}
        if mode == Mode.TAUT:
            out["bottom"] = (y[4] - p.L, y[5] > 0.0)
            if self.funnel is not None:
                out["seat"] = (p.ell_min - y[4], y[5] < 0.0)
        elif mode == Mode.FREE:
            H, Hd, _ = self._hook_kin(t, *y[0:4])
            rx, ry = y[4] - H[0], y[5] - H[1]
            d = math.hypot(rx, ry)
            out["snap"] = (d - y[9], True)
            if self.funnel is not None:
                v_r = ((y[6] - Hd[0]) * rx + (y[7] - Hd[1]) * ry) / max(d, 1e-9)
                out["seat"] = (p.ell_min - d, v_r < 0.0)
        return out

    def _find_event(self, t0, y0, y1, mode, h):
        g0s = self._event_fn(t0, y0, mode)
        g1s = self._event_fn(t0 + h, y1, mode)
        for name, (g1, ok1) in g1s.items():
            g0, _ = g0s[name]
            if g1 >= 0.0 and ok1:
                frac = g0 / (g0 - g1) if g0 < 0.0 else 0.0
                return name, min(max(frac, 0.0), 1.0)
        return None

    def _record(self, kind: str, **kw):
        ev = {"t": self.s.t, "type": kind}
        ev.update(kw)
        self.events.append(ev)
        return ev

    def _apply_event(self, name: str):
        s, p = self.s, self.p
        if name == "bottom":
            self._event_bottom()
        elif name == "seat":
            self._event_seat()
        elif name == "snap":
            self._event_snap()

    def _event_bottom(self):
        """§3.6"""
        s, p = self.s, self.p
        r, _ = p.radius(s.ell)
        omega_m = s.sigma * s.elld / r
        E_minus = 0.5 * p.m * s.elld**2 + 0.5 * p.I * omega_m**2
        v_minus = s.elld
        res = yd.bottom_event(p, s.elld, s.sigma, omega_m)
        (ux, uy), _ = yd.string_dirs(s.theta)
        s.elld, s.sigma = res.elld, res.sigma
        s.ell = min(s.ell, p.L)
        s.pulse_t0, s.pulse_J, s.pulse_dt = s.t, res.J, res.pulse_dt
        s.pulse_ux, s.pulse_uy = ux, uy
        s.n_bottom += 1
        H, Hd, _ = self._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
        self._sync_taut_cartesian(H, Hd)
        E_plus = 0.5 * p.m * s.elld**2 + 0.5 * p.I * s.omega**2
        self._record("bottom", v_minus=v_minus, J=res.J, pulse_dt=res.pulse_dt,
                     omega_minus=omega_m, omega_plus=s.omega, sigma=s.sigma,
                     E_minus=E_minus, E_plus=E_plus, n=s.n_bottom, delta=self.diag.delta)

    def _event_snap(self):
        """§3.5 string re-tensions."""
        s, p = self.s, self.p
        H, Hd, _ = self._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
        rx, ry = s.px - H[0], s.py - H[1]
        d = math.hypot(rx, ry)
        ux, uy = rx / d, ry / d
        v_r = (s.vx - Hd[0]) * ux + (s.vy - Hd[1]) * uy
        r, _ = p.radius(s.ell_s)
        lam_uu, Minv, Jac = 0.0, None, None
        if not self.kin:
            terms = self._arm_solve_terms(s.t, [s.q1, s.phi2, s.q1d, s.phi2d], False, 0.0, 0.0)
            Lxx, Lxy, Lyy = terms[5]
            Minv, Jac = terms[6], terms[7]
            lam_uu = ux * (Lxx * ux + Lxy * uy) + uy * (Lxy * ux + Lyy * uy)
        P, e = yd.snap_impulse(p, v_r, s.omega, s.sigma, r, lam_uu)
        ev = {"P": P, "e": e, "v_r_minus": v_r, "omega_minus": s.omega,
              "ux": ux, "uy": uy, "vx_minus": s.vx, "vy_minus": s.vy,
              "q1d_minus": s.q1d, "phi2d_minus": s.phi2d}
        if P > 0.0:
            s.vx -= P / p.m * ux
            s.vy -= P / p.m * uy
            s.omega += s.sigma * P * r / p.I
            if Minv is not None:
                (i11, i12, i22), ((j00, j01), (j10, j11)) = Minv, Jac
                g1, g2 = j00 * P * ux + j10 * P * uy, j01 * P * ux + j11 * P * uy
                s.q1d += i11 * g1 + i12 * g2
                s.phi2d += i12 * g1 + i22 * g2
                H, Hd, _ = self._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
        if P > 0.0 and e > 0.0:
            ev["result"] = "rebound"  # stays FREE, waits for the next snap
        else:
            self._to_taut(H, Hd, ux, uy, d, r)
            ev["result"] = "taut"
        v_r_plus = (s.vx - Hd[0]) * ux + (s.vy - Hd[1]) * uy
        ev.update(v_r_plus=v_r_plus, omega_plus=s.omega, q1d_plus=s.q1d, phi2d_plus=s.phi2d,
                  vx_plus=s.vx, vy_plus=s.vy, sigma=s.sigma, r=r)
        self._record("snap", **ev)

    def _to_taut(self, H, Hd, ux, uy, d, r):
        s = self.s
        tx, ty = -uy, ux  # u_theta for u = (sin th, -cos th)
        s.mode = int(Mode.TAUT)
        s.ell = d
        s.theta = math.atan2(ux, -uy)
        s.elld = (s.vx - Hd[0]) * ux + (s.vy - Hd[1]) * uy
        s.thetad = ((s.vx - Hd[0]) * tx + (s.vy - Hd[1]) * ty) / d
        if not self.p.planar:
            s.theta, s.thetad = 0.0, 0.0
        s.omega = s.sigma * s.elld / r

    def _event_seat(self):
        """§4.2 step 3: slow -> STOWED, fast -> rebound (TAUT or, amendment, FREE)."""
        s, p = self.s, self.p
        H, Hd, _ = self._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
        if s.mode == Mode.TAUT:
            (ux, uy), _ = yd.string_dirs(s.theta)
            v = -s.elld
        else:
            rx, ry = s.px - H[0], s.py - H[1]
            d = math.hypot(rx, ry)
            ux, uy = rx / d, ry / d
            v = -((s.vx - Hd[0]) * ux + (s.vy - Hd[1]) * uy)
        if v < p.v_cap_max:
            self._stow(v)
            return
        if not self.kin:  # impulse on the hook -m (1 + e_cup) v u (§4.2)
            Jimp = -p.m * (1 + p.e_cup) * v
            terms = self._arm_solve_terms(s.t, [s.q1, s.phi2, s.q1d, s.phi2d], False, 0.0, 0.0)
            (i11, i12, i22), ((j00, j01), (j10, j11)) = terms[6], terms[7]
            g1, g2 = j00 * Jimp * ux + j10 * Jimp * uy, j01 * Jimp * ux + j11 * Jimp * uy
            s.q1d += i11 * g1 + i12 * g2
            s.phi2d += i12 * g1 + i22 * g2
        H, Hd, _ = self._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
        if s.mode == Mode.TAUT:
            s.elld = p.e_cup * v
            s.ell = p.ell_min
            self._sync_taut_cartesian(H, Hd)
        else:  # reverse the radial relative velocity, keep the tangential one
            vrx, vry = s.vx - Hd[0], s.vy - Hd[1]
            dv = (1.0 + p.e_cup) * v
            s.vx, s.vy = Hd[0] + vrx + dv * ux, Hd[1] + vry + dv * uy
        self._record("seat_rebound", v=v, e_cup=p.e_cup)

    def _stow(self, v):
        s, p = self.s, self.p
        if not self.kin:  # perfectly inelastic merge: conserve generalized momentum
            m11, m12, m22 = ad.mass_matrix(self.c_free, s.q1, s.phi2)
            g1 = m11 * s.q1d + m12 * s.phi2d
            g2 = m12 * s.q1d + m22 * s.phi2d
            sx, sy = self._stow_point(s.q1, s.phi2)
            (j00, j01), (j10, j11) = ak.world_point_jacobian(p.l1, s.q1, s.phi2, sx, sy)
            g1 += j00 * p.m * s.vx + j10 * p.m * s.vy
            g2 += j01 * p.m * s.vx + j11 * p.m * s.vy
            n11, n12, n22 = ad.mass_matrix(self.c_stow, s.q1, s.phi2)
            s.q1d, s.phi2d = ad.solve2(n11, n12, n22, g1, g2)
        s.mode = int(Mode.STOWED)
        s.inside = True
        s.caught = True
        s.t_caught = s.t
        s.pulse_t0 = -1.0
        self._sync_stowed_yoyo()
        self._record("stowed", v=v)

    def _release(self, F_in):
        """§4.1 magnet lets go: FREE with the seat point's velocity."""
        s, p = self.s, self.p
        self._sync_stowed_yoyo()
        s.mode = int(Mode.FREE)
        s.omega, s.sigma, s.ell_s = 0.0, 1, p.ell_min
        s.inside = False
        s.released = True
        s.caught = False
        self._record("release", F_in=F_in, vx=s.vx, vy=s.vy)

    def _to_free(self, T):
        s = self.s
        s.mode = int(Mode.FREE)
        s.ell_s = s.ell
        self._record("slack", T=T, ell=s.ell)

    def _fail(self, reason: str):
        s = self.s
        s.mode = int(Mode.FAIL)
        s.fail_reason = reason
        s.inside = False
        s.vx = s.vy = s.omega = 0.0
        self._record("fail", reason=reason)

    # ================================================================== step
    def _update_motors(self):
        s, p = self.s, self.p
        while self.cmd_queue and self.cmd_queue[0][0] <= s.t:
            _, s.q1_ref, s.q2_ref = self.cmd_queue.popleft()
        q1m, q2m, q1dm, q2dm, _, _ = self.sensors.reading()
        i1c, i2c = am.pd_current_command(p, s.q1_ref, s.q2_ref, q1m, q2m, q1dm, q2dm)
        am.step_currents(p, self.motor, i1c, i2c, s.q1d, s.q2d, p.dt)
        self._tau = am.motor_torques(p, self.motor, p.N1 * s.q1, s.q2)

    def step(self) -> Diag:
        s, p = self.s, self.p
        h = p.dt
        if not self.kin:
            self._update_motors()
        t0, mode = s.t, s.mode
        y0 = self._pack()
        y1 = self._rk4(t0, y0, h, mode)
        ev = self._find_event(t0, y0, y1, mode, h)
        if ev is not None:
            name, frac = ev
            ye = self._rk4(t0, y0, frac * h, mode) if frac > 0 else y0
            self._unpack(ye, mode, t0 + frac * h)
            self._apply_event(name)
            rest = (1.0 - frac) * h
            if rest > 0:
                y = self._rk4(s.t, self._pack(), rest, s.mode)
                self._unpack(y, s.mode, t0 + h)
            else:
                s.t = t0 + h
        else:
            self._unpack(y1, mode, t0 + h)
        if s.pulse_t0 >= 0.0 and s.t - s.pulse_t0 > s.pulse_dt:
            s.pulse_t0 = -1.0
        self.diag = self._diagnostics(s.t, self._pack(), s.mode)
        self._step_end_checks()
        s.spin_angle += s.omega * h
        if not self.kin:
            self.sensors.update(s.q1, s.q2, self.motor.i1, self.motor.i2)
        if self.log is not None:
            self.log.record(s, self.diag, self.motor, self.sensors)
        return self.diag

    def _step_end_checks(self):
        s, p, dg = self.s, self.p, self.diag
        if s.mode == Mode.STOWED:
            if dg.F_in > p.F_mag0:
                self._release(dg.F_in)
        elif s.mode == Mode.TAUT and dg.T < SLACK_T:
            self._to_free(dg.T)
        if s.mode in (Mode.TAUT, Mode.FREE) and self.funnel is not None:
            self._mouth_checks()
        if s.mode in (Mode.TAUT, Mode.FREE) and self.funnel is not None \
                and p.capture_model == "geometry":
            H = (dg.Hx, dg.Hy)
            ax, ay = math.cos(s.phi2), math.sin(s.phi2)
            rx, ry = s.px - H[0], s.py - H[1]
            if self.funnel.outside_hit(-rx * ay + ry * ax, rx * ax + ry * ay):
                self._fail("hit_rim")
        if not self.kin and s.mode != Mode.FAIL and not (p.q1_min <= s.q1 <= p.q1_max):
            self._fail("joint_limit")

    def _mouth_checks(self):
        """§4.2 step 1: crossing the mouth (H distance ell_mouth) sets/clears ``inside``."""
        s, p, dg = self.s, self.p, self.diag
        if s.mode == Mode.TAUT:
            d, v_r = s.ell, s.elld
        else:
            rx, ry = s.px - dg.Hx, s.py - dg.Hy
            d = math.hypot(rx, ry)
            v_r = ((s.vx - dg.Hdx) * rx + (s.vy - dg.Hdy) * ry) / max(d, 1e-9)
        # entering = crossing the mouth plane from outside (not merely being inside
        # and moving inwards, e.g. right after a release)
        entering = d < p.ell_mouth <= self._prev_d and v_r < 0.0 and not s.inside
        leaving = d >= p.ell_mouth and s.inside
        self._prev_d = d
        if entering:
            if p.capture_model == "gate":
                (ux, uy), _ = yd.string_dirs(s.theta)
                if s.mode == Mode.FREE:
                    ux, uy = rx / d, ry / d
                ax, ay = math.cos(s.phi2), math.sin(s.phi2)
                beta = math.acos(max(-1.0, min(1.0, ux * ax + uy * ay)))
                if beta >= p.alpha_cap:
                    self._record("mouth", beta=beta, aligned=False)
                    self._fail("misaligned")
                    return
                self._record("mouth", beta=beta, aligned=True)
                if s.mode == Mode.TAUT:
                    s.theta, s.thetad = s.phi2 + math.pi / 2, s.phi2d
            else:
                self._record("mouth", aligned=True)
            s.inside = True
        elif leaving:
            s.inside = False

    # ============================================================ convenience
    def run(self, duration: float, controller: Callable | None = None,
            control_dt: float | None = None, stop_on_fail: bool = True):
        """Run for ``duration`` seconds. ``controller(sim) -> (q1_ref, q2_ref) | None``
        is called every control_dt (default: substeps * dt, i.e. 200 Hz)."""
        p = self.p
        n_ctrl = max(1, int(round((control_dt or p.substeps * p.dt) / p.dt)))
        n = int(round(duration / p.dt))
        for k in range(n):
            if controller is not None and k % n_ctrl == 0:
                ref = controller(self)
                if ref is not None:
                    self.set_reference(*ref)
            self.step()
            if stop_on_fail and self.s.mode == Mode.FAIL:
                break
        return self
