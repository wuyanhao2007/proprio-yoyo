"""Zlajpah-style pulse baseline with true state (roadmap stage 1 step 9).

"Cheating" controller: it reads the simulator's true yo-yo state (later stages
replace this with the current-based observer). Sequence:

    throw (open loop) -> for each bottom k = 1..n_bounces: a hook pulse around the
    bottom -> the last pulse is sized so the yo-yo just reaches the seat -> hold.

Energy injection. In the hook frame the string obeys l_dd = kappa (g + a) (§3.2,
a = upward hook acceleration), so the relative energy
E = (m/kappa) l_dot^2 / 2 - m g l changes at  dE/dt = m a l_dot.
Energy goes in when the hook accelerates up while the yo-yo descends (l_dot > 0)
and decelerates while it climbs (l_dot < 0). |l_dot| peaks at the bottom, so the
hook speeds up to v_max right before the bottom and slows down right after it:

    dKE+ ~= eta * m v v_max (zeta_e^2 + zeta_e)     (v: string speed at the bottom;
                                                    the pre-bottom part is scaled by zeta_e^2)

Two physical limits shape the pulse:
  * the deceleration after the bottom must stay below g, otherwise T < 0 and the
    string goes slack (T = m (1 - kappa)(g + a));
  * the up-acceleration is limited by joint-1 torque.
eta absorbs the remaining tracking error and is adapted after every pulse.
Between pulses the hook drifts back to its base height around the apex, where
l_dot ~ 0 so this costs almost nothing.

Reference shaping. The driver runs a PD without a velocity reference
(i ~ Kp (q_ref - q) - Kd q_dot, §6). Knowing the gains, the controller adds the
feed-forward itself:  q_ref = q_des + (Kd q_dot_des + J q_dd_des) / Kp, and
evaluates the desired trajectory ``lead`` seconds ahead (command latency).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from arm import dynamics as ad
from control.trajectories import Throw
from sim.env import predict_time_to_bottom
from yoyo.modes import Mode


@dataclass
class BaselineConfig:
    n_bounces: int = 3            # bottoms; the last one is the catch pulse (<= 0: never catch)
    apex_target: float = 0.13     # string length at the apex of normal bounces [m]
    v_arrive: float = 0.30        # target seat arrival speed for the catch [m/s]
    a_up: float = 20.0            # hook up-acceleration before the bottom [m/s^2]
    a_down: float = 8.0            # hook deceleration after the bottom (< g) [m/s^2]
    v_lim: float = 2.4            # max hook pulse speed [m/s]
    y_base: float = -0.10         # lowest base hook height (offset from horizontal link 1) [m]
    centre_bottom: bool = True    # start each pulse at -v_max^2/(2 a_up) so link 1 is
                                  # horizontal at the bottom (least sideways hook motion)
    y_max: float = 0.10           # never command the hook above this [m]
    lead: float = 0.007           # evaluate the plan this far ahead (latency) [s]
    T_return: float = 0.9         # slow return to the base height, centred on the apex [s]
    T_reposition: float = 0.5     # move from the throw end to y_base [s]
    eta0: float = 1.0             # initial guess of the injection efficiency
    eta_gain: float = 0.8         # adaptation rate of eta
    feedforward: bool = True
    align_wrist: bool = True      # catch leg: point the funnel along the string
    align_below: float = 0.35     # ... once l is below this [m]
    align_max: float = math.radians(25)
    throw_t0: float = 0.2
    throw_acc: float = 8.0
    throw_t_acc: float = 0.12


@dataclass
class PulseRecord:
    k: int
    t_start: float
    t_bottom_pred: float
    v: float
    v_max: float
    KE_minus: float
    KE_target: float
    eta_used: float
    catch: bool
    KE_plus: float = float("nan")
    eta_measured: float = float("nan")
    apex: float = float("nan")


class _Seg:
    """Hook height segment: y(t), v(t), a(t) on [t0, t0 + T]."""

    def __init__(self, t0, T, fn):
        self.t0, self.T, self.fn = t0, T, fn

    def done(self, t):
        return t >= self.t0 + self.T


def _cos_move(t0, T, ya, yb):
    D = yb - ya

    def fn(t):
        tau = min(max(t - t0, 0.0), T)
        c, s = math.cos(math.pi * tau / T), math.sin(math.pi * tau / T)
        k = math.pi / T
        return ya + D * 0.5 * (1 - c), D * 0.5 * k * s, D * 0.5 * k * k * c
    return _Seg(t0, T, fn)


def _pulse(t0, y0, v_max, a_up, a_dn):
    """Hook velocity ramps 0 -> v_max at a_up, then v_max -> 0 at a_dn."""
    ta, td = v_max / a_up, v_max / a_dn
    ya = y0 + 0.5 * a_up * ta * ta

    def fn(t):
        tau = t - t0
        if tau <= 0:
            return y0, 0.0, 0.0
        if tau < ta:
            return y0 + 0.5 * a_up * tau * tau, a_up * tau, a_up
        tau2 = min(tau - ta, td)
        return ya + v_max * tau2 - 0.5 * a_dn * tau2 * tau2, v_max - a_dn * tau2, (-a_dn if tau - ta < td else 0.0)
    return _Seg(t0, ta + td, fn), ta, ya + 0.5 * v_max * td


def _decel(t0, y0, v0, a_dn):
    """From height y0 and upward speed v0, decelerate at a_dn to rest."""
    td = max(v0, 0.0) / a_dn

    def fn(t):
        tau = min(max(t - t0, 0.0), td)
        a = -a_dn if t - t0 < td else 0.0
        return y0 + v0 * tau - 0.5 * a_dn * tau * tau, v0 - a_dn * tau, a
    return _Seg(t0, td, fn)


def _post_bottom(t0, y0, v0, dv_J, pdt, a_dn):
    """Plan after the bottom: the known raised-cosine drag of the bottom impulse
    (total dv_J over pdt, §3.6) plus a constant deceleration a_dn, until rest.
    Planning the drag avoids the PD fighting it (which removes yo-yo energy or
    decelerates the hook faster than g -> slack)."""
    w = 2 * math.pi / pdt

    def drag(tau):  # velocity lost and its integral
        if tau <= 0:
            return 0.0, 0.0
        if tau < pdt:
            F = (tau - math.sin(w * tau) / w) / pdt
            G = (tau * tau / 2 + (math.cos(w * tau) - 1) / (w * w)) / pdt
            a = (1 - math.cos(w * tau)) / pdt
            return dv_J * F, dv_J * G, dv_J * a
        return dv_J, dv_J * (pdt / 2 + (tau - pdt)), 0.0

    # stop time: v0 - drag - a_dn tau = 0
    lo, hi = 0.0, max(v0, 0.0) / a_dn + pdt
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if v0 - drag(mid)[0] - a_dn * mid > 0:
            lo = mid
        else:
            hi = mid
    T = lo

    def fn(t):
        tau = min(max(t - t0, 0.0), T)
        dv, dy, da = drag(tau)
        v = v0 - dv - a_dn * tau
        y = y0 + v0 * tau - dy - 0.5 * a_dn * tau * tau
        # feed-forward acceleration: only the motor's share; the drag comes from F_b
        a = -a_dn if t - t0 < T else 0.0
        return y, v, a
    return _Seg(t0, T, fn)


class ZlajpahBaseline:
    def __init__(self, params, cfg: BaselineConfig | None = None):
        self.p = params
        self.cfg = cfg or BaselineConfig()
        c = self.cfg
        self.throw = Throw(params.l1, t0=c.throw_t0, acc=c.throw_acc, t_acc=c.throw_t_acc)
        self.y_hold = self.throw.hook_offset(1e9)
        self.seg: _Seg | None = None
        self.phase = "throw"
        self.eta = c.eta0
        self.pulses: list[PulseRecord] = []
        self._await_measure = False
        self._track_apex = False
        self._apex_min = float("inf")
        # feed-forward inertia about joint 1: arm + string load m (1 - kappa) l1^2
        cst = ad.link_constants(params, stowed=False)
        self._cst = cst
        self._n_bottom0 = -1
        self._replanned = False
        self._J1 = cst.J1 + cst.M2 * params.l1**2 + params.m * (1 - params.kappa()) * params.l1**2

    # ------------------------------------------------------------ hook plan
    def _plan(self, t):
        if self.seg is not None:
            y, v, a = self.seg.fn(t)
            if self.seg.done(t) and t > self.seg.t0 + self.seg.T + self.cfg.lead:
                self.y_hold = self.seg.fn(self.seg.t0 + self.seg.T)[0]
                self.seg = None
            return y, v, a
        return self.y_hold, 0.0, 0.0

    def _refs(self, t, phi2):
        p, c = self.p, self.cfg
        y, v, a = self._plan(t + c.lead)
        y = min(y, c.y_max)
        s1 = max(min(y / p.l1, 0.999), -0.999)
        q1 = math.asin(s1)
        cq = math.cos(q1)
        q1d = v / (p.l1 * cq)
        q1dd = (a + p.l1 * s1 * q1d * q1d) / (p.l1 * cq)
        q2 = phi2 - q1
        q1r, q2r = q1, q2
        if c.feedforward:
            q1r = q1 + (p.Kd1 * q1d + self._J1 * q1dd) / p.Kp1
            q2r = q2 + (p.Kd2 * (-q1d)) / p.Kp2      # keep phi2 still: q2_dot = -q1_dot
        return q1r, q2r

    # ---------------------------------------------------------------- energy
    def _ke_target(self, catch: bool) -> float:
        p = self.p
        if catch:  # reach the seat with v_arrive (relative KE at the bottom equivalent)
            return p.m * p.g * (p.L - p.ell_min) + 0.5 * p.m / p.kappa() * self.cfg.v_arrive**2
        return p.m * p.g * (p.L - self.cfg.apex_target)

    def _start_pulse(self, sim, ttb: float):
        p, c, s = self.p, self.cfg, sim.s
        k = len(self.pulses)
        catch = c.n_bounces > 0 and k == c.n_bounces - 1
        kap = p.kappa()
        v = math.sqrt(s.elld**2 + 2 * kap * p.g * max(p.L - s.ell, 0.0))
        KE_m = 0.5 * p.m / kap * v * v
        KE_t = self._ke_target(catch)
        z = p.zeta_e
        # the bottom impulse J drags the hook down by dv_J = J u.Lam.u before the
        # post-bottom deceleration starts (it is not fought, see _reanchor)
        y0 = self._plan(s.t)[0]
        gain = self.eta * p.m * v * (z * z + z)
        v_max = c.v_lim
        for _ in range(3):  # dv_J depends on the arm posture at the bottom, i.e. on v_max
            y_b = min(y0 + v_max**2 / (2 * c.a_up), c.y_max)
            dv_J = p.m * (1 + z) * v * self._lam_yy_at(math.asin(y_b / p.l1))
            v_max = (KE_t - z * z * KE_m + self.eta * p.m * v * z * dv_J) / gain
            v_max = min(max(v_max, 0.0), c.v_lim)
        # the hook must not run past y_max: shrink v_max if needed (travel = up phase
        # + post-bottom deceleration after the drag dv_J)
        while v_max > 0.05:
            y_end = y0 + v_max**2 / (2 * c.a_up) + max(v_max - dv_J, 0.0)**2 / (2 * c.a_down)
            if y_end <= c.y_max:
                break
            v_max *= 0.97
        ta = v_max / c.a_up
        t0 = s.t + ttb - ta
        seg, _, _ = _pulse(t0, y0, v_max, c.a_up, c.a_down) if v_max > 0 else (_cos_move(s.t, 0.01, y0, y0), 0, y0)
        self.seg = seg
        self._n_bottom0 = s.n_bottom
        self._replanned = False
        self.pulses.append(PulseRecord(k=k + 1, t_start=t0, t_bottom_pred=s.t + ttb, v=v, v_max=v_max,
                                       KE_minus=KE_m, KE_target=KE_t, eta_used=self.eta, catch=catch))
        self._await_measure = True
        self.phase = "catch" if catch else "pulse"

    def _lam_yy(self, sim) -> float:
        return self._lam_yy_at(sim.s.q1, sim.s.phi2)

    def _lam_yy_at(self, q1: float, phi2: float = -math.pi / 2) -> float:
        """Vertical inverse operational-space inertia of the arm at H."""
        from arm import kinematics as ak
        p = self.p
        s = type("q", (), {"q1": q1, "phi2": phi2})
        m11, m12, m22 = ad.mass_matrix(self._cst, s.q1, s.phi2)
        i11, i12, i22 = ad.inv2(m11, m12, m22)
        (_, _), (j10, j11) = ak.hook_jacobian(p.l1, p.l2, s.q1, s.phi2)
        return j10 * (i11 * j10 + i12 * j11) + j11 * (i12 * j10 + i22 * j11)

    def _replan_after_bottom(self, sim):
        """At the detected bottom (exact time s.pulse_t0): continue from the planned
        hook state with the known drag of the bottom impulse added."""
        p, s = self.p, sim.s
        tb = s.pulse_t0
        y0, v0, _ = self.seg.fn(tb) if self.seg is not None else (self.y_hold, 0.0, 0.0)
        dv_J = s.pulse_J * abs(s.pulse_uy) * self._lam_yy(sim)
        self.seg = _post_bottom(tb, y0, v0, dv_J, s.pulse_dt, self.cfg.a_down)

    def _next_base(self, v_max_guess: float) -> float:
        """Base height for the next pulse. The hook moves on an arc of radius l1,
        so it also moves sideways (dx = -tan(q1) dy) and pumps the string swing;
        reaching peak speed with link 1 horizontal keeps that small."""
        c = self.cfg
        if not c.centre_bottom:
            return c.y_base
        return max(-v_max_guess**2 / (2 * c.a_up), c.y_base)

    def _measure(self, sim):
        """After the pulse: compare the achieved relative energy with the plan."""
        p, s, rec = self.p, sim.s, self.pulses[-1]
        kap = p.kappa()
        rec.KE_plus = 0.5 * p.m / kap * s.elld**2 + p.m * p.g * (p.L - s.ell)
        z = p.zeta_e
        if rec.v_max > 0.1:
            eta_m = (rec.KE_plus - z * z * rec.KE_minus) / (p.m * rec.v * rec.v_max * (z * z + z))
            rec.eta_measured = eta_m
            eta_m = min(max(eta_m, 0.3), 1.5)
            self.eta += self.cfg.eta_gain * (eta_m - self.eta)
        c = self.cfg
        self._await_measure = False
        self._track_apex = True
        self._apex_min = float("inf")
        if not rec.catch:  # drift back to the base height, centred on the apex
            t_apex = s.t + (-s.elld) / (kap * p.g)
            T = self.cfg.T_return
            t0 = max(s.t, t_apex - T / 2)
            next_is_catch = c.n_bounces > 0 and len(self.pulses) == c.n_bounces - 1
            # the catch pulse needs the most stroke: start it from the lowest base
            y_next = c.y_base if next_is_catch else self._next_base(rec.v_max)
            self.seg = _cos_move(t0, T, self.y_hold, y_next)

    # --------------------------------------------------------------- control
    def __call__(self, sim):
        p, c, s = self.p, self.cfg, sim.s
        t = s.t
        if self.phase == "throw":
            if s.released and t > c.throw_t0 + c.throw_t_acc + 0.15:
                self.phase = "track"
                self.seg = _cos_move(t, c.T_reposition, self.y_hold, c.y_base)
            else:
                return self.throw(sim)

        if s.mode == Mode.TAUT:
            if self._track_apex:
                self._apex_min = min(self._apex_min, s.ell)
                if s.elld > 0.0 and self.pulses:
                    self.pulses[-1].apex = self._apex_min
                    self._track_apex = False
            busy = (self.seg is not None and self.phase in ("pulse", "catch")) or self._await_measure
            n_ok = c.n_bounces <= 0 or len(self.pulses) < c.n_bounces
            if not busy and self.phase != "catch" and n_ok and s.elld > 0.0:
                ttb = predict_time_to_bottom(p, s.ell, s.elld)
                ta = c.v_lim / c.a_up
                if ttb <= ta + 0.03:
                    self._start_pulse(sim, ttb)
            if (self._await_measure and not self._replanned and s.n_bottom > self._n_bottom0
                    and s.pulse_t0 >= 0.0):
                self._replan_after_bottom(sim)
                self._replanned = True
            if self._await_measure and self.seg is None and s.elld < 0.0:
                self._measure(sim)
            if self.phase == "pulse" and self.seg is None and not self._await_measure:
                self.phase = "track"
            elif self.phase == "pulse" and not self._await_measure and self.seg is not None:
                self.phase = "track"   # return segment running

        phi2 = -math.pi / 2
        if (c.align_wrist and self.phase == "catch" and s.mode == Mode.TAUT
                and s.elld < 0.0 and s.ell < c.align_below):
            phi2 = min(max(s.theta, -c.align_max), c.align_max) - math.pi / 2  # funnel axis along u
        return self._refs(t, phi2)

    def summary(self) -> list[dict]:
        return [vars(r).copy() for r in self.pulses]
