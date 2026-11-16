"""Flip catch (PHYSICS_SPEC §4.4, ``catch_mode = "flip"``) with amendment A9.

IN_CUP: the seat walls keep the yo-yo on the funnel axis a = (cos phi2, sin phi2);
it can only slide along it. s >= 0 is its distance from the seat. Coordinates
z = (q1, phi2, s, theta_y), theta_y = absolute spin angle of the yo-yo:

    P  = E(q1) + x a(phi2),   x = l2 + ell_min + s       (x = s for the flip hand)
    M4(q) z_dd = f4(q, z_d) + G^T lam

M4 is the bare arm (c_free: funnel without yo-yo) plus m J_P^T J_P plus I on
theta_y; f4 holds drive, friction, bias and gravity (and a weak magnet, if any).
Two unilateral constraints, each with force lam >= 0:

    seat    s >= 0                                   lam = seat normal force N
    string  s_d = sigma r (theta_y_d - phi2_d)       lam = tension T (only while taut)

**A9 (spin coupling).** §4.4 (Claude Doc §4.3) as first written used s_dd = kappa A_out. The
string row above adds two effects that are as large as s_max:
  * the cup rotation turns the string direction while the yo-yo's spin inertia
    holds the yo-yo back, so rotating winds or unwinds string:
        s_dd = kappa A_out - sigma (1 - kappa) r phi2_dd      (taut)
  * the string cannot push: T = (I kappa / r^2) (A_out + sigma r phi2_dd) must
    stay >= 0, otherwise it goes slack and s_dd = A_out (no kappa slow-down).
Flipping with phi2_dd of sign sigma first winds the string and pins the yo-yo
in the seat; the other direction lets it fall out (control/stow_flip.py picks
the direction from sigma). ``spin_coupling: false`` restores the first-written
bead model (string row without phi2, bilateral) for the §4.4 regression numbers.

Simplifications, all TODO(sysid):
  * r = r_full in the cup (string almost fully wound); TAUT uses p.radius(ell)
    (r0 unless variable_radius), so the spin rate jumps when leaving the cup.
  * no wall or seat friction on the spin or the slide (v1, §4.4).
  * seat impacts follow §4.2: inelastic below v_cap_max, e_cup above; string
    snaps in the cup follow §3.5 (e_taut above v_snap_min).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from arm import dynamics as ad
from arm import kinematics as ak
from sim.params import Params
from yoyo.catch import CatchBase
from yoyo.funnel import magnet_force
from yoyo.modes import Mode

SEAT_ROW = np.array([0.0, 0.0, 1.0, 0.0])
_TOL = 1e-10


def cup_radius(p: Params) -> float:
    """Winding radius while in the cup. TODO(sysid): r(l) close to the seat."""
    return p.r_full


def string_row(p: Params, sigma: int, r: float) -> np.ndarray:
    """String constraint row, oriented so that slack opening is positive:
    g.z_d = sigma r (theta_y_d - phi2_d) - s_d  (paid-out minus needed string)."""
    if p.spin_coupling:
        return np.array([0.0, -sigma * r, -1.0, sigma * r])
    return np.array([0.0, 0.0, -1.0, sigma * r])


def cup_system(p: Params, q1, phi2, q1d, phi2d, s, sd, Ma, ra):
    """M4 z_dd = f4 (+ G^T lam). Ma = (m11, m12, m22), ra = drive + friction - bias
    of the bare arm. Returns M4, f4, J_P (2x3) and the bias acceleration of P."""
    x = p.l2 + p.ell_min + s
    s1, c1 = math.sin(q1), math.cos(q1)
    sa, ca = math.sin(phi2), math.cos(phi2)
    J = np.array([[-p.l1 * s1, -x * sa, ca],
                  [p.l1 * c1, x * ca, sa]])
    b = np.array([-p.l1 * c1 * q1d**2 - x * ca * phi2d**2 - 2.0 * sd * phi2d * sa,
                  -p.l1 * s1 * q1d**2 - x * sa * phi2d**2 + 2.0 * sd * phi2d * ca])
    M = np.zeros((4, 4))
    M[0, 0], M[0, 1], M[1, 0], M[1, 1] = Ma[0], Ma[1], Ma[1], Ma[2]
    M[:3, :3] += p.m * (J.T @ J)
    M[3, 3] = p.I
    f = np.zeros(4)
    f[0], f[1] = ra
    f[:3] -= p.m * (J.T @ (b + np.array([0.0, p.g])))
    f[2] -= magnet_force(p, p.ell_min + s)
    return M, f, J, b


def _kkt(M, rhs, rows, targets):
    n, k = M.shape[0], len(rows)
    if k == 0:
        return np.linalg.solve(M, rhs), np.zeros(0)
    G = np.array(rows)
    K = np.zeros((n + k, n + k))
    K[:n, :n] = M
    K[:n, n:] = -G.T
    K[n:, :n] = G
    sol = np.linalg.solve(K, np.concatenate([rhs, targets]))
    return sol[:n], sol[n:]


def solve_lcp(M, rhs, uni, bil):
    """Smallest mixed LCP over the active set (at most a few unilateral rows).

    M x = rhs + G^T lam.  ``uni``: list of (g, c): lam >= 0 when active (g.x = c),
    g.x >= c when inactive.  ``bil``: list of (g, c), always g.x = c.
    Used both at acceleration level (rhs = f4) and for impulses (rhs = M v-).
    Returns x, lam_uni (0 where inactive), lam_bil, active mask of ``uni``."""
    n_u = len(uni)
    best, best_err = None, math.inf
    for mask in itertools.product((1, 0), repeat=n_u):
        act = [u for u, a in zip(uni, mask) if a]
        rows = [g for g, _ in act] + [g for g, _ in bil]
        tg = [c for _, c in act] + [c for _, c in bil]
        x, lam = _kkt(M, rhs, rows, tg)
        lu = np.zeros(n_u)
        j = 0
        err = 0.0
        for i, a in enumerate(mask):
            if a:
                lu[i] = lam[j]
                err = max(err, -lam[j])
                j += 1
            else:
                g, c = uni[i]
                err = max(err, c - g @ x)
        if err <= _TOL * (1.0 + np.abs(rhs).max()):
            return x, lu, lam[len(act):], mask
        if err < best_err:
            best, best_err = (x, lu, lam[len(act):], mask), err
    return best


def _candidates(p, sigma, s, sd, gap, v):
    """Unilateral / bilateral constraint rows that may be active at this state."""
    r = cup_radius(p)
    gs = string_row(p, sigma, r)
    uni, bil, names = [], [], []
    if s <= 0.0 and sd <= 0.0:
        uni.append((SEAT_ROW, 0.0))
        names.append("N")
    if not p.spin_coupling:
        bil.append((gs, 0.0))
    elif gap <= 0.0 and gs @ v <= 1e-12:
        uni.append((gs, 0.0))
        names.append("T")
    return uni, bil, names, gs


def cup_accel(p: Params, sigma: int, M, f, s, sd, gap, v, prescribed=()):
    """Accelerations z_dd and the contact forces (N, T) at one instant.
    ``prescribed``: extra bilateral rows (g, value), e.g. kinematic arm motion."""
    uni, bil, names, gs = _candidates(p, sigma, s, sd, gap, v)
    z, lu, lb, mask = solve_lcp(M, f, uni, bil + list(prescribed))
    act = dict(zip(names, mask))
    N = lu[names.index("N")] if "N" in names else 0.0
    if "T" in names:
        T, taut = lu[names.index("T")], bool(act["T"])
    elif bil:                                   # spin_coupling off: bilateral string
        T, taut = lb[0], True
    else:
        T, taut = 0.0, False
    if act.get("N"):
        z[2] = 0.0  # held in the seat: keep s and s_d exactly zero
    gap_rate = 0.0 if taut else gs @ v
    return z, N, T, taut, gap_rate


def cup_impact(p: Params, sigma: int, M, v, s, gap, kind: str, prescribed=()):
    """Velocity jump at a seat impact (``kind="seat"``) or a string snap (``"snap"``)
    inside the cup. Returns v+, (seat impulse, string impulse), restitution used."""
    r = cup_radius(p)
    gs = string_row(p, sigma, r)
    uni, bil, names = [], [], []
    if kind == "seat":
        v_in = -v[2]
        e = p.e_cup if v_in >= p.v_cap_max else 0.0       # §4.2 rule
        uni.append((SEAT_ROW, e * v_in))
        names.append("N")
        if not p.spin_coupling:
            bil.append((gs, 0.0))
        elif gap <= 0.0:
            uni.append((gs, 0.0))
            names.append("T")
    else:
        v_c = -(gs @ v)                                    # closing speed
        e = p.e_taut if v_c > p.v_snap_min else 0.0       # §3.5 rule
        uni.append((gs, e * v_c))
        names.append("T")
        if s <= 0.0:
            uni.append((SEAT_ROW, 0.0))
            names.append("N")
    vp, lu, _, _ = solve_lcp(M, M @ v, uni, bil + list(prescribed))
    imp = dict(zip(names, lu))
    return vp, (imp.get("N", 0.0), imp.get("T", 0.0)), e


# =============================================================== simulator mode
class FlipCatch(CatchBase):
    """IN_CUP mode inside ``Simulator``. State vector: arm + [s, s_d, omega, gap]
    with omega = theta_y_d - phi2_d (spin relative to the string direction, the
    TAUT convention) and gap = paid-out string minus the geometric length."""

    # ------------------------------------------------------------ helpers
    def _arm_terms(self, q1, f2, q1d, f2d):
        sim, p = self.sim, self.sim.p
        c = sim.c_free
        Ma = ad.mass_matrix(c, q1, f2)
        h1, h2 = ad.bias_forces(c, q1, f2, q1d, f2d)
        d1, d2 = ad.drive_forces(p, *sim._tau)
        fr1, fr2 = ad.friction_forces(p, q1d, f2d)
        return Ma, (d1 + fr1 - h1, d2 + fr2 - h2)

    def _system(self, y):
        p = self.sim.p
        q1, f2, q1d, f2d, cs, csd, om, gap = y[:8]
        Ma, ra = self._arm_terms(q1, f2, q1d, f2d)
        M, f, J, b = cup_system(p, q1, f2, q1d, f2d, cs, csd, Ma, ra)
        v = np.array([q1d, f2d, csd, om + f2d])
        return M, f, J, b, v, Ma, ra

    def _check_kin(self):
        if self.sim.kin:
            raise NotImplementedError("IN_CUP needs the arm dynamics (no KinematicHook)")

    def _sync(self):
        """Cartesian and TAUT-style coordinates of the yo-yo from the cup state."""
        sim = self.sim
        s, p = sim.s, sim.p
        x = p.l2 + p.ell_min + s.cup_s
        ca, sa = math.cos(s.phi2), math.sin(s.phi2)
        ex, ey = ak.elbow(p.l1, s.q1)
        edx, edy = -p.l1 * math.sin(s.q1) * s.q1d, p.l1 * math.cos(s.q1) * s.q1d
        s.px, s.py = ex + x * ca, ey + x * sa
        s.vx = edx + s.cup_sd * ca - x * s.phi2d * sa
        s.vy = edy + s.cup_sd * sa + x * s.phi2d * ca
        s.ell, s.elld = p.ell_min + s.cup_s, s.cup_sd
        s.theta, s.thetad = s.phi2 + math.pi / 2, s.phi2d
        s.ell_s = s.ell + s.cup_gap

    def _enter(self, entry: str, omega: float):
        s = self.sim.s
        s.mode = int(Mode.IN_CUP)
        s.cup_s = s.cup_sd = s.cup_gap = s.cup_hold = 0.0
        s.omega = omega
        s.cup_entry = entry
        s.inside = True
        s.pulse_t0 = -1.0
        self._sync()

    def reset_in_cup(self, s: float = 0.0, sd: float = 0.0, q1_ref: float = 0.0,
                     q2_ref: float = -math.pi / 2, sigma: int = 1, entry: str = "catch"):
        from sim.simulator import SimState
        sim = self.sim
        self._check_kin()
        sim.s = SimState(mode=int(Mode.IN_CUP), q1_ref=q1_ref, q2_ref=q2_ref, sigma=sigma)
        sim._place_arm_static(q1_ref, q2_ref, hanging=False)
        st = sim.s
        st.released = entry == "catch"
        st.cup_s, st.cup_sd, st.cup_entry, st.inside = s, sd, entry, True
        r = cup_radius(sim.p)
        # taut string: omega from the rolling constraint (theta_y_d = phi2_d + s_d / (sigma r))
        st.omega = sd / (sigma * r) if sim.p.spin_coupling else sd / (sigma * r) - st.phi2d
        self._sync()
        sim._finish_reset()
        return st

    # ------------------------------------------------------- simulator hooks
    def pack(self):
        s = self.sim.s
        return [s.cup_s, s.cup_sd, s.omega, s.cup_gap]

    def unpack(self, y, t):
        s = self.sim.s
        s.cup_s, s.cup_sd, s.omega, s.cup_gap = y[4], y[5], y[6], y[7]
        self._sync()

    def deriv(self, t, y, want_diag=False):
        self._check_kin()
        sim = self.sim
        p = sim.p
        M, f, J, b, v, Ma, ra = self._system(y)
        z, N, T, taut, gap_rate = cup_accel(p, sim.s.sigma, M, f, y[4], y[5], y[7], v)
        q1dd, f2dd, sdd, thdd = z
        dy = [float(v) for v in (y[2], y[3], q1dd, f2dd, y[5], sdd, thdd - f2dd, gap_rate)]
        if not want_diag:
            return dy
        from sim.simulator import Diag
        dg = Diag()
        q1, f2, q1d, f2d = y[:4]
        dg.T, dg.F_mag, dg.N_seat = T, magnet_force(p, p.ell_min + y[4]), N
        H = ak.hook(p.l1, p.l2, q1, f2)
        (j00, j01), (j10, j11) = ak.hook_jacobian(p.l1, p.l2, q1, f2)
        dg.Hx, dg.Hy = H
        dg.Hdx, dg.Hdy = j00 * q1d + j01 * f2d, j10 * q1d + j11 * f2d
        dg.Hddx = j00 * q1dd + j01 * f2dd - p.l1 * math.cos(q1) * q1d**2 - p.l2 * math.cos(f2) * f2d**2
        dg.Hddy = j10 * q1dd + j11 * f2dd - p.l1 * math.sin(q1) * q1d**2 - p.l2 * math.sin(f2) * f2d**2
        dg.q1dd, dg.phi2dd = q1dd, f2dd
        dg.tau_m1, dg.tau_m2 = sim._tau
        Pdd = J @ z[:3] + b
        dg.Fhx, dg.Fhy = -p.m * Pdd[0], -p.m * (Pdd[1] + p.g)   # net force of the yo-yo on link 2
        Q2 = Ma[1] * q1dd + Ma[2] * f2dd - ra[1]                 # its generalized force on phi2
        dg.tau_wrist_rope = Q2
        return dy, dg

    def event_fn(self, t, y):
        """Interpolated IN_CUP events; whether they are armed uses the state at the
        start of the step (``sim.s`` is only updated after the step)."""
        s0, p = self.sim.s, self.sim.p
        out = {}
        if s0.cup_s > 0.0:
            out["cup_seat"] = (-float(y[4]), y[5] < 0.0)
        if p.spin_coupling and s0.cup_gap > 0.0:
            out["cup_snap"] = (-float(y[7]), True)
        return out

    def apply_event(self, name):
        if name not in ("cup_seat", "cup_snap"):
            raise ValueError(f"unknown event {name}")
        self._impact("seat" if name == "cup_seat" else "snap", name)

    def _impact(self, kind: str, name: str):
        sim = self.sim
        s, p = sim.s, sim.p
        if kind == "seat":
            s.cup_s = 0.0
        else:
            s.cup_gap = 0.0
        M, _, _, _, v, _, _ = self._system(self.pack_full())
        vp, (P_seat, P_str), e = cup_impact(p, s.sigma, M, v, s.cup_s, s.cup_gap, kind)
        s.q1d, s.phi2d, s.cup_sd = float(vp[0]), float(vp[1]), float(vp[2])
        s.omega = float(vp[3] - vp[1])
        if kind == "seat" and abs(s.cup_sd) < 1e-12:
            s.cup_sd = 0.0
        self._sync()
        sim._record(name, v_minus=float(v[2]), v_plus=float(s.cup_sd), e=e,
                    P_seat=float(P_seat), P_string=float(P_str))

    def pack_full(self):
        s = self.sim.s
        return [s.q1, s.phi2, s.q1d, s.phi2d] + self.pack()

    def step_end_in_cup(self):
        sim = self.sim
        s, p = sim.s, sim.p
        # only one interpolated event per step (§2): a second crossing later in the
        # same step is caught here and projected back onto the constraint
        if s.cup_s < 0.0 or (s.cup_s == 0.0 and s.cup_sd < 0.0):
            self._impact("seat", "cup_seat")
        if p.spin_coupling and s.cup_gap < 0.0:
            self._impact("snap", "cup_snap")
        if s.cup_s > p.s_max:
            self._leave()
            return
        seated = s.cup_s <= 0.0 and s.cup_sd == 0.0
        if seated and math.sin(s.phi2) > math.cos(p.stow_tilt_max):
            s.cup_hold += p.dt
            if s.cup_hold >= p.t_stow_hold - 1e-9:
                self.stow(0.0)
        else:
            s.cup_hold = 0.0

    # --------------------------------------------------------- transitions
    def event_seat(self):
        """§4.4: slow arrival -> IN_CUP (s = 0, s_d = 0), fast -> rebound as §4.2."""
        p = self.sim.p
        v, ux, uy = self.arrival()
        if v < p.v_cap_max:
            self._from_flight(v)
            return
        self.seat_rebound(v, ux, uy)

    def _from_flight(self, v):
        """Perfectly inelastic arrival: the yo-yo stops in the seat on a taut string,
        so it also stops spinning relative to the cup; its spin angular momentum
        goes into link 2 (generalized momentum about (q1, phi2) is conserved)."""
        sim = self.sim
        s, p = sim.s, sim.p
        self._check_kin()
        w_abs = s.omega + s.thetad          # TAUT/FREE omega is relative to the string
        m11, m12, m22 = ad.mass_matrix(sim.c_free, s.q1, s.phi2)
        g1 = m11 * s.q1d + m12 * s.phi2d
        g2 = m12 * s.q1d + m22 * s.phi2d
        sx, sy = sim._stow_point(s.q1, s.phi2)
        (j00, j01), (j10, j11) = ak.world_point_jacobian(p.l1, s.q1, s.phi2, sx, sy)
        g1 += j00 * p.m * s.vx + j10 * p.m * s.vy
        g2 += j01 * p.m * s.vx + j11 * p.m * s.vy
        if p.spin_coupling:
            g2 += p.I * w_abs
        n11, n12, n22 = ad.mass_matrix(sim.c_stow, s.q1, s.phi2)
        q1d_m, phi2d_m = s.q1d, s.phi2d
        s.q1d, s.phi2d = ad.solve2(n11, n12, n22, g1, g2)
        self._enter("catch", 0.0 if p.spin_coupling else -s.phi2d)
        sim._record("in_cup", v=float(v), omega_abs=float(w_abs), dphi2d=s.phi2d - phi2d_m,
                    dq1d=s.q1d - q1d_m)

    def release(self, F_in):
        """STOWED -> IN_CUP once the cup no longer holds the yo-yo (F_in > F_mag0,
        i.e. the mouth tilts downwards for the flip hand without magnet)."""
        sim = self.sim
        sim._sync_stowed_yoyo()
        p = sim.p
        self._enter("throw", 0.0 if p.spin_coupling else -sim.s.phi2d)
        sim._record("unseat", F_in=F_in)

    def stow(self, v):
        """Seated with the mouth up for t_stow_hold -> STOWED (rigid with link 2)."""
        sim = self.sim
        s, p = sim.s, sim.p
        M, _, _, _, vv, _, _ = self._system(self.pack_full())
        A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
                      [0.0, 1.0 if p.spin_coupling else 0.0]])
        g1, g2 = A.T @ (M @ vv)
        n11, n12, n22 = ad.mass_matrix(sim.c_stow, s.q1, s.phi2)
        s.q1d, s.phi2d = ad.solve2(n11, n12, n22, g1, g2)
        s.mode = int(Mode.STOWED)
        s.inside = True
        s.caught = s.released
        if s.caught:
            s.t_caught = s.t
        s.pulse_t0 = -1.0
        sim._sync_stowed_yoyo()
        sim._record("stowed", v=v, via="flip", omega_rel=s.omega)
        s.omega = 0.0

    def _leave(self):
        """s > s_max: the yo-yo has left the seat -> TAUT (FREE if the string is slack)."""
        sim = self.sim
        s, p = sim.s, sim.p
        H, Hd, _ = sim._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
        kind = "release" if s.cup_entry == "throw" else "drop"
        s_out, sd_out = s.cup_s, s.cup_sd
        if s.cup_gap <= 0.0 or not p.spin_coupling:
            s.mode = int(Mode.TAUT)
            s.ell, s.elld = p.ell_min + s.cup_s, s.cup_sd
            s.theta, s.thetad = s.phi2 + math.pi / 2, s.phi2d
            sim._sync_taut_cartesian(H, Hd)
        else:
            s.mode = int(Mode.FREE)
            s.ell_s = p.ell_min + s.cup_s + s.cup_gap
        s.inside = True
        s.released = True
        s.caught = False
        s.cup_entry = ""
        sim._prev_d = s.ell
        sim._record(kind, s=s_out, sd=sd_out, vx=s.vx, vy=s.vy, to=int(s.mode))


# ===================================================== prescribed link-2 motion
def simulate_prescribed(p: Params, phi_fn, duration: float, sigma: int = 1,
                        q1: float = 0.0, s0: float = 0.0, dt: float = 1e-5):
    """IN_CUP with the wrist axis fixed at E(q1) and phi2(t) prescribed by
    ``phi_fn(t) -> (phi2, phi2_d, phi2_dd)`` (the §4.4 regression / estimate).
    Same equations and impacts as the simulator mode, integrated with RK4 at dt.
    Returns a dict of arrays (t, s, sd, T, N, taut) and ``fell_at`` (time s > s_max,
    or None)."""
    Ma, ra = (1.0, 0.0, 1.0), (0.0, 0.0)    # arm rows are prescribed; any SPD block works
    r = cup_radius(p)
    st = np.array([s0, 0.0, 0.0, 0.0])      # s, s_d, omega (= theta_y_d - phi2_d), gap
    phi, phid, _ = phi_fn(0.0)
    st[2] = 0.0 if p.spin_coupling else -phid

    def rows(t):
        _, phd, phdd = phi_fn(t)
        return (np.array([1.0, 0, 0, 0]), 0.0), (np.array([0, 1.0, 0, 0]), phdd), phd

    def deriv(t, y, out=None):
        ph, phd, phdd = phi_fn(t)
        M, f, _, _ = cup_system(p, q1, ph, 0.0, phd, y[0], y[1], Ma, ra)
        v = np.array([0.0, phd, y[1], y[2] + phd])
        r0, r1, _ = rows(t)
        z, N, T, taut, gr = cup_accel(p, sigma, M, f, y[0], y[1], y[3], v, (r0, r1))
        if out is not None:
            out.update(N=N, T=T, taut=taut)
        return np.array([y[1], z[2], z[3] - z[1], gr])

    n = int(round(duration / dt))
    rec = {k: np.zeros(n + 1) for k in ("t", "s", "sd", "T", "N", "taut")}
    fell_at = None
    t = 0.0
    for k in range(n + 1):
        info = {}
        deriv(t, st, info)
        rec["t"][k], rec["s"][k], rec["sd"][k] = t, st[0], st[1]
        rec["T"][k], rec["N"][k], rec["taut"][k] = info["T"], info["N"], info["taut"]
        if st[0] > p.s_max:
            fell_at = t
            for key in rec:
                rec[key] = rec[key][:k + 1]
            break
        if k == n:
            break
        k1 = deriv(t, st)
        k2 = deriv(t + dt / 2, st + dt / 2 * k1)
        k3 = deriv(t + dt / 2, st + dt / 2 * k2)
        k4 = deriv(t + dt, st + dt * k3)
        new = st + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        kind = None
        if st[0] > 0.0 and new[0] <= 0.0 and new[1] < 0.0:
            new[0], kind = 0.0, "seat"
        elif p.spin_coupling and st[3] > 0.0 and new[3] <= 0.0:
            new[3], kind = 0.0, "snap"
        if kind is not None:
            ph, phd, _ = phi_fn(t)
            M, _, _, _ = cup_system(p, q1, ph, 0.0, phd, new[0], new[1], Ma, ra)
            v = np.array([0.0, phd, new[1], new[2] + phd])
            r0, _, _ = rows(t)
            vp, _, _ = cup_impact(p, sigma, M, v, new[0], new[3], kind,
                                  (r0, (np.array([0, 1.0, 0, 0]), phd)))
            new[1], new[2] = vp[2], vp[3] - vp[1]
            if kind == "seat" and abs(new[1]) < 1e-12:
                new[1] = 0.0
        st = new
    rec["fell_at"] = fell_at
    rec["r"] = r
    return rec


def cosine_flip(t_flip: float, phi0: float = -math.pi / 2, direction: int = 1,
                angle: float = math.pi, t0: float = 0.0):
    """Raised-cosine velocity profile: phi_d = (angle / T)(1 - cos(2 pi tau / T))."""
    w = 2.0 * math.pi / t_flip

    def fn(t):
        tau = t - t0
        if tau <= 0.0:
            return phi0, 0.0, 0.0
        if tau >= t_flip:
            return phi0 + direction * angle, 0.0, 0.0
        ph = phi0 + direction * angle * (tau / t_flip - math.sin(w * tau) / (2.0 * math.pi))
        phd = direction * angle / t_flip * (1.0 - math.cos(w * tau))
        phdd = direction * angle / t_flip * w * math.sin(w * tau)
        return ph, phd, phdd
    return fn
