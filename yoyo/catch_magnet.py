"""Magnet catch (PHYSICS_SPEC §4.1-§4.2, ``catch_mode = "magnet"``, the default).

Moved unchanged out of ``sim/simulator.py``: a slow arrival in the seat merges
the yo-yo into link 2 (STOWED, perfectly inelastic), a fast one bounces off the
seat, and the magnet lets go once the inward pull needed to keep the yo-yo
seated exceeds F_mag0. The magnet force on a yo-yo that is inside the funnel
(TAUT/FREE) stays in the simulator's derivative (``yoyo.funnel.magnet_force``).
"""
from __future__ import annotations

from arm import dynamics as ad
from arm import kinematics as ak
from yoyo.catch import CatchBase
from yoyo.modes import Mode


class MagnetCatch(CatchBase):
    def event_seat(self):
        """§4.2 step 3: slow -> STOWED, fast -> rebound (TAUT or, amendment, FREE)."""
        p = self.sim.p
        v, ux, uy = self.arrival()
        if v < p.v_cap_max:
            self.stow(v)
            return
        self.seat_rebound(v, ux, uy)

    def stow(self, v):
        sim = self.sim
        s, p = sim.s, sim.p
        if not sim.kin:  # perfectly inelastic merge: conserve generalized momentum
            m11, m12, m22 = ad.mass_matrix(sim.c_free, s.q1, s.phi2)
            g1 = m11 * s.q1d + m12 * s.phi2d
            g2 = m12 * s.q1d + m22 * s.phi2d
            sx, sy = sim._stow_point(s.q1, s.phi2)
            (j00, j01), (j10, j11) = ak.world_point_jacobian(p.l1, s.q1, s.phi2, sx, sy)
            g1 += j00 * p.m * s.vx + j10 * p.m * s.vy
            g2 += j01 * p.m * s.vx + j11 * p.m * s.vy
            n11, n12, n22 = ad.mass_matrix(sim.c_stow, s.q1, s.phi2)
            s.q1d, s.phi2d = ad.solve2(n11, n12, n22, g1, g2)
        s.mode = int(Mode.STOWED)
        s.inside = True
        s.caught = True
        s.t_caught = s.t
        s.pulse_t0 = -1.0
        sim._sync_stowed_yoyo()
        sim._record("stowed", v=v)

    def release(self, F_in):
        """§4.1 magnet lets go: FREE with the seat point's velocity."""
        sim = self.sim
        s, p = sim.s, sim.p
        sim._sync_stowed_yoyo()
        s.mode = int(Mode.FREE)
        s.omega, s.sigma, s.ell_s = 0.0, 1, p.ell_min
        s.inside = False
        s.released = True
        s.caught = False
        sim._record("release", F_in=F_in, vx=s.vx, vy=s.vy)
