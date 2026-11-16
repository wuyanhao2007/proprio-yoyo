"""Catch-mode dispatch (PHYSICS_SPEC §4.4).

``Simulator`` delegates everything that depends on how the yo-yo is caught and
held to a catch object chosen by ``Params.catch_mode``:

    "magnet"  yoyo/catch_magnet.py  §4.1-§4.2: magnet holds the seated yo-yo (default)
    "flip"    yoyo/catch_flip.py    §4.4: IN_CUP slide model, the hand flips the mouth up

Both share the fast-arrival seat rebound of §4.2 (``CatchBase.seat_rebound``).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from yoyo import dynamics as yd
from yoyo.modes import Mode

if TYPE_CHECKING:  # pragma: no cover
    from sim.simulator import Simulator


class CatchBase:
    """Interface. Methods take no simulator argument: ``self.sim`` is the owner."""

    def __init__(self, sim: "Simulator"):
        self.sim = sim

    # --- events shared by every mode -----------------------------------------
    def event_seat(self):
        """TAUT/FREE yo-yo reaches the seat (ell <= ell_min, moving inwards)."""
        raise NotImplementedError

    def stow(self, v: float):
        raise NotImplementedError

    def release(self, F_in: float):
        """STOWED and the inward pull needed exceeds what holds the yo-yo."""
        raise NotImplementedError

    def step_end_stowed(self, dg):
        p = self.sim.p
        if dg.F_in > p.F_mag0:
            self.release(dg.F_in)

    # --- IN_CUP hooks (flip only) --------------------------------------------
    def pack(self):
        raise RuntimeError("IN_CUP is only used with catch_mode = 'flip'")

    unpack = deriv = event_fn = apply_event = step_end_in_cup = pack

    # --- §4.2 fast arrival: bounce off the seat ------------------------------
    def seat_rebound(self, v: float, ux: float, uy: float):
        """Impulse -m (1 + e_cup) v u on the hook, yo-yo bounces back (§4.2)."""
        sim = self.sim
        s, p = sim.s, sim.p
        if not sim.kin:  # impulse on the hook -m (1 + e_cup) v u (§4.2)
            Jimp = -p.m * (1 + p.e_cup) * v
            terms = sim._arm_solve_terms(s.t, [s.q1, s.phi2, s.q1d, s.phi2d], False, 0.0, 0.0)
            (i11, i12, i22), ((j00, j01), (j10, j11)) = terms[6], terms[7]
            g1, g2 = j00 * Jimp * ux + j10 * Jimp * uy, j01 * Jimp * ux + j11 * Jimp * uy
            s.q1d += i11 * g1 + i12 * g2
            s.phi2d += i12 * g1 + i22 * g2
        H, Hd, _ = sim._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
        if s.mode == Mode.TAUT:
            s.elld = p.e_cup * v
            s.ell = p.ell_min
            sim._sync_taut_cartesian(H, Hd)
        else:  # reverse the radial relative velocity, keep the tangential one
            vrx, vry = s.vx - Hd[0], s.vy - Hd[1]
            dv = (1.0 + p.e_cup) * v
            s.vx, s.vy = Hd[0] + vrx + dv * ux, Hd[1] + vry + dv * uy
        sim._record("seat_rebound", v=v, e_cup=p.e_cup)

    def arrival(self):
        """Direction H -> yo-yo and inward speed v at a seat event."""
        sim = self.sim
        s = sim.s
        H, Hd, _ = sim._hook_kin(s.t, s.q1, s.phi2, s.q1d, s.phi2d)
        if s.mode == Mode.TAUT:
            (ux, uy), _ = yd.string_dirs(s.theta)
            v = -s.elld
        else:
            rx, ry = s.px - H[0], s.py - H[1]
            d = math.hypot(rx, ry)
            ux, uy = rx / d, ry / d
            v = -((s.vx - Hd[0]) * ux + (s.vy - Hd[1]) * uy)
        return v, ux, uy


def make_catch(sim: "Simulator") -> CatchBase:
    mode = sim.p.catch_mode
    if mode == "magnet":
        from yoyo.catch_magnet import MagnetCatch
        return MagnetCatch(sim)
    if mode == "flip":
        from yoyo.catch_flip import FlipCatch
        return FlipCatch(sim)
    raise ValueError(f"unknown catch_mode {mode!r}")
