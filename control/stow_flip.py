"""Flip stow (PHYSICS_SPEC §4.4, ``catch_mode = "flip"``).

Once the yo-yo sits in the seat after a catch, turn the hand 180 deg so the
mouth faces up and gravity holds the yo-yo; the simulator then switches IN_CUP ->
STOWED after t_stow_hold.

* Detection cheats for now: it reads the true simulator state (IN_CUP entered by
  a catch, seated). Stage 3 replaces this with the proprioceptive estimate.
* The absolute funnel angle phi2 follows a raised-cosine velocity profile over
  t_flip (``yoyo.catch_flip.cosine_flip``); joint 1 holds its reference.
* Direction (A9): ``"auto"`` turns with sign sigma, so the start of the turn
  winds the string and pins the yo-yo in the seat. The opposite sign unwinds it
  and the yo-yo slides out (tests/test_catch_flip.py shows both).
* Feedforward as in control/baseline.py: q_ref = q_des + (tau_ff + Kd qd_des)/Kp
  (the firmware PD has no velocity reference), tau_ff from the arm inverse
  dynamics with the seated yo-yo (spin inertia included), taken relative to the
  start so the reference does not step at t0. References lead by cmd_delay.

Settings come from the ``controller`` block of config/hand_flip.yaml.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable

from arm import dynamics as ad
from sim.params import Params, load_controller
from yoyo.catch_flip import cosine_flip
from yoyo.modes import Mode


@dataclass
class StowFlipConfig:
    t_flip: float
    direction: str | int = "auto"   # "auto" = sign(sigma) (A9), or +1 / -1
    q1_hold: bool = True
    feedforward: bool = True

    @classmethod
    def load(cls, hand: str = "flip", **overrides) -> "StowFlipConfig":
        vals = load_controller(hand)
        vals.update(overrides)
        known = {f.name for f in fields(cls)}
        unknown = set(vals) - known
        if unknown:
            raise ValueError(f"unknown stow_flip settings: {sorted(unknown)}")
        return cls(**vals)


class StowFlip:
    """``ctrl(sim) -> (q1_ref, q2_ref) | None``. ``before`` drives the arm until
    the trigger (default: keep the current references)."""

    def __init__(self, p: Params, cfg: StowFlipConfig | None = None,
                 before: Callable | None = None):
        if p.catch_mode != "flip":
            raise ValueError("StowFlip needs catch_mode = 'flip'")
        self.p = p
        self.cfg = cfg if cfg is not None else StowFlipConfig.load()
        self.before = before
        self.c = ad.link_constants(p, stowed=True, yoyo_spin=p.spin_coupling)
        self.t0: float | None = None
        self.direction = 0
        self.profile = None

    # ------------------------------------------------------------ detection
    @staticmethod
    def seated_after_catch(sim) -> bool:
        s = sim.s
        return (s.mode == Mode.IN_CUP and s.cup_entry == "catch"
                and s.cup_s <= 0.0 and s.cup_sd == 0.0)

    def __call__(self, sim):
        s, p = sim.s, self.p
        if self.t0 is None:
            if not self.seated_after_catch(sim):
                return self.before(sim) if self.before is not None else None
            self._start(sim)
        return self._refs(s.t - self.t0 + p.cmd_delay)

    # ------------------------------------------------------------ trajectory
    def _start(self, sim):
        s, c = sim.s, self.cfg
        self.t0 = s.t
        d = c.direction
        self.direction = (1 if s.sigma >= 0 else -1) if d == "auto" else int(d)
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be 'auto', 1 or -1, got {d!r}")
        if sim.cmd_queue:
            _, q1r, q2r = sim.cmd_queue[-1]
        else:
            q1r, q2r = s.q1_ref, s.q2_ref
        self.q1r0, self.q2r0 = q1r, q2r
        self.q1m = s.q1
        self.phi0 = s.phi2
        self.profile = cosine_flip(c.t_flip, phi0=s.phi2, direction=self.direction)
        self.ff0 = self._ff(self.phi0, 0.0, 0.0)
        sim._record("flip_start", direction=self.direction, t_flip=c.t_flip, phi0=s.phi2)

    def _ff(self, phi, phid, phidd):
        """Reference offsets (joint 1, wrist) that make the PD deliver the torques."""
        if not self.cfg.feedforward:
            return 0.0, 0.0
        p, c = self.p, self.c
        q1 = self.q1m
        m11, m12, m22 = ad.mass_matrix(c, q1, phi)
        h1, h2 = ad.bias_forces(c, q1, phi, 0.0, phid)
        fr1, fr2 = ad.friction_forces(p, 0.0, phid)
        Q1 = m12 * phidd + h1 - fr1
        Q2 = m22 * phidd + h2 - fr2
        tau2 = Q2                       # wrist motor torque (Q_drive2 = tau2)
        tau1j = Q1 + tau2               # joint-side torque N eta tau1 (Q_drive1 = N eta tau1 - tau2)
        return tau1j / p.Kp1, (tau2 + p.Kd2 * phid) / p.Kp2

    def _refs(self, tau):
        phi, phid, phidd = self.profile(tau)
        f1, f2 = self._ff(phi, phid, phidd)
        q1_ref = self.q1r0 + (f1 - self.ff0[0] if self.cfg.q1_hold else 0.0)
        q2_ref = self.q2r0 + (phi - self.phi0) + (f2 - self.ff0[1])
        return q1_ref, q2_ref

    @property
    def flipping(self) -> bool:
        return self.t0 is not None
