"""Hand-written joint reference generators (roadmap stage 1).

Controllers are callables ``ctrl(sim) -> (q1_ref, q2_ref)`` evaluated at the
policy rate (200 Hz). They read the simulator clock only (open loop), so the
same objects can drive the real arm later.

Bounce posture (§5.1): link 1 about horizontal, link 2 (funnel) pointing down,
the wrist counter-rotates q2 = -pi/2 - q1 so the funnel stays vertical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def funnel_down_refs(q1: float) -> tuple[float, float]:
    return q1, -math.pi / 2 - q1


def q1_for_hook_drop(l1: float, dy: float) -> float:
    """Joint-1 angle that moves the wrist axis by dy vertically from horizontal."""
    return math.asin(max(-1.0, min(1.0, dy / l1)))


@dataclass
class Hold:
    q1: float = 0.0

    def __call__(self, sim):
        return funnel_down_refs(self.q1)


@dataclass
class Throw:
    """Open-loop throw (§4.1): accelerate the funnel downwards, then stop hard.

    The hook reference follows y(t) = -acc (t - t0)^2 / 2 for t_acc seconds and
    then freezes. The PD saturates while braking; the upward deceleration pulls
    the yo-yo out of the magnet once m (a_up + g) > F_mag0 (~10.2 m/s^2 nominal).
    Parameters are hand-picked; stage 2 optimizes them with CMA-ES.
    """
    l1: float
    t0: float = 0.2
    acc: float = 6.0
    t_acc: float = 0.12
    q1_start: float = 0.0
    q1_after: float | None = None   # posture to settle to after the throw
    t_settle: float = 0.4

    def hook_offset(self, t: float) -> float:
        tau = min(max(t - self.t0, 0.0), self.t_acc)
        return -0.5 * self.acc * tau * tau

    def __call__(self, sim):
        t = sim.s.t
        q1 = self.q1_start + q1_for_hook_drop(self.l1, self.hook_offset(t))
        if self.q1_after is not None and t > self.t0 + self.t_acc + self.t_settle:
            q1 = self.q1_after
        return funnel_down_refs(q1)


@dataclass
class ThrowFlip:
    """Open-loop throw of the flip hand (§4.4): no magnet, so the throw is "turn the
    mouth from up to down" and the yo-yo slides out by itself (events unseat ->
    release). Joint 1 holds still; the absolute funnel angle follows a raised-cosine
    velocity profile from ``phi_up`` over ``t_turn`` seconds.

    ``direction`` = sign of the turn (-1: the mouth passes the +x side, away from
    link 1). With an even number of bottoms the stow flip returns on the same side,
    so the cup never passes link 1. TODO(sysid): the best t_turn; a slow turn pours the yo-yo out with little sideways speed.
    """
    t0: float = 0.2
    t_turn: float = 0.3
    direction: int = -1
    q1: float = 0.0
    phi_up: float = math.pi / 2
    lead: float = 0.0               # evaluate the profile this far ahead (command latency)

    def __post_init__(self):
        from yoyo.catch_flip import cosine_flip
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be 1 or -1, got {self.direction!r}")
        self._profile = cosine_flip(self.t_turn, phi0=self.phi_up, direction=self.direction, t0=self.t0)

    @property
    def t_end(self) -> float:
        return self.t0 + self.t_turn

    @property
    def phi_down(self) -> float:
        """Absolute funnel angle after the throw (mouth down, = -pi/2 mod 2 pi)."""
        return self.phi_up + self.direction * math.pi

    def hook_offset(self, t: float) -> float:
        return 0.0                  # the wrist axis does not move during this throw

    def __call__(self, sim):
        phi2 = self._profile(sim.s.t + self.lead)[0]
        return self.q1, phi2 - self.q1
