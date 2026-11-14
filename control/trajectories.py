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
