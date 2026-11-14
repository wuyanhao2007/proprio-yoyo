"""Proprioceptive sensor models (PHYSICS_SPEC §7).

Only these signals reach the policy: joint angles from the motor encoders,
differentiated + low-passed joint velocities, and the two q-axis currents.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from sim.params import Params


class Sensors:
    """Delay lines, quantization, noise and bias. Call ``update`` once per physics step."""

    def __init__(self, p: Params, rng: np.random.Generator | None = None, noise: bool = True):
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.noise = noise
        dt = p.dt
        self.n_enc = max(0, int(round(p.enc_delay / dt)))
        self.n_cur = max(0, int(round(p.i_delay / dt)))
        self.q_lsb = 2 * math.pi / p.enc_counts
        self.alpha = dt / (dt + 1.0 / (2 * math.pi * p.vel_cutoff))
        self.bias = (self.rng.uniform(-p.i_bias_max, p.i_bias_max, size=2) if noise else np.zeros(2))
        self._enc = deque(maxlen=self.n_enc + 1)
        self._cur = deque(maxlen=self.n_cur + 1)
        self._prev = None
        self.q1 = self.q2 = 0.0
        self.q1d = self.q2d = 0.0
        self.i1 = self.i2 = 0.0

    def reset(self, q1: float, q2: float, i1: float = 0.0, i2: float = 0.0):
        self._enc.clear()
        self._cur.clear()
        for _ in range(self.n_enc + 1):
            self._enc.append((q1, q2))
        for _ in range(self.n_cur + 1):
            self._cur.append((i1, i2))
        self._prev = None
        self.q1d = self.q2d = 0.0
        self.update(q1, q2, i1, i2)

    def _quantize_q1(self, q1: float) -> float:
        # encoder is on the motor: theta_m = N1 q1 (rigid belt); read back / N1
        n = self.p.N1
        return round(n * q1 / self.q_lsb) * self.q_lsb / n

    def _quantize_q2(self, q2: float) -> float:
        return round(q2 / self.q_lsb) * self.q_lsb

    def update(self, q1: float, q2: float, i1: float, i2: float):
        self._enc.append((q1, q2))
        self._cur.append((i1, i2))
        dq1, dq2 = self._enc[0]
        di1, di2 = self._cur[0]
        m1, m2 = self._quantize_q1(dq1), self._quantize_q2(dq2)
        if self._prev is not None:
            v1 = (m1 - self._prev[0]) / self.p.dt
            v2 = (m2 - self._prev[1]) / self.p.dt
            self.q1d += self.alpha * (v1 - self.q1d)
            self.q2d += self.alpha * (v2 - self.q2d)
        self._prev = (m1, m2)
        self.q1, self.q2 = m1, m2
        if self.noise:
            n = self.rng.normal(0.0, self.p.i_noise, size=2)
            self.i1 = di1 + self.bias[0] + n[0]
            self.i2 = di2 + self.bias[1] + n[1]
        else:
            self.i1, self.i2 = di1, di2

    def reading(self) -> tuple[float, float, float, float, float, float]:
        return self.q1, self.q2, self.q1d, self.q2d, self.i1, self.i2
