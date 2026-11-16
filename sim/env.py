"""Environment interface (PHYSICS_SPEC §8), numpy reference version.

    env   = YoyoEnv()
    state = env.reset(seed, params=None, start="stowed")
    state, obs, priv, reward, done, info = env.step(state, action)

Same signature as the planned JAX version. Here ``state`` wraps a mutable
Simulator and ``step`` updates it in place (and returns it); the JAX port
(roadmap stage 5) will make it a pure pytree.

Action (position mode): (dq1_ref, dq2_ref), increments of the joint references
relative to the current command, clipped to ``max_delta`` per step. The wrist
uses the relative angle q2, like the real firmware.
One env step = ``substeps`` physics steps (25 x 0.2 ms = 5 ms, 200 Hz).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from control.trajectories import Throw, funnel_down_refs
from sim.params import Params, load_randomization, sample_params
from sim.simulator import Simulator
from yoyo.modes import Mode

OBS_NAMES = ("q1", "q2", "q1d", "q2d", "i1", "i2", "q1_ref", "q2_ref", "a_prev1", "a_prev2")
PRIV_NAMES = ("mode", "ell", "elld", "theta", "thetad", "omega", "sigma", "T",
              "t_to_bottom", "ell_min", "zeta_e", "kappa")


@dataclass
class EnvState:
    sim: Simulator
    q_cmd: np.ndarray                 # current joint-reference command (q1_ref, q2_ref)
    a_prev: np.ndarray
    t_limit: float
    steps: int = 0
    done: bool = False
    hold_start: float = -1.0
    extras: dict = field(default_factory=dict)

    @property
    def t(self) -> float:
        return self.sim.s.t


def predict_time_to_bottom(p: Params, ell: float, elld: float) -> float:
    """Time until the next bottom for a fixed hook and constant kappa (§3.7)."""
    a = p.kappa() * p.g
    if elld < 0.0:  # climbing: time to the apex, then fall from it
        t_up = -elld / a
        ell_top = ell - elld * elld / (2 * a)
        return t_up + math.sqrt(max(2 * (p.L - ell_top) / a, 0.0))
    disc = elld * elld + 2 * a * max(p.L - ell, 0.0)
    return (-elld + math.sqrt(disc)) / a


class YoyoEnv:
    def __init__(self, params: Params | None = None, randomize: bool = False,
                 rand_scale: float = 1.0, max_delta: float = 0.05,
                 episode_time: float = 12.0, hold_time: float = 2.0,
                 sensor_noise: bool = True, log: bool = False,
                 reward_fn: Callable | None = None):
        self.nominal = params if params is not None else Params.load()
        self.randomize = randomize
        self.rand_scale = rand_scale
        self.ranges = load_randomization(hand=self.nominal.catch_mode)
        self.max_delta = max_delta
        self.episode_time = episode_time
        self.hold_time = hold_time
        self.sensor_noise = sensor_noise
        self.log = log
        self.reward_fn = reward_fn or (lambda state, info: 0.0)

    # --------------------------------------------------------------- reset
    def reset(self, seed: int | None = None, params: Params | None = None,
              start: str = "stowed", k_bottom: int = 1) -> EnvState:
        """start = "stowed" | "after_bottom" (fast-forwards a scripted throw and
        k_bottom passive bounces, for skill-wise training, §8)."""
        rng = np.random.default_rng(seed)
        p = params if params is not None else self.nominal
        if self.randomize and params is None:
            p = sample_params(self.nominal, rng, self.ranges, self.rand_scale)
        sim = Simulator(p, rng=rng, sensor_noise=self.sensor_noise, log=self.log)
        q1r, q2r = funnel_down_refs(0.0)
        sim.reset_stowed(q1r, q2r)
        state = EnvState(sim=sim, q_cmd=np.array([q1r, q2r]), a_prev=np.zeros(2),
                         t_limit=self.episode_time)
        if start == "after_bottom":
            throw = Throw(p.l1)
            while sim.s.n_bottom < k_bottom and sim.s.mode != Mode.FAIL and sim.s.t < 10.0:
                state.q_cmd[:] = throw(sim)
                sim.set_reference(*state.q_cmd)
                for _ in range(p.substeps):
                    sim.step()
            state.t_limit = sim.s.t + self.episode_time
        elif start != "stowed":
            raise ValueError(f"unknown start '{start}'")
        return state

    # ---------------------------------------------------------------- step
    def step(self, state: EnvState, action):
        sim, p = state.sim, state.sim.p
        a = np.clip(np.asarray(action, dtype=float), -self.max_delta, self.max_delta)
        state.q_cmd = state.q_cmd + a
        sim.set_reference(*state.q_cmd)
        n_ev0 = len(sim.events)
        T_peak = 0.0
        tau1_peak = tau2_peak = wrist_peak = 0.0
        for _ in range(p.substeps):
            dg = sim.step()
            T_peak = max(T_peak, dg.T)
            tau1_peak = max(tau1_peak, abs(dg.tau_m1))
            tau2_peak = max(tau2_peak, abs(dg.tau_m2))
            wrist_peak = max(wrist_peak, abs(dg.tau_wrist_rope))
            if sim.s.mode == Mode.FAIL:
                break
        state.steps += 1
        state.a_prev = a
        events = sim.events[n_ev0:]
        s = sim.s
        # success: caught after a throw and held for hold_time
        if s.mode == Mode.STOWED and s.released and s.caught:
            if state.hold_start < 0:
                state.hold_start = s.t
        else:
            state.hold_start = -1.0
        success = state.hold_start >= 0 and s.t - state.hold_start >= self.hold_time
        failed = s.mode == Mode.FAIL
        timeout = s.t >= state.t_limit
        state.done = bool(success or failed or timeout)
        info = {
            "events": events,
            "T_peak": T_peak,
            "tau_m1_peak": tau1_peak,
            "tau_m2_peak": tau2_peak,
            "tau_wrist_rope_peak": wrist_peak,
            "n_bottom": s.n_bottom,
            "success": bool(success),
            "fail_reason": s.fail_reason,
            "timeout": bool(timeout and not (success or failed)),
        }
        reward = float(self.reward_fn(state, info))
        return state, self.observe(state), self.privileged(state), reward, state.done, info

    # ------------------------------------------------------------ observations
    def observe(self, state: EnvState) -> np.ndarray:
        """§7 proprioception only: measured angles/velocities/currents + command."""
        q1, q2, q1d, q2d, i1, i2 = state.sim.sensors.reading()
        return np.array([q1, q2, q1d, q2d, i1, i2, *state.q_cmd, *state.a_prev])

    def privileged(self, state: EnvState) -> np.ndarray:
        """§7 privileged observation for critic / teacher / logs."""
        s, p, dg = state.sim.s, state.sim.p, state.sim.diag
        ttb = predict_time_to_bottom(p, s.ell, s.elld) if s.mode == Mode.TAUT else -1.0
        return np.array([s.mode, s.ell, s.elld, s.theta, s.thetad, s.omega, s.sigma, dg.T,
                         ttb, p.ell_min, p.zeta_e, p.kappa()])
