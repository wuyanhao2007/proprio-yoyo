"""Full flow for either hand: throw -> n bounces -> catch (-> flip stow).

One place that knows how the pieces chain, so scripts and tests do not repeat it:

    magnet  ZlajpahBaseline alone; the magnet seats the yo-yo (STOWED).
    flip    ZlajpahBaseline until the yo-yo sits in the seat (IN_CUP), then
            StowFlip turns the hand mouth-up, direction "auto" = sign sigma (A9).

    sim = Simulator(p); ctrl, base = make_full_flow(p); reset_full_flow(sim); sim.run(T, ctrl)
"""
from __future__ import annotations

import math

import numpy as np

from control.baseline import BaselineConfig, ZlajpahBaseline
from control.stow_flip import StowFlip, StowFlipConfig
from sim.params import Params


def make_full_flow(p: Params, cfg: BaselineConfig | None = None,
                   stow_cfg: StowFlipConfig | None = None):
    """Returns ``(controller, baseline)``: run the first, read pulse records from the second."""
    base = ZlajpahBaseline(p, cfg if cfg is not None else BaselineConfig.load(p.catch_mode))
    if p.catch_mode != "flip":
        return base, base
    return StowFlip(p, stow_cfg if stow_cfg is not None else StowFlipConfig.load(), before=base), base


def reset_full_flow(sim):
    """Start posture: stowed; the flip hand holds the yo-yo by gravity, mouth up."""
    if sim.p.catch_mode == "flip":
        return sim.reset_stowed(q2_ref=math.pi / 2)
    return sim.reset_stowed()


def catch_report(sim) -> dict:
    """Numbers that judge a catch, from the events and the per-step log:
    seat arrival speed [m/s], string swing angle at arrival [deg], peak wrist
    current overall and during the flip [A] (limit: Params.i_max2)."""
    d = sim.log.arrays()
    out = {"i2_peak": float(np.abs(d["i2"]).max()), "i2_max": sim.p.i_max2}
    arrive = next((e for e in sim.events if e["type"] in ("in_cup", "stowed")), None)
    if arrive is not None:
        k = max(int(np.searchsorted(d["t"], arrive["t"])) - 1, 0)   # last step before the seat event
        out.update(t_arrive=arrive["t"], v_arrive=float(arrive["v"]),
                   theta_arrive_deg=math.degrees(float(d["theta"][k])))
    flip = next((e for e in sim.events if e["type"] == "flip_start"), None)
    if flip is not None:
        w = (d["t"] >= flip["t"]) & (d["t"] <= flip["t"] + flip["t_flip"] + 0.05)
        out.update(t_flip_start=flip["t"], flip_direction=flip["direction"],
                   i2_peak_flip=float(np.abs(d["i2"][w]).max()))
    stowed = next((e for e in sim.events if e["type"] == "stowed"), None)
    if stowed is not None:
        out["t_stowed"] = stowed["t"]
    return out
