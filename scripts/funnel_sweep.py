"""Capture map of the funnel model: outcome vs. swing angle and climb speed.

A first look at roadmap stage 2 "funnel shape": the yo-yo starts at l0 below a
still arm, climbing at v0 with a swing angle theta0, and we record whether it
ends STOWED, FAIL (hit the rim), or falls back out (no catch).

    python -m scripts.funnel_sweep
    python -m scripts.funnel_sweep --set mu_wall=0.1 flare_exponent=1
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from control.trajectories import Hold
from sim.params import Params
from sim.simulator import Simulator
from yoyo.modes import Mode

SYMBOL = {"stowed": "S", "fail": "x", "out": "."}


def outcome(p: Params, theta0_deg: float, v0: float, ell0: float = 0.40, t_max: float = 1.4) -> str:
    sim = Simulator(p, sensor_noise=False, log=False, rng=np.random.default_rng(0))
    sim.reset_taut(ell=ell0, elld=-v0, theta=math.radians(theta0_deg), sigma=-1)
    sim.run(t_max, Hold(0.0))
    if sim.s.caught:
        return "stowed"
    if sim.s.mode == Mode.FAIL:
        return "fail"
    return "out"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--thetas", default="0,2,4,6,8,10,12,15")
    ap.add_argument("--speeds", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00")
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            over[k] = float(v)
        except ValueError:
            over[k] = v
    p = Params.load(**over)
    thetas = [float(x) for x in a.thetas.split(",")]
    speeds = [float(x) for x in a.speeds.split(",")]
    kg = p.kappa() * p.g
    print(f"capture_model={p.capture_model} mu_wall={p.mu_wall} flare_exponent={p.flare_exponent} "
          f"seat={p.seat_width * 1e3:.0f}mm mouth={p.mouth_width * 1e3:.0f}mm")
    print("rows: swing angle at l0=0.40 m; columns: climb speed at l0 "
          "(-> ideal arrival speed at the seat). S = stowed, x = hit rim, . = fell back out\n")
    head = "theta0 | " + " ".join(f"{v:4.2f}" for v in speeds)
    arr = "       | " + " ".join(
        f"{math.sqrt(max(v * v - 2 * kg * (0.40 - p.ell_min), 0)):4.2f}" for v in speeds)
    print(head)
    print(arr + "   <- arrival speed without the funnel [m/s]")
    print("-" * len(head))
    stats = {"stowed": 0, "fail": 0, "out": 0}
    for th in thetas:
        row = []
        for v in speeds:
            o = outcome(p, th, v)
            stats[o] += 1
            row.append(f"  {SYMBOL[o]}  ")
        print(f"{th:5.1f}° | " + "".join(row))
    n = sum(stats.values())
    print(f"\ncaught {stats['stowed']}/{n}, rim {stats['fail']}/{n}, fell out {stats['out']}/{n}")


if __name__ == "__main__":
    main()
