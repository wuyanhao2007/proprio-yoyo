"""Run a demo scenario, save the log and render the animation.

Scenarios
  throw_bounce   stowed -> open-loop throw -> passive bounces with the arm holding
                 still (no energy pumping yet: that is roadmap stage 1 step 9).
  funnel_catch   yo-yo already climbing towards the funnel with a swing offset,
                 to look at the funnel model (wall guidance, magnet, seat).
  full_flow      roadmap stage 1 step 9: throw -> 3 bottoms with Zlajpah-style
                 hook pulses (true-state baseline) -> the 3rd pulse sends the
                 yo-yo back into the funnel. Uses i_peak_ratio = 2 (2x continuous
                 current, short-term) unless overridden with --set.
                 --hand flip: starts mouth up, the throw turns the mouth down,
                 and after the catch control/stow_flip.py turns it up again.
  flip_stow      flip hand (§4.4): yo-yo just arrived in the seat, mouth down;
                 control/stow_flip.py turns the hand 180 deg (--t-flip, --direction).
  flip_catch     flip hand: yo-yo climbing into the cup (like funnel_catch, no
                 magnet), then the same flip once it sits in the seat.
  The flip scenarios load config/hand_flip.yaml and use i_peak_ratio = 2.

Examples
  python -m scripts.run_demo throw_bounce
  python -m scripts.run_demo funnel_catch --theta0 6 --v0 0.85
  python -m scripts.run_demo funnel_catch --capture-model gate --speed 0.25
  python -m scripts.run_demo throw_bounce --no-anim          # log + PNG only
  python -m scripts.run_demo full_flow
  python -m scripts.run_demo full_flow --hand flip
  python -m scripts.run_demo full_flow --n-bounces 0 --duration 30   # keep bouncing
  python -m scripts.run_demo flip_stow --speed 0.25
  python -m scripts.run_demo flip_stow --direction wrong --speed 0.25   # unwinds: drops out
  python -m scripts.run_demo flip_catch --theta0 0 --sigma -1
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from control.baseline import BaselineConfig
from control.full_flow import catch_report, make_full_flow, reset_full_flow
from control.stow_flip import StowFlip, StowFlipConfig
from control.trajectories import Hold, Throw
from sim.log import default_run_dir, save_run
from sim.params import Params
from sim.simulator import Simulator
from yoyo.modes import MODE_NAMES


def build(args, p: Params):
    """Returns (sim, controller, duration, baseline or None)."""
    base = None
    rng = np.random.default_rng(args.seed)
    sim = Simulator(p, rng=rng, sensor_noise=not args.no_noise)
    if args.scenario == "throw_bounce":
        sim.reset_stowed()
        ctrl = Throw(p.l1)
        duration = args.duration or 7.0
    elif args.scenario == "funnel_catch":
        sim.reset_taut(ell=args.ell0, elld=-args.v0, theta=math.radians(args.theta0), sigma=-1)
        ctrl = Hold(0.0)
        duration = args.duration or 1.6
    elif args.scenario == "full_flow":
        reset_full_flow(sim)
        ctrl, base = make_full_flow(p, BaselineConfig.load(p.catch_mode, n_bounces=args.n_bounces))
        duration = args.duration or 11.0
    elif args.scenario in ("flip_stow", "flip_catch"):
        sigma = args.sigma
        d = {"auto": "auto", "wrong": -sigma}.get(args.direction, args.direction)
        over = {"t_flip": args.t_flip} if args.t_flip is not None else {}
        cfg = StowFlipConfig.load(direction=int(d) if d != "auto" else d, **over)
        if args.scenario == "flip_stow":
            sim.reset_in_cup(q2_ref=-math.pi / 2, sigma=sigma)
            ctrl = StowFlip(p, cfg)
            duration = args.duration or 1.0
        else:
            sim.reset_taut(ell=args.ell0, elld=-args.v0, theta=math.radians(args.theta0), sigma=sigma)
            ctrl = StowFlip(p, cfg, before=Hold(0.0))
            duration = args.duration or 1.6
    else:
        raise SystemExit(f"unknown scenario {args.scenario}")
    return sim, ctrl, duration, base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=["throw_bounce", "funnel_catch", "full_flow",
                                         "flip_stow", "flip_catch"])
    ap.add_argument("--hand", choices=["magnet", "flip"], default=None,
                    help="hand overlay config/hand_<hand>.yaml (flip_* scenarios: flip)")
    ap.add_argument("--t-flip", type=float, default=None, help="flip_*: flip duration [s]")
    ap.add_argument("--direction", default="auto", help="flip_*: auto | wrong | 1 | -1")
    ap.add_argument("--sigma", type=int, default=-1, choices=[-1, 1],
                    help="flip_*: winding direction of the arriving yo-yo")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noise", action="store_true", help="ideal sensors")
    ap.add_argument("--capture-model", choices=["geometry", "gate", "none"], default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any parameter, e.g. --set mu_wall=0.1 flare_exponent=1")
    ap.add_argument("--ell0", type=float, default=0.40, help="funnel_catch: start string length")
    ap.add_argument("--v0", type=float, default=0.85, help="funnel_catch: start climb speed")
    ap.add_argument("--theta0", type=float, default=6.0, help="funnel_catch: start swing angle [deg]")
    ap.add_argument("--n-bounces", type=int, default=3,
                    help="full_flow: bottoms before/including the catch (0 = never catch)")
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--speed", type=float, default=None, help="playback speed (default 1, catch 0.25)")
    ap.add_argument("--out", default=None, help="run directory (default runs/<time>_<scenario>)")
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = v if not _is_number(v) else float(v)
    if args.capture_model:
        overrides["capture_model"] = args.capture_model
    if args.scenario in ("full_flow", "flip_stow", "flip_catch"):
        overrides.setdefault("i_peak_ratio", 2.0)
    hand = args.hand or ("flip" if args.scenario.startswith("flip_") else None)
    p = Params.load(hand=hand, **overrides)
    sim, ctrl, duration, base = build(args, p)

    t0 = time.time()
    sim.run(duration, ctrl, stop_on_fail=False)
    if args.scenario == "full_flow":
        _print_pulses(base, sim, p)
        _print_catch(sim)
    wall = time.time() - t0
    out = Path(args.out) if args.out else default_run_dir(args.scenario)
    save_run(out, sim.log, sim.events, p, meta=vars(args))
    print(f"simulated {duration:.2f} s in {wall:.2f} s wall -> {out}")
    print(f"final mode {MODE_NAMES[sim.s.mode]} {sim.s.fail_reason}")
    for ev in sim.events:
        if ev["type"] in ("release", "bottom", "stowed", "fail", "seat_rebound", "mouth",
                          "in_cup", "unseat", "drop", "flip_start"):
            extra = {k: round(v, 4) for k, v in ev.items()
                     if k in ("v_minus", "J", "F_in", "v", "beta", "reason") and isinstance(v, float)}
            extra.update({k: v for k, v in ev.items() if k == "reason"})
            print(f"  t={ev['t']:7.4f}  {ev['type']:13s} {extra}")

    from viz.animate import animate, snapshot
    d = sim.log.arrays()
    t_snap = next((e["t"] for e in sim.events if e["type"] in ("stowed", "fail")), d["t"][-1])
    snapshot(d, sim.events, p, t_snap, out / "summary.png", title=f"{args.scenario}")
    if not args.no_anim:
        speed = args.speed or (0.25 if args.scenario in ("funnel_catch", "flip_stow", "flip_catch")
                               else 1.0)
        animate(d, sim.events, p, out=out / "anim.gif", fps=args.fps, speed=speed,
                title=f"{args.scenario}  (playback x{speed:g})")


def _print_pulses(ctrl, sim, p):
    print(f"i_peak_ratio = {p.i_peak_ratio:g}; time above continuous current: "
          f"joint 1 {sim.motor.t_over1:.3f} s, wrist {sim.motor.t_over2:.3f} s")
    print(" k  catch  v_bottom  v_max   eta   KE-     KE+    KE_target  apex (target "
          f"{ctrl.cfg.apex_target:.2f} m)")
    for r in ctrl.pulses:
        print(f"{r.k:2d}  {'yes' if r.catch else ' no'}    {r.v:5.2f}   {r.v_max:5.2f}  "
              f"{r.eta_used:5.2f}  {r.KE_minus:.3f}  {r.KE_plus:.3f}   {r.KE_target:.3f}     "
              f"{r.apex:.3f}")


def _print_catch(sim):
    r = catch_report(sim)
    if "v_arrive" not in r:
        print("catch: the yo-yo did not reach the seat")
        return
    print(f"catch: seat arrival t = {r['t_arrive']:.3f} s, speed {r['v_arrive']:.3f} m/s, "
          f"swing angle {r['theta_arrive_deg']:+.1f} deg")
    line = f"wrist current peak {r['i2_peak']:.2f} A (limit {r['i2_max']:.2f} A)"
    if "i2_peak_flip" in r:
        line += f"; during the flip (direction {r['flip_direction']:+d}) {r['i2_peak_flip']:.2f} A"
    print(line)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
