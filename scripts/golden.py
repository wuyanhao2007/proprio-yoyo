"""Golden-log regression check for the magnet-hand full flow.

Refactors that add the flip hand must not change the magnet hand at all. Record
the per-step logs once on a known-good commit, then compare bit by bit:

    git stash / checkout the reference commit
    python -m scripts.golden record            # writes runs/_ref_magnet/seed{0,1,2}.npz
    ... come back to the working tree ...
    python -m scripts.golden check             # np.array_equal on every log field + events

``runs/`` is git-ignored, so the reference lives only on this machine.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from control.baseline import BaselineConfig, ZlajpahBaseline
from sim.params import REPO_ROOT, Params
from sim.simulator import Simulator

REF = REPO_ROOT / "runs" / "_ref_magnet"
SEEDS = (0, 1, 2)


def run(seed: int, duration: float = 11.0):
    """Exactly what tests/test_baseline.py and ``run_demo full_flow`` do for the magnet hand."""
    p = Params.load(i_peak_ratio=2.0)
    sim = Simulator(p, rng=np.random.default_rng(seed))
    sim.reset_stowed()
    sim.run(duration, ZlajpahBaseline(p, BaselineConfig()))
    events = json.loads(json.dumps(sim.events, default=float))
    return sim.log.arrays(), events


def record():
    REF.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        log, events = run(seed)
        np.savez(REF / f"seed{seed}.npz", **log)
        with open(REF / f"seed{seed}_events.json", "w") as f:
            json.dump(events, f)
        print(f"seed {seed}: recorded {len(log['t'])} steps, {len(events)} events")


def check() -> bool:
    ok = True
    for seed in SEEDS:
        ref = np.load(REF / f"seed{seed}.npz")
        with open(REF / f"seed{seed}_events.json") as f:
            ref_events = json.load(f)
        log, events = run(seed)
        bad = [k for k in ref.files if k not in log or not np.array_equal(ref[k], log[k], equal_nan=True)]
        bad += [k for k in log if k not in ref.files]
        same_events = events == ref_events
        ok &= not bad and same_events
        print(f"seed {seed}: {len(ref.files)} fields, "
              f"{'bit-identical' if not bad else 'DIFFERENT: ' + ', '.join(bad)}; "
              f"events {'identical' if same_events else 'DIFFERENT'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["record", "check"])
    a = ap.parse_args()
    if a.action == "record":
        record()
    elif not check():
        sys.exit(1)


if __name__ == "__main__":
    main()
