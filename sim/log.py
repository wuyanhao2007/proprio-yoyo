"""Phase log and event table (PHYSICS_SPEC §8).

One row per physics step. Field names are shared with the hardware logger so
sim and real logs can be overlaid directly during system identification.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

FIELDS = (
    "t", "mode", "ell", "elld", "theta", "thetad", "omega", "sigma", "T", "F_mag",
    "Hx", "Hy", "Hdx", "Hdy", "Hddx", "Hddy",
    "q1", "q2", "q1d", "q2d", "phi2", "q1_ref", "q2_ref",
    "tau_m1", "tau_m2", "i1", "i2", "i1_meas", "i2_meas", "q1_meas", "q2_meas",
    "F_Hx", "F_Hy", "F_bx", "F_by", "F_cx", "F_cy",
    "px", "py", "vx", "vy", "spin_angle", "delta", "tau_wrist_rope", "inside", "contact",
)


class SimLog:
    def __init__(self):
        self.rows: list[tuple] = []

    def clear(self):
        self.rows.clear()

    def record(self, s, dg, motor, sensors):
        self.rows.append((
            s.t, s.mode, s.ell, s.elld, s.theta, s.thetad, s.omega, s.sigma, dg.T, dg.F_mag,
            dg.Hx, dg.Hy, dg.Hdx, dg.Hdy, dg.Hddx, dg.Hddy,
            s.q1, s.phi2 - s.q1, s.q1d, s.phi2d - s.q1d, s.phi2, s.q1_ref, s.q2_ref,
            dg.tau_m1, dg.tau_m2, motor.i1, motor.i2, sensors.i1, sensors.i2,
            sensors.q1, sensors.q2,
            dg.Fhx, dg.Fhy, dg.Fbx, dg.Fby, dg.Fcx, dg.Fcy,
            s.px, s.py, s.vx, s.vy, s.spin_angle, dg.delta, dg.tau_wrist_rope,
            float(s.inside), float(dg.contact),
        ))

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.rows:
            return {k: np.zeros(0) for k in FIELDS}
        a = np.asarray(self.rows, dtype=float)
        return {k: a[:, i] for i, k in enumerate(FIELDS)}


def save_run(out_dir: str | Path, log: SimLog, events: list[dict], params, meta: dict | None = None) -> Path:
    """Write log.npz (all fields), events.csv/json and params.json into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "log.npz", **log.arrays())
    with open(out / "events.json", "w") as f:
        json.dump(events, f, indent=1, default=float)
    keys = sorted({k for e in events for k in e})
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for e in events:
            w.writerow(e)
    with open(out / "params.json", "w") as f:
        json.dump(params.as_dict(), f, indent=1)
    if meta:
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=1, default=str)
    return out


def default_run_dir(name: str) -> Path:
    from sim.params import REPO_ROOT
    return REPO_ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"


def load_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    data = dict(np.load(run_dir / "log.npz"))
    with open(run_dir / "events.json") as f:
        events = json.load(f)
    with open(run_dir / "params.json") as f:
        params = json.load(f)
    return data, events, params
