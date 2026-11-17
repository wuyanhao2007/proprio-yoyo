"""Roadmap stage 2, step "mechanics sweep": N randomized full flows per hand.

Every case draws one parameter set from the ``randomization`` block of
config/params.yaml, runs throw -> bounces -> catch (-> flip stow; 3 bottoms for the
magnet hand, 4 for the flip hand) with the nominal controller, and reduces the per-step log to one row of load metrics:

    T_peak        string tension peak [N]
    tau_j1_peak   joint-1 torque peak at the joint, N1*eta1*|tau_m1| [N m]
    tau_m1_ratio  |tau_m1| peak / continuous motor torque tau_max1
    tau_m2_peak   wrist motor torque peak [N m] (direct drive), tau_m2_ratio likewise
    rpm1_peak     joint-1 motor speed peak, N1*|q1d| [rpm];  rpm2_peak: wrist
    hook_dv       worst hook speed lost during a bottom pulse [m/s]
    hook_drop     worst distance the hook ends up below "coasting at its speed
                  at the bottom" during a bottom pulse [m]
    t_over1/2     time with motor torque above continuous [s]

and to a staged score instead of only stowed / not stowed. Four stages, 0.25
each, counted in order (a stage only scores if every earlier one did):

    throw     the yo-yo is released and reaches the first bottom
    bounce1   first bounce: apex no lower than APEX_TOL below apex_target and not
              in the seat (ell > ell_min + s_max), then the second bottom
    bounces   every later bounce likewise, up to the last (n_bounces-th) bottom
    catch     STOWED at the end

    python -m scripts.mech_sweep run  --n 1000            # both hands, all cores
    python -m scripts.mech_sweep plot                     # figures from results.csv
    python -m scripts.mech_sweep curve --names a b c --name curve   # stage rates vs --scale
    python -m scripts.mech_sweep run  --n 20 --name smoke # quick check

Results go to runs/sweeps/<name>/ (git-ignored). Case k uses the RNG stream
(seed, k) whatever the worker or the order, and both hands get the same draw.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from control.full_flow import catch_report, make_full_flow, reset_full_flow
from sim.params import REPO_ROOT, Params, load_randomization, sample_params
from sim.simulator import Simulator
from yoyo.modes import Mode

HANDS = ("magnet", "flip")
DURATION = {"magnet": 11.0, "flip": 13.0}       # as in tests/test_baseline.py
RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)
APEX_TOL = 0.10                                   # m, "the bounce worked" band around apex_target
STAGES = ("throw", "bounce1", "bounces", "catch")
METRICS = ("T_peak", "tau_j1_peak", "tau_m1_ratio", "tau_m2_peak", "tau_m2_ratio",
           "rpm1_peak", "rpm2_peak", "hook_dv", "hook_drop", "t_over1", "t_over2")


# ----------------------------------------------------------------- one case
def draw_params(hand: str, seed: int, k: int, scale: float, i_peak_ratio: float) -> tuple[Params, dict]:
    """Randomized parameters of case k. Both hands consume the same random numbers
    (the base ranges, in file order); entries the hand overlay removes go back to
    the hand's nominal value afterwards."""
    nominal = Params.load(hand=hand, i_peak_ratio=i_peak_ratio)
    base = load_randomization()
    rng = np.random.default_rng([seed, k])
    p = sample_params(nominal, rng, base, scale)
    dropped = [name for name in base if name not in load_randomization(hand=hand)]
    if dropped:
        p = p.replace(**{name: getattr(nominal, name) for name in dropped})
    drawn = {name: getattr(p, name) for name in base if name not in dropped}
    return p, drawn


def load_metrics(p: Params, d: dict[str, np.ndarray], events: list[dict]) -> dict:
    """Reduce one per-step log to the load metrics of the module docstring."""
    dt = p.dt
    tm1, tm2 = np.abs(d["tau_m1"]), np.abs(d["tau_m2"])
    out = {
        "T_peak": float(d["T"].max()),
        "tau_j1_peak": float(p.N1 * p.eta1 * tm1.max()),
        "tau_m1_ratio": float(tm1.max() / p.tau_max1),
        "tau_m2_peak": float(tm2.max()),
        "tau_m2_ratio": float(tm2.max() / p.tau_max2),
        "rpm1_peak": float(p.N1 * np.abs(d["q1d"]).max() * RAD_S_TO_RPM),
        "rpm2_peak": float(np.abs(d["q2d"]).max() * RAD_S_TO_RPM),
        "t_over1": float((tm1 > p.tau_max1).sum() * dt),
        "t_over2": float((tm2 > p.tau_max2).sum() * dt),
    }
    dv, drop = 0.0, 0.0
    for e in events:
        if e["type"] != "bottom":
            continue
        k0 = int(np.searchsorted(d["t"], e["t"]))
        k1 = min(k0 + int(round((e["pulse_dt"] + 0.010) / dt)), len(d["t"]) - 1)
        if k1 <= k0:
            continue
        t = d["t"][k0:k1 + 1] - d["t"][k0]
        v0 = d["Hdy"][k0]
        dv = max(dv, float(v0 - d["Hdy"][k0:k1 + 1].min()))
        drop = max(drop, float((d["Hy"][k0] + v0 * t - d["Hy"][k0:k1 + 1]).max()))
    out.update(hook_dv=dv, hook_drop=drop)
    return out


def stage_scores(p: Params, cfg, d: dict[str, np.ndarray], events: list[dict], stowed: bool) -> dict:
    """Staged score of the module docstring: s_<stage> in {0, 1}, ``score`` = 0.25 * how many
    stages passed in a row, plus the apex string length of every bounce [m]."""
    t_bot = [e["t"] for e in events if e["type"] == "bottom"][:cfg.n_bounces]
    apex = []
    for ta, tb in zip(t_bot[:-1], t_bot[1:]):
        ka, kb = np.searchsorted(d["t"], [ta, tb])
        apex.append(float(d["ell"][ka:kb + 1].min()))
    good = [p.ell_min + p.s_max < a <= cfg.apex_target + APEX_TOL for a in apex]
    released = any(e["type"] == "release" for e in events)
    passed = [
        released and len(t_bot) >= 1,
        len(good) >= 1 and good[0],
        len(t_bot) == cfg.n_bounces and all(good[1:]),
        stowed,
    ]
    out, alive = {}, True
    for name, ok in zip(STAGES, passed):
        alive = alive and bool(ok)
        out[f"s_{name}"] = int(alive)
    out["score"] = 0.25 * sum(out[f"s_{name}"] for name in STAGES)
    for i in range(cfg.n_bounces - 1):
        out[f"apex{i + 1}"] = apex[i] if i < len(apex) else float("nan")
    return out


def run_case(job: tuple) -> dict:
    """Top-level so ProcessPoolExecutor can pickle it. ``job`` = (hand, seed, k, scale, i_peak_ratio)."""
    hand, seed, k, scale, i_peak_ratio = job
    p, drawn = draw_params(hand, seed, k, scale, i_peak_ratio)
    sim = Simulator(p, rng=np.random.default_rng([seed, k, 1]))
    ctrl, base = make_full_flow(p)
    reset_full_flow(sim)
    row = {"hand": hand, "case": k, "scale": scale}
    try:
        sim.run(DURATION[hand], ctrl)
    except Exception as exc:                      # one bad draw must not kill the sweep
        row.update(outcome="error", fail_reason=f"{type(exc).__name__}: {exc}")
        row.update({f"p_{name}": v for name, v in drawn.items()})
        return row
    s = sim.s
    stowed = s.mode == Mode.STOWED and s.caught
    row["outcome"] = "stowed" if stowed else ("fail" if s.mode == Mode.FAIL else "timeout")
    row["n_bottom"] = sum(e["type"] == "bottom" for e in sim.events)
    if s.mode == Mode.FAIL:
        row["fail_reason"] = s.fail_reason
    elif stowed:
        row["fail_reason"] = ""
    else:   # ran out of time: where did the flow stop?
        row["fail_reason"] = "bounce_died" if row["n_bottom"] < 3 else "no_catch"
    row["t_end"] = s.t
    rep = catch_report(sim)
    row["v_arrive"] = rep.get("v_arrive", float("nan"))
    row["theta_arrive_deg"] = rep.get("theta_arrive_deg", float("nan"))
    d = sim.log.arrays()
    row.update(stage_scores(p, base.cfg, d, sim.events, stowed))
    row.update(load_metrics(p, d, sim.events))
    row.update({f"p_{name}": v for name, v in drawn.items()})
    return row


# --------------------------------------------------------------------- sweep
def sweep(n: int, hands, seed: int, scale: float, i_peak_ratio: float, workers: int, out: Path):
    jobs = [(hand, seed, k, scale, i_peak_ratio) for hand in hands for k in range(n)]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(run_case, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} cases, {time.time() - t0:.0f} s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'results.csv'}")
    return rows


def read_results(out: Path) -> list[dict]:
    rows = []
    with open(out / "results.csv") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k not in ("hand", "outcome", "fail_reason"):
                    r[k] = float(v) if v not in ("", None) else float("nan")
            rows.append(r)
    return rows


def summarize(rows: list[dict]) -> str:
    lines = []
    for hand in dict.fromkeys(r["hand"] for r in rows):
        rs = [r for r in rows if r["hand"] == hand]
        ok = [r for r in rs if r["outcome"] == "stowed"]
        lines.append(f"\n== {hand}: {len(ok)}/{len(rs)} stowed ({100 * len(ok) / len(rs):.1f} %)")
        done = [r for r in rs if r["outcome"] != "error"]
        lines.append("   stages passed: " + ", ".join(
            f"{name} {100 * np.mean([r[f's_{name}'] for r in done]):.1f} %" for name in STAGES)
            + f";  mean score {np.mean([r['score'] for r in done]):.3f}")
        reasons: dict[str, int] = {}
        for r in rs:
            if r["outcome"] != "stowed":
                key = r["fail_reason"] or r["outcome"]
                reasons[key] = reasons.get(key, 0) + 1
        for key, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"   {c:5d}  {key}")
        lines.append(f"   {'metric':14s} {'median':>9s} {'p95':>9s} {'max':>9s}   (all runs)")
        for m in METRICS:
            x = np.array([r[m] for r in rs if r["outcome"] != "error"], dtype=float)
            x = x[np.isfinite(x)]
            if len(x):
                lines.append(f"   {m:14s} {np.median(x):9.3f} {np.percentile(x, 95):9.3f} {x.max():9.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------- plots
C_OK, C_BAD, C_INK, C_MUTED, C_GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
PANELS = (("T_peak", "string tension peak [N]"),
          ("tau_m1_ratio", "joint-1 motor torque peak / continuous"),
          ("tau_m2_ratio", "wrist motor torque peak / continuous"),
          ("rpm1_peak", "joint-1 motor speed peak [rpm]"),
          ("rpm2_peak", "wrist motor speed peak [rpm]"),
          ("hook_dv", "hook speed lost at a bottom [m/s]"),
          ("hook_drop", "hook pulled below coasting at a bottom [m]"),
          ("t_over1", "joint-1 time above continuous torque [s]"))


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.grid(axis="y", color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_distributions(rows, hand, out: Path):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [r for r in rows if r["hand"] == hand and r["outcome"] != "error"]
    ok = np.array([r["outcome"] == "stowed" for r in rs])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), facecolor="#fcfcfb")
    for ax, (m, label) in zip(axes.ravel(), PANELS):
        x = np.array([r[m] for r in rs], dtype=float)
        good = np.isfinite(x)
        hi = np.percentile(x[good], 99.5)                    # a few outliers would flatten the rest
        bins = np.linspace(x[good].min(), hi + 1e-12, 36)
        x = np.minimum(x, hi)
        ax.hist([x[good & ok], x[good & ~ok]], bins=bins, stacked=True, color=[C_OK, C_BAD],
                edgecolor="#fcfcfb", linewidth=0.5, label=["stowed", "not stowed"])
        p95 = np.percentile(x[good], 95)
        ax.axvline(p95, color=C_INK, linewidth=1.0, linestyle="--")
        ax.text(p95, ax.get_ylim()[1], f" p95 = {p95:.3g}", color=C_INK, fontsize=8, va="top", ha="left")
        ax.set_title(label, fontsize=9, color=C_INK, loc="left")
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=C_MUTED, loc="center right")
    fig.suptitle(f"{hand} hand: load distributions over {len(rs)} randomized full flows "
                 f"({int(ok.sum())} stowed, {100 * ok.mean():.1f} %)", fontsize=12, color=C_INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"dist_{hand}.png", dpi=130)
    plt.close(fig)


def plot_sensitivity(rows, hand, out: Path, n_bins: int = 5, top: int = 8):
    """Mean staged score vs each randomized parameter (quantile bins), the most influential first."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [r for r in rows if r["hand"] == hand and r["outcome"] != "error"]
    ok = np.array([r["score"] for r in rs], dtype=float)
    curves = []
    for name in [k for k in rs[0] if k.startswith("p_")]:
        x = np.array([r[name] for r in rs], dtype=float)
        if not np.isfinite(x).all() or np.ptp(x) == 0:
            continue
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
        rate = np.array([ok[idx == b].mean() for b in range(n_bins)])
        mid = np.array([x[idx == b].mean() for b in range(n_bins)])
        curves.append((np.ptp(rate), name[2:], mid, rate))
    curves.sort(key=lambda c: -c[0])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.0), facecolor="#fcfcfb", sharey=True)
    for ax, (spread, name, mid, rate) in zip(axes.ravel(), curves[:top]):
        ax.plot(mid, 100 * rate, color=C_OK, linewidth=2, marker="o", markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.axhline(100 * ok.mean(), color=C_MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(f"{name}   (spread {100 * spread:.0f} pts)", fontsize=9, color=C_INK, loc="left")
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean score x 100", fontsize=8, color=C_MUTED)
    fig.suptitle(f"{hand} hand: mean staged score vs randomized parameter ({n_bins} quantile bins; "
                 f"dotted = overall {100 * ok.mean():.1f})", fontsize=12, color=C_INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"sensitivity_{hand}.png", dpi=130)
    plt.close(fig)
    return [(name, spread) for spread, name, _, _ in curves]


def plot_stages(rows, out: Path):
    """Share of runs that pass each stage (in order), one group of bars per stage, one bar per hand."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hands = list(dict.fromkeys(r["hand"] for r in rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="#fcfcfb")
    width = 0.8 / len(hands)
    for j, (hand, color) in enumerate(zip(hands, (C_OK, C_BAD))):
        rs = [r for r in rows if r["hand"] == hand and r["outcome"] != "error"]
        rate = [100 * np.mean([r[f"s_{name}"] for r in rs]) for name in STAGES]
        x = np.arange(len(STAGES)) + (j - (len(hands) - 1) / 2) * width
        ax.bar(x, rate, width * 0.92, color=color, label=f"{hand} (mean score {np.mean([r['score'] for r in rs]):.2f})")
        for xi, v in zip(x, rate):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=C_INK)
    ax.set_xticks(np.arange(len(STAGES)), STAGES)
    ax.set_ylim(0, 108)
    ax.set_ylabel("runs that got this far [%]", fontsize=8, color=C_MUTED)
    ax.set_title(f"Stage funnel, randomization scale {rows[0]['scale']:g}, {len(rows) // len(hands)} runs per hand",
                 fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=C_MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "stages.png", dpi=130)
    plt.close(fig)


def plot_curve(names: list[str], out: Path):
    """Stage pass rates vs randomization scale, from several sweeps (one panel per hand)."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for n in names for r in read_results(REPO_ROOT / "runs" / "sweeps" / n) if r["outcome"] != "error"]
    hands = list(dict.fromkeys(r["hand"] for r in rows))
    scales = sorted({r["scale"] for r in rows})
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b")
    fig, axes = plt.subplots(1, len(hands), figsize=(6.2 * len(hands), 4.2), facecolor="#fcfcfb", sharey=True)
    for ax, hand in zip(np.atleast_1d(axes), hands):
        for name, color in zip(STAGES, colors):
            y = [100 * np.mean([r[f"s_{name}"] for r in rows if r["hand"] == hand and r["scale"] == sc]) for sc in scales]
            ax.plot(scales, y, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name)
            ax.text(scales[-1], y[-1], f"  {name}", fontsize=8, color=C_INK, va="center")
        ax.set_title(f"{hand} hand", fontsize=10, color=C_INK, loc="left")
        ax.set_xlabel("randomization scale (1 = full ranges of params.yaml)", fontsize=8, color=C_MUTED)
        ax.set_ylim(0, 105)
        ax.set_xlim(0, scales[-1] * 1.18)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("runs that passed the stage [%]", fontsize=8, color=C_MUTED)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "curve.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out / 'curve.png'}")


def plot_all(out: Path):
    rows = read_results(out)
    print(summarize(rows))
    plot_stages(rows, out)
    for hand in dict.fromkeys(r["hand"] for r in rows):
        plot_distributions(rows, hand, out)
        ranking = plot_sensitivity(rows, hand, out)
        print(f"\n{hand}: score spread across bins (x100), top 8: "
              + ", ".join(f"{n} {100 * s:.0f}" for n, s in ranking[:8]))
    print(f"\nfigures in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "plot", "curve"])
    ap.add_argument("--n", type=int, default=1000, help="cases per hand")
    ap.add_argument("--hand", choices=HANDS + ("both",), default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="shrink the randomization ranges (1 = full)")
    ap.add_argument("--i-peak-ratio", type=float, default=2.0, help="current limit / continuous, as in the tests")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--name", default="mech")
    ap.add_argument("--names", nargs="*", default=[], help="curve: sweeps (run with different --scale) to combine")
    a = ap.parse_args()
    out = REPO_ROOT / "runs" / "sweeps" / a.name
    if a.action == "curve":
        return plot_curve(a.names, out)
    if a.action == "run":
        hands = HANDS if a.hand == "both" else (a.hand,)
        sweep(a.n, hands, a.seed, a.scale, a.i_peak_ratio, a.workers, out)
    plot_all(out)


if __name__ == "__main__":
    main()
