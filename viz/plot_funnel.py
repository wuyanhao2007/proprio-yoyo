"""Draw the funnel profile (PHYSICS_SPEC §4.3) for design review / CAD.

    python -m viz.plot_funnel                      # writes runs/funnel_profile.png
    python -m viz.plot_funnel --set flare_exponent=1 mouth_width=0.09
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

from sim.params import REPO_ROOT, Params
from yoyo.funnel import FunnelGeometry


def _section(ax, f: FunnelGeometry, p: Params, title: str, half_seat: float, half_mouth: float,
             yoyo_half: float, round_yoyo: bool):
    q = p.replace(seat_width=2 * half_seat, mouth_width=2 * half_mouth)
    g = FunnelGeometry(q, check=False) if (half_seat != f.ws or half_mouth != f.wm) else f
    for poly in g.outline_local():
        ax.add_patch(mp.Polygon(np.column_stack([poly[:, 0], -poly[:, 1]]) * 1e3, closed=True,
                                fc="#9fb7d6", ec="#33557a", lw=1))
    zc = p.ell_min
    if round_yoyo:
        ax.add_patch(mp.Circle((0, -zc * 1e3), p.R * 1e3, fc="#55a868", alpha=0.35, ec="k"))
        ax.add_patch(mp.Circle((0, -zc * 1e3), p.r0 * 1e3, fc="0.2"))
    else:
        ax.add_patch(mp.Rectangle((-yoyo_half * 1e3, -(zc + p.R) * 1e3), 2 * yoyo_half * 1e3,
                                  2 * p.R * 1e3, fc="#55a868", alpha=0.35, ec="k"))
    ax.plot([0, 0], [0, -(zc + p.R + 0.03) * 1e3], color="#7a5a2e", lw=1)
    ax.axhline(-p.seat_depth * 1e3, color="0.6", ls=":", lw=0.8)
    ax.axhline(-p.d_cup * 1e3, color="0.6", ls=":", lw=0.8)
    ax.annotate("", (-g.ws * 1e3, -p.seat_depth * 0.5e3), (g.ws * 1e3, -p.seat_depth * 0.5e3),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(0, -p.seat_depth * 0.5e3 + 1.5, f"seat {2 * g.ws * 1e3:.1f} mm "
            f"(clearance {(g.ws - yoyo_half) * 1e3:.1f}/side)", ha="center", fontsize=8)
    ax.annotate("", (-g.wm * 1e3, -p.d_cup * 1e3 - 3), (g.wm * 1e3, -p.d_cup * 1e3 - 3),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(0, -p.d_cup * 1e3 - 8, f"mouth {2 * g.wm * 1e3:.0f} mm", ha="center", fontsize=8)
    ax.text(g.wm * 1e3 + 3, -p.seat_depth * 1e3, f"seat depth {p.seat_depth * 1e3:.0f}", fontsize=7, va="center")
    ax.text(g.wm * 1e3 + 3, -p.d_cup * 1e3, f"total {p.d_cup * 1e3:.0f}", fontsize=7, va="center")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("lateral [mm]")
    ax.set_ylabel("depth below H [mm]")
    ax.set_xlim(-60, 75)
    ax.set_ylim(-(p.ell_min + p.R + 0.035) * 1e3, 8)
    ax.grid(alpha=0.25)


def plot(p: Params, out: Path):
    f = FunnelGeometry(p)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    _section(axs[0], f, p, "in-plane section (arm plane, yo-yo diameter)", f.ws, f.wm, p.R, True)
    _section(axs[1], f, p, "out-of-plane section (yo-yo thickness) — same flare law",
             p.seat_thickness / 2, p.mouth_thickness / 2, 0.0171, False)
    z = np.linspace(0, p.d_cup, 300)
    for n, ls in [(1.0, ":"), (2.0, "-"), (3.0, "--")]:
        g = FunnelGeometry(p.replace(flare_exponent=n))
        ang = [math.degrees(math.atan(g.wall_slope(zz))) for zz in z]
        axs[2].plot(z * 1e3, ang, ls=ls, label=f"n = {n:g}" + ("  (current)" if n == p.flare_exponent else ""))
    axs[2].axvline(p.seat_depth * 1e3, color="0.6", ls=":", lw=0.8)
    axs[2].set_xlabel("depth below H [mm]")
    axs[2].set_ylabel("wall angle from the axis [deg]")
    axs[2].set_title("flare wall angle: cone (n=1) has a corner at the seat", fontsize=10)
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "funnel_profile.png"))
    a = ap.parse_args()
    over = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    plot(Params.load(**over), Path(a.out))


if __name__ == "__main__":
    main()
