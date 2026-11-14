"""Animation and synchronized plots (PHYSICS_SPEC §8, roadmap stage 1 step 8).

Left: the whole rig (stand, link 1, U-fork/wrist, funnel, string, yo-yo coloured
by mode). Middle: a close-up that follows the funnel (wall contact, magnet,
seat). Right: l(t), T(t), i1(t), i2(t) with mode-coloured background, event
lines and a moving cursor.

    python -m viz.animate runs/<run_dir>            # replay a saved run
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib
import numpy as np

from yoyo.funnel import FunnelGeometry
from yoyo.modes import MODE_COLORS, MODE_NAMES, Mode

EVENT_STYLE = {
    "release": ("#4c72b0", "-"),
    "bottom": ("#555555", "--"),
    "snap": ("#dd8452", ":"),
    "slack": ("#dd8452", "-."),
    "mouth": ("#8172b3", ":"),
    "seat_rebound": ("#937860", "--"),
    "stowed": ("#4c72b0", "-"),
    "fail": ("#c44e52", "-"),
}
LABELED_EVENTS = {"release", "bottom", "stowed", "fail", "seat_rebound"}


def _mode_segments(t, mode):
    if len(t) == 0:
        return []
    cuts = np.flatnonzero(np.diff(mode) != 0) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(t) - 1]])
    return [(t[a], t[b], int(mode[a])) for a, b in zip(starts, ends)]


def _funnel_world(polys_local, H, phi2):
    ax, ay = math.cos(phi2), math.sin(phi2)
    nx, ny = -ay, ax
    out = []
    for poly in polys_local:
        x, z = poly[:, 0], poly[:, 1]
        out.append(np.column_stack([H[0] + x * nx + z * ax, H[1] + x * ny + z * ay]))
    return out


class _Scene:
    """Static geometry + artists shared by the full view and the close-up."""

    def __init__(self, ax, p, funnel, show_stand: bool, lw_scale: float = 1.0):
        import matplotlib.patches as mp
        self.ax, self.p, self.funnel = ax, p, funnel
        if show_stand:
            h = p.stand_height
            ax.plot([-0.25, 0.55], [-h, -h], color="0.3", lw=2)
            ax.plot([-0.22, -0.22], [-h, 0.07], color="0.45", lw=5 * lw_scale, solid_capstyle="butt")
            ax.plot([-0.22, 0.03], [0.07, 0.07], color="0.45", lw=5 * lw_scale, solid_capstyle="butt")
            ax.plot([0, 0], [0.07, 0.0], color="0.45", lw=3 * lw_scale)
        self.link1, = ax.plot([], [], color="0.15", lw=4 * lw_scale, solid_capstyle="round", zorder=4)
        self.link2, = ax.plot([], [], color="0.35", lw=3 * lw_scale, solid_capstyle="round", zorder=4)
        self.base = mp.Circle((0, 0), 0.012, color="0.2", zorder=5)
        self.elbow = mp.Circle((0, 0), 0.010, color="#8c8c8c", zorder=5)
        ax.add_patch(self.base)
        ax.add_patch(self.elbow)
        self.funnel_polys = []
        if funnel is not None:
            for _ in funnel.outline_local():
                poly = mp.Polygon(np.zeros((3, 2)), closed=True, fc="#9fb7d6", ec="#33557a",
                                  lw=0.8 * lw_scale, zorder=6)
                ax.add_patch(poly)
                self.funnel_polys.append(poly)
            self._outline = funnel.outline_local()
        self.rope, = ax.plot([], [], color="#7a5a2e", lw=1.0 * lw_scale, zorder=3)
        self.yoyo = mp.Circle((0, 0), p.R, fc=MODE_COLORS[Mode.STOWED], ec="k", lw=0.6, zorder=7, alpha=0.9)
        ax.add_patch(self.yoyo)
        self.spokes = [ax.plot([], [], color="w", lw=1.2 * lw_scale, zorder=8)[0] for _ in range(2)]
        self.hub = mp.Circle((0, 0), p.r0, fc="0.2", zorder=8)
        ax.add_patch(self.hub)

    def update(self, d, k):
        p = self.p
        q1, phi2 = d["q1"][k], d["phi2"][k]
        ex, ey = p.l1 * math.cos(q1), p.l1 * math.sin(q1)
        hx, hy = d["Hx"][k], d["Hy"][k]
        self.link1.set_data([0, ex], [0, ey])
        self.link2.set_data([ex, hx], [ey, hy])
        self.elbow.center = (ex, ey)
        for poly, xy in zip(self.funnel_polys, _funnel_world(self._outline if self.funnel_polys else [], (hx, hy), phi2)):
            poly.set_xy(xy)
        px, py = d["px"][k], d["py"][k]
        mode = int(d["mode"][k])
        self.rope.set_data([hx, px], [hy, py])
        self.yoyo.center = (px, py)
        self.hub.center = (px, py)
        self.yoyo.set_facecolor(MODE_COLORS[Mode(mode)])
        self.yoyo.set_edgecolor("#c44e52" if d["contact"][k] > 0 else "k")
        self.yoyo.set_linewidth(2.0 if d["contact"][k] > 0 else 0.6)
        a = d["spin_angle"][k]
        for i, sp in enumerate(self.spokes):
            ang = a + i * math.pi / 2
            c, s = math.cos(ang), math.sin(ang)
            r = 0.85 * p.R
            sp.set_data([px - r * c, px + r * c], [py - r * s, py + r * s])
        artists = [self.link1, self.link2, self.elbow, self.rope, self.yoyo, self.hub,
                   *self.spokes, *self.funnel_polys]
        return artists


def make_figure(d, events, p, title: str = ""):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    funnel = FunnelGeometry(p) if p.capture_model != "none" else None
    fig = plt.figure(figsize=(15, 8.2))
    gs = GridSpec(5, 3, figure=fig, width_ratios=[1.05, 0.85, 1.55],
                  height_ratios=[0.25, 1, 1, 1, 1], wspace=0.25, hspace=0.35,
                  left=0.04, right=0.98, top=0.93, bottom=0.07)
    ax_full = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[:, 1])
    ax_phase = fig.add_subplot(gs[0, 2])
    axs = [fig.add_subplot(gs[i, 2], sharex=ax_phase) for i in range(1, 5)]
    fig.suptitle(title or "proprio-yoyo simulation", fontsize=13)

    # --- full scene
    ax_full.set_aspect("equal")
    ax_full.set_xlim(-0.33, 0.47)
    ax_full.set_ylim(-p.stand_height - 0.05, 0.15)
    ax_full.set_title("rig")
    ax_full.set_xlabel("x [m]")
    ax_full.grid(alpha=0.25)
    full = _Scene(ax_full, p, funnel, show_stand=True)
    # --- close-up
    ax_zoom.set_aspect("equal")
    ax_zoom.set_title("funnel close-up (follows H)")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    zoom = _Scene(ax_zoom, p, funnel, show_stand=False, lw_scale=2.5)
    mode_txt = ax_zoom.text(0.03, 0.03, "", transform=ax_zoom.transAxes, va="bottom", fontsize=10,
                            family="monospace",
                            bbox=dict(fc="w", ec="0.7", alpha=0.9))
    time_txt = ax_full.text(0.03, 0.98, "", transform=ax_full.transAxes, va="top", fontsize=11,
                            family="monospace")

    # --- time series
    t = d["t"]
    segs = _mode_segments(t, d["mode"])
    for a, b, m in segs:
        ax_phase.axvspan(a, b, color=MODE_COLORS[Mode(m)], lw=0)
        for ax in axs:
            ax.axvspan(a, b, color=MODE_COLORS[Mode(m)], alpha=0.08, lw=0)
    ax_phase.set_yticks([])
    ax_phase.set_title("mode: " + "  ".join(f"{MODE_NAMES[m.value]}" for m in Mode), fontsize=9)
    from matplotlib.patches import Patch
    ax_phase.legend(handles=[Patch(color=MODE_COLORS[m], label=MODE_NAMES[m.value]) for m in Mode],
                    ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax_phase.set_title("")
    plt.setp(ax_phase.get_xticklabels(), visible=False)

    ax_l, ax_T, ax_i1, ax_i2 = axs
    ax_l.plot(t, d["ell"], color="k", lw=1)
    ax_l.axhline(p.L, color="0.5", lw=0.6, ls="--")
    ax_l.axhline(p.ell_mouth, color="#8172b3", lw=0.6, ls=":")
    ax_l.set_ylabel(r"$\ell$ [m]")
    ax_l.invert_yaxis()
    ax_T.plot(t, d["T"], color="#55a868", lw=0.9, label="T (string)")
    fb = np.hypot(d["F_bx"], d["F_by"])
    ax_T.plot(t, fb, color="#c44e52", lw=0.9, label="F_b (bottom pulse)")
    ax_T.set_ylabel("force [N]")
    ax_T.legend(fontsize=7, loc="upper right")
    ax_i1.plot(t, d["i1_meas"], color="#bbbbbb", lw=0.5, label="measured")
    ax_i1.plot(t, d["i1"], color="#4c72b0", lw=0.9, label="true")
    ax_i1.axhline(p.i_max1, color="#c44e52", lw=0.6, ls="--")
    ax_i1.axhline(-p.i_max1, color="#c44e52", lw=0.6, ls="--")
    ax_i1.set_ylabel("i1 [A]")
    ax_i1.legend(fontsize=7, loc="upper right")
    ax_i2.plot(t, d["i2_meas"], color="#bbbbbb", lw=0.5)
    ax_i2.plot(t, d["i2"], color="#4c72b0", lw=0.9)
    ax_i2.axhline(p.i_max2, color="#c44e52", lw=0.6, ls="--")
    ax_i2.axhline(-p.i_max2, color="#c44e52", lw=0.6, ls="--")
    ax_i2.set_ylabel("i2 [A]")
    ax_i2.set_xlabel("t [s]")
    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ev in events:
        col, ls = EVENT_STYLE.get(ev["type"], ("0.5", ":"))
        for ax in axs:
            ax.axvline(ev["t"], color=col, ls=ls, lw=0.7, alpha=0.8)
        if ev["type"] in LABELED_EVENTS:
            ax_l.annotate(ev["type"], (ev["t"], 0.02), xycoords=("data", "axes fraction"),
                          rotation=90, fontsize=7, color=col, va="bottom", ha="right")
    for ax in axs + [ax_phase]:
        ax.set_xlim(t[0], t[-1])
        ax.grid(alpha=0.25)
    cursors = [ax.axvline(t[0], color="k", lw=1.0) for ax in axs + [ax_phase]]

    def update(k):
        arts = full.update(d, k) + zoom.update(d, k)
        hx, hy, phi2 = d["Hx"][k], d["Hy"][k], d["phi2"][k]
        # close-up window: centred below H along the funnel axis
        cx, cy = hx + 0.045 * math.cos(phi2), hy + 0.045 * math.sin(phi2)
        ax_zoom.set_xlim(cx - 0.085, cx + 0.085)
        ax_zoom.set_ylim(cy - 0.12, cy + 0.08)
        m = int(d["mode"][k])
        mode_txt.set_text(
            f"{MODE_NAMES[m]:7s} inside={'Y' if d['inside'][k] > 0 else 'n'}\n"
            f"l   = {d['ell'][k]:6.3f} m\nl'  = {d['elld'][k]:+6.2f} m/s\n"
            f"ω   = {d['omega'][k]:+6.0f} rad/s\nT   = {d['T'][k]:6.2f} N\n"
            f"Fmag= {d['F_mag'][k]:6.2f} N")
        mode_txt.set_color(MODE_COLORS[Mode(m)])
        time_txt.set_text(f"t = {d['t'][k]:5.2f} s")
        for c in cursors:
            c.set_xdata([d["t"][k], d["t"][k]])
        return arts + [mode_txt, time_txt] + cursors

    return fig, update


def animate(d, events, p, out: str | Path | None = None, fps: int = 25, speed: float = 1.0,
            t_start: float | None = None, t_end: float | None = None, title: str = "",
            dpi: int = 80, show: bool = False):
    """Render an animation of a log (dict of arrays from SimLog.arrays())."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    t = d["t"]
    t0 = t[0] if t_start is None else t_start
    t1 = t[-1] if t_end is None else t_end
    frame_t = np.arange(t0, t1, speed / fps)
    idx = np.clip(np.searchsorted(t, frame_t), 0, len(t) - 1)
    fig, update = make_figure(d, events, p, title)
    anim = animation.FuncAnimation(fig, lambda i: update(idx[i]), frames=len(idx),
                                   interval=1000 / fps, blit=False)
    if out is not None:
        out = Path(out)
        if out.suffix == ".mp4" and shutil.which("ffmpeg"):
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        else:
            if out.suffix == ".mp4":
                out = out.with_suffix(".gif")
                print("ffmpeg not found; writing a GIF instead")
            writer = animation.PillowWriter(fps=fps)
        anim.save(out, writer=writer, dpi=dpi)
        print(f"saved {out}")
    if show:
        plt.show()
    plt.close(fig)
    return out


def snapshot(d, events, p, t_snap: float, out: str | Path, title: str = "", dpi: int = 110):
    """Static figure: full time series with the scene drawn at t_snap."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, update = make_figure(d, events, p, title)
    k = int(np.clip(np.searchsorted(d["t"], t_snap), 0, len(d["t"]) - 1))
    update(k)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"saved {out}")
    return out


def main():
    from sim.log import load_run
    from sim.params import Params

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="output .gif/.mp4 (default <run_dir>/anim.gif)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (0.25 = slow motion)")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--show", action="store_true", help="open an interactive window instead")
    a = ap.parse_args()
    d, events, pdict = load_run(a.run_dir)
    p = Params(**pdict)
    out = None if a.show else (a.out or Path(a.run_dir) / "anim.gif")
    animate(d, events, p, out=out, fps=a.fps, speed=a.speed, t_start=a.t0, t_end=a.t1,
            title=Path(a.run_dir).name, show=a.show)


if __name__ == "__main__":
    main()
