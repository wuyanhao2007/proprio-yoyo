"""Build runs/index.html: one card per run directory, newest first.

Each card shows the outcome, event counts, the parameters that differ from
config/params.yaml, summary.png, and anim.gif when present.

    python -m scripts.gallery            # writes runs/index.html
    python -m scripts.gallery --open     # ... and opens it in the default browser
"""
from __future__ import annotations

import argparse
import html
import json
import webbrowser
from collections import Counter
from pathlib import Path

from sim.params import REPO_ROOT, Params

RUNS = REPO_ROOT / "runs"
MAIN_EVENTS = ("unseat", "release", "bottom", "mouth", "seat_rebound", "in_cup", "flip_start",
               "drop", "stowed", "fail")

CSS = """
:root { --bg:#f7f7f5; --card:#fff; --fg:#222; --mute:#777; --ok:#2e7d32; --bad:#c62828; --line:#e3e3e0; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161616; --card:#222; --fg:#eee; --mute:#999; --ok:#81c784; --bad:#ef9a9a; --line:#333; } }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
       font:14px/1.45 -apple-system, system-ui, sans-serif; }
h1 { font-size:20px; margin:0 0 4px; } .sub { color:var(--mute); margin-bottom:20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:16px; margin:0 auto 18px; max-width:1200px; }
.head { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:baseline; }
.name { font-weight:600; font-size:16px; } .tag { color:var(--mute); }
.ok { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.cols { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:12px; margin-top:12px; }
img { width:100%; border-radius:6px; border:1px solid var(--line); background:#fff; }
table { border-collapse:collapse; font-size:12px; } td { padding:1px 10px 1px 0; vertical-align:top; }
code { font-size:12px; } details { margin-top:8px; } summary { cursor:pointer; color:var(--mute); }
"""


def _load(p: Path, default):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _outcome(events):
    types = [e["type"] for e in events]
    if "fail" in types:
        reason = next(e.get("reason", "") for e in events if e["type"] == "fail")
        return f"FAIL ({reason})", "bad"
    if types and types[-1] == "stowed":
        return "caught", "ok"
    if "release" in types:
        return "thrown, not caught", ""
    return "—", ""


def _param_diff(params: dict, nominal: dict) -> list[tuple[str, str, str]]:
    out = []
    for k, v in params.items():
        nv = nominal.get(k)
        if nv is None:
            continue
        if isinstance(v, float) and isinstance(nv, float):
            if abs(v - nv) > 1e-12 * max(1.0, abs(nv)):
                out.append((k, f"{nv:g}", f"{v:g}"))
        elif v != nv:
            out.append((k, str(nv), str(v)))
    return out


def card(run: Path, nominal: dict) -> str:
    events = _load(run / "events.json", [])
    params = _load(run / "params.json", {})
    meta = _load(run / "meta.json", {})
    outcome, cls = _outcome(events)
    counts = Counter(e["type"] for e in events)
    t_end = max((e["t"] for e in events), default=0.0)
    head = (f'<div class="head"><span class="name">{html.escape(run.name)}</span>'
            f'<span class="tag">{html.escape(str(meta.get("scenario", "")))}</span>'
            f'<span class="tag">{html.escape(str(params.get("catch_mode", "magnet")))} hand</span>'
            f'<span class="{cls}">{outcome}</span>'
            f'<span class="tag">' + ", ".join(f"{k}×{counts[k]}" for k in MAIN_EVENTS if counts[k])
            + "</span></div>")
    diff = _param_diff(params, nominal)
    diff_html = ("<table>" + "".join(f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(a)} → "
                                     f"<b>{html.escape(b)}</b></td></tr>" for k, a, b in diff) + "</table>"
                 if diff else '<span class="tag">default parameters</span>')
    args = {k: v for k, v in meta.items() if v not in (None, False, [], "") and k not in ("out",)}
    imgs = []
    for fn in ("summary.png", "anim.gif"):
        if (run / fn).exists():
            rel = f"{run.name}/{fn}"
            imgs.append(f'<a href="{rel}"><img loading="lazy" src="{rel}" alt="{fn}"></a>')
    ev_rows = "".join(
        f"<tr><td>{e['t']:.4f}</td><td>{html.escape(e['type'])}</td><td><code>"
        + html.escape(", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in e.items() if k not in ("t", "type")))[:160]
        + "</code></td></tr>"
        for e in events if e["type"] in MAIN_EVENTS)
    return (f'<div class="card">{head}'
            f'<div class="cols">{"".join(imgs) or "<span class=tag>no images</span>"}</div>'
            f"<details open><summary>parameters vs config/params.yaml</summary>{diff_html}</details>"
            f"<details><summary>command-line options</summary><code>{html.escape(json.dumps(args))}</code></details>"
            f"<details><summary>events (t ≤ {t_end:.2f} s)</summary><table>{ev_rows}</table></details>"
            "</div>")


def build(out: Path = RUNS / "index.html") -> Path:
    nominal = Params.load().as_dict()
    runs = sorted((d for d in RUNS.iterdir() if (d / "events.json").exists()),
                  key=lambda d: d.stat().st_mtime, reverse=True) if RUNS.exists() else []
    extra = "".join(f'<div class="card"><div class="name">{f.name}</div><div class="cols">'
                    f'<a href="{f.name}"><img src="{f.name}"></a></div></div>'
                    for f in sorted(RUNS.glob("*.png")))
    body = "".join(card(r, nominal) for r in runs)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>proprio-yoyo runs</title><style>{CSS}</style></head><body>"
           f"<h1>proprio-yoyo runs</h1><div class='sub'>{len(runs)} runs in runs/, newest first. "
           f"Rebuild with <code>python -m scripts.gallery</code>.</div>{body}{extra}</body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    out = build()
    print(f"wrote {out}")
    if a.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
