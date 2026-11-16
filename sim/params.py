"""Parameter set, derived quantities and domain randomization (PHYSICS_SPEC §1).

All numeric values live in ``config/params.yaml``; this module only defines the
schema. Derived quantities are functions/properties, never stored constants.
Placeholders that still need hardware identification are marked TODO(sysid)
in the YAML file.

Hand variants (§4.4) are overlays on params.yaml: ``Params.load(hand="flip")``
reads ``config/hand_flip.yaml`` on top of the defaults. An overlay may also carry
a ``randomization`` block (entries replace / ``null`` removes) and non-physics
controller blocks (``controller``, ``baseline``; see ``load_controller``).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "config" / "params.yaml"
CATCH_MODES = ("magnet", "flip")
CONTROLLER_BLOCKS = ("controller", "baseline")   # hand-overlay blocks that are not physics


def hand_path(hand: str | Path) -> Path:
    """``"magnet"`` / ``"flip"`` -> config/hand_<name>.yaml; anything else is a path."""
    if isinstance(hand, str) and hand in CATCH_MODES:
        return REPO_ROOT / "config" / f"hand_{hand}.yaml"
    return Path(hand)


@dataclass(frozen=True)
class Params:
    # --- yo-yo and string (§1) ---
    m: float
    R: float
    I_coef: float
    r0: float
    L: float
    d_s: float
    r_full: float
    c_loss: float
    c_l: float
    e_taut: float
    v_snap_min: float
    # --- arm (§1, §5) ---
    l1: float
    l2: float
    m1: float
    m2: float
    mc: float
    lc_offset: float
    Ic_radius: float
    m_e: float
    q1_min: float
    q1_max: float
    # --- drive (§1, §6) ---
    N1: float
    eta1: float
    Jr1: float
    Jr2: float
    Kt1: float  # TODO(sysid)
    Kt2: float  # TODO(sysid)
    R1: float
    R2: float
    tau_max1: float
    tau_max2: float
    i_peak_ratio: float
    V_bus: float
    V_lim2: float
    use_back_emf: bool
    b1: float  # TODO(sysid)
    b2: float  # TODO(sysid)
    tc1: float  # TODO(sysid)
    tc2: float  # TODO(sysid)
    tau_i: float
    cogging_amp: float
    # --- firmware PD (§6) ---
    Kp1: float
    Kd1: float
    Kp2: float
    Kd2: float
    cmd_delay: float
    # --- funnel (§1, §4) ---
    capture_model: str
    d_cup: float
    d_apex: float
    seat_width: float
    seat_thickness: float
    seat_depth: float
    mouth_width: float
    mouth_thickness: float
    flare_exponent: float
    wall_thickness: float
    alpha_cap: float
    v_cap_max: float
    e_cup: float
    F_mag0: float
    d_mag: float
    k_wall: float
    e_wall: float
    mu_wall: float
    # --- catch mode (§4.4) ---
    catch_mode: str
    s_max: float
    stow_tilt_max: float
    t_stow_hold: float
    spin_coupling: bool
    # --- sensors (§7) ---
    enc_counts: int
    enc_delay: float
    vel_cutoff: float
    i_noise: float
    i_bias_max: float
    i_delay: float
    # --- simulation (§2, §8) ---
    g: float
    dt: float
    substeps: int
    planar: bool
    variable_radius: bool
    stand_height: float

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None, hand: str | Path | None = None,
             **overrides: Any) -> "Params":
        """Load from YAML (default ``config/params.yaml``), then the hand overlay
        (``"magnet"``, ``"flip"`` or a path), then keyword overrides."""
        values, _ = _read_yaml(path)
        if hand is not None:
            hv, _ = _read_yaml(hand_path(hand))
            unknown = set(hv) - set(values)
            if unknown:
                raise ValueError(f"unknown parameters in {hand_path(hand)}: {sorted(unknown)}")
            values.update(hv)
        values.update(overrides)
        out = cls(**_coerce(cls, values))
        if out.catch_mode not in CATCH_MODES:
            raise ValueError(f"catch_mode must be one of {CATCH_MODES}, got {out.catch_mode!r}")
        return out

    def replace(self, **changes: Any) -> "Params":
        return dataclasses.replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    # ------------------------------------------------------------ derived (§1)
    @property
    def I(self) -> float:
        """Spin inertia of the yo-yo about its axle."""
        return self.I_coef * self.m * self.R**2

    def kappa(self, r: float | None = None) -> float:
        """kappa(r) = m r^2 / (I + m r^2): falling-acceleration ratio."""
        r = self.r0 if r is None else r
        mr2 = self.m * r * r
        return mr2 / (self.I + mr2)

    @property
    def zeta(self) -> float:
        """Zlajpah bottom speed ratio (I - m r0^2) / (I + m r0^2)."""
        mr2 = self.m * self.r0**2
        return (self.I - mr2) / (self.I + mr2)

    @property
    def zeta_e(self) -> float:
        return self.c_loss * self.zeta

    @property
    def ell_min(self) -> float:
        """H -> yo-yo centre when stowed."""
        return self.R + self.d_apex

    @property
    def ell_mouth(self) -> float:
        """H -> yo-yo centre when the yo-yo top reaches the mouth plane."""
        return self.R + self.d_cup

    @property
    def w_eff(self) -> float:
        """Effective winding-groove width from the fully-wound radius (§3.4)."""
        return self.L * self.d_s**2 / (4.0 * (self.r_full**2 - self.r0**2))

    @property
    def lc(self) -> float:
        return self.l2 + self.lc_offset

    @property
    def Ic(self) -> float:
        return self.mc * self.Ic_radius**2

    def radius(self, ell: float) -> tuple[float, float]:
        """Winding radius r(l) and dr/dl (§3.4). Constant r0 unless variable_radius."""
        if not self.variable_radius:
            return self.r0, 0.0
        ell_c = min(max(ell, 0.0), self.L)
        r2 = self.r0**2 + (self.L - ell_c) * self.d_s**2 / (4.0 * self.w_eff)
        r = math.sqrt(r2)
        return r, -self.d_s**2 / (8.0 * self.w_eff * r)

    @property
    def i_max1(self) -> float:
        return self.i_peak_ratio * self.tau_max1 / self.Kt1

    @property
    def i_max2(self) -> float:
        return self.i_peak_ratio * self.tau_max2 / self.Kt2


# ---------------------------------------------------------------- YAML helpers
def _read_yaml(path: str | Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path) if path is not None else DEFAULT_YAML
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    rand = raw.pop("randomization", {}) or {}
    for block in CONTROLLER_BLOCKS:
        raw.pop(block, None)
    values: dict[str, Any] = {}
    for section, entries in raw.items():
        if not isinstance(entries, Mapping):
            raise ValueError(f"section '{section}' in {path} must be a mapping")
        for key, val in entries.items():
            if key in values:
                raise ValueError(f"duplicate parameter '{key}' in {path}")
            values[key] = val
    return values, rand


def _coerce(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    missing = set(names) - set(values)
    if missing:
        raise ValueError(f"missing parameters: {sorted(missing)}")
    out = {}
    for name, val in values.items():
        typ = names[name].type
        if typ in ("float", float):
            out[name] = float(val)
        elif typ in ("int", int):
            out[name] = int(val)
        elif typ in ("bool", bool):
            out[name] = bool(val)
        else:
            out[name] = val
    return out


def load_randomization(path: str | Path | None = None,
                       hand: str | Path | None = None) -> dict[str, Any]:
    """Randomization ranges of params.yaml, updated by the hand overlay (if any):
    an overlay entry replaces the default one, ``null`` removes it."""
    rand = dict(_read_yaml(path)[1])
    if hand is not None and hand_path(hand).exists():
        for k, v in _read_yaml(hand_path(hand))[1].items():
            if v is None:
                rand.pop(k, None)
            else:
                rand[k] = v
    return rand


def load_controller(hand: str | Path, block: str = "controller") -> dict[str, Any]:
    """A non-physics block of a hand overlay ({} if absent): ``controller`` holds the
    flip-stow settings (control/stow_flip.py), ``baseline`` overrides
    ``BaselineConfig`` defaults (control/baseline.py)."""
    if block not in CONTROLLER_BLOCKS:
        raise ValueError(f"unknown controller block {block!r}")
    with open(hand_path(hand)) as f:
        raw = yaml.safe_load(f) or {}
    return dict(raw.get(block) or {})


def sample_params(
    nominal: Params,
    rng: np.random.Generator,
    ranges: Mapping[str, Any] | None = None,
    scale: float = 1.0,
) -> Params:
    """Draw one randomized parameter set (uniform). ``scale`` shrinks the ranges
    towards the nominal values (curriculum: start narrow, widen later)."""
    ranges = load_randomization(hand=nominal.catch_mode) if ranges is None else ranges
    changes = {}
    for name, spec in ranges.items():
        nom = getattr(nominal, name)
        if "rel" in spec:
            lo, hi = nom * (1 - spec["rel"]), nom * (1 + spec["rel"])
        else:
            lo, hi = spec["range"]
        lo = nom + scale * (lo - nom)
        hi = nom + scale * (hi - nom)
        changes[name] = float(rng.uniform(min(lo, hi), max(lo, hi)))
    return nominal.replace(**changes)
