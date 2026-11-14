"""Hybrid-system modes (PHYSICS_SPEC §2)."""
from enum import IntEnum


class Mode(IntEnum):
    STOWED = 0   # yo-yo held in the funnel seat by the magnet (rigid with link 2)
    FREE = 1     # string slack: ballistic flight
    TAUT = 2     # string taut: rolling constraint l_dot = sigma r omega
    FAIL = 3     # hit the funnel rim / outside, joint limit, ...


MODE_NAMES = {m.value: m.name for m in Mode}
MODE_COLORS = {
    Mode.STOWED: "#4c72b0",
    Mode.FREE: "#dd8452",
    Mode.TAUT: "#55a868",
    Mode.FAIL: "#c44e52",
}
