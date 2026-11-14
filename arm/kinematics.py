"""Planar kinematics of the 2-DOF arm (PHYSICS_SPEC §5.1).

Generalized coordinates for dynamics are q = (q1, phi2):
  q1   = link-1 angle from horizontal (joint 1, belt-driven),
  phi2 = absolute angle of link 2 (wrist / funnel); q2 = phi2 - q1 is the
         relative wrist angle read by the direct-drive GM3506.
Joint-1 axis is the world origin; x right, y up.
"""
from __future__ import annotations

import math


def elbow(l1: float, q1: float) -> tuple[float, float]:
    """Wrist-axis position E = l1 (cos q1, sin q1)."""
    return l1 * math.cos(q1), l1 * math.sin(q1)


def hook(l1: float, l2: float, q1: float, phi2: float) -> tuple[float, float]:
    """String exit H = E + l2 (cos phi2, sin phi2)."""
    return (l1 * math.cos(q1) + l2 * math.cos(phi2),
            l1 * math.sin(q1) + l2 * math.sin(phi2))


def hook_jacobian(l1: float, l2: float, q1: float, phi2: float):
    """dH/d(q1, phi2) as ((J00, J01), (J10, J11)). §5.1"""
    s1, c1 = math.sin(q1), math.cos(q1)
    s2, c2 = math.sin(phi2), math.cos(phi2)
    return ((-l1 * s1, -l2 * s2),
            (l1 * c1, l2 * c2))


def point_on_link2(l1, q1, phi2, s, t=0.0):
    """World position of a point with link-2 local coords (s along a, t along n),
    where a = (cos phi2, sin phi2) and n = (-sin phi2, cos phi2)."""
    ex, ey = elbow(l1, q1)
    c2, s2 = math.cos(phi2), math.sin(phi2)
    return ex + s * c2 - t * s2, ey + s * s2 + t * c2


def world_point_jacobian(l1, q1, phi2, px, py):
    """Jacobian dP/d(q1, phi2) of a world point P rigidly attached to link 2."""
    ex, ey = elbow(l1, q1)
    rx, ry = px - ex, py - ey
    return ((-l1 * math.sin(q1), -ry),
            (l1 * math.cos(q1), rx))


def world_point_bias_accel(l1, q1, phi2, q1d, phi2d, px, py):
    """Jdot * qdot for a point rigidly attached to link 2 (centripetal terms)."""
    ex, ey = elbow(l1, q1)
    rx, ry = px - ex, py - ey
    return (-l1 * math.cos(q1) * q1d**2 - rx * phi2d**2,
            -l1 * math.sin(q1) * q1d**2 - ry * phi2d**2)


def world_point_velocity(l1, q1, phi2, q1d, phi2d, px, py):
    (j00, j01), (j10, j11) = world_point_jacobian(l1, q1, phi2, px, py)
    return j00 * q1d + j01 * phi2d, j10 * q1d + j11 * phi2d
