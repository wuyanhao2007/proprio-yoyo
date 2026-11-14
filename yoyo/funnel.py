"""Funnel geometry, magnet and wall contact (PHYSICS_SPEC §4, §4.3).

Funnel frame (local): origin at the string exit H, axial coordinate z along the
opening direction a = (cos phi2, sin phi2) (z = 0 at H, z = d_cup at the mouth
plane), lateral coordinate x along n = (-sin phi2, cos phi2). Only the in-plane
cross-section is modelled (the plane of the arm = the yo-yo's spin plane).

Inner wall half-width w(z), shape parameters (§4.3):

    seat   0 <= z <= seat_depth :  w = w_s = seat_width / 2          (loose straight seat)
    flare  seat_depth < z <= d_cup:  w = w_s + (w_m - w_s) * xi**n,
                                     xi = (z - seat_depth) / (d_cup - seat_depth)

With n > 1 the flare leaves the seat wall tangentially (no corner for the rim
to catch on) and steepens towards the mouth ("trumpet"); n = 1 is a cone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.params import Params

_N_FLARE = 40


@dataclass
class Contact:
    pen: float          # penetration depth (m, > 0)
    nx: float           # unit normal from the wall towards the disc centre (local)
    nz: float
    cx: float           # contact point on the wall (local)
    cz: float


class FunnelGeometry:
    def __init__(self, p: Params, check: bool = True):
        self.p = p
        self.ws = p.seat_width / 2
        self.wm = p.mouth_width / 2
        self.zs = p.seat_depth
        self.zm = p.d_cup
        if not (0 < self.zs < self.zm):
            raise ValueError("need 0 < seat_depth < d_cup")
        if check and self.ws < p.R:
            raise ValueError("seat narrower than the yo-yo")
        z = np.concatenate([[0.0], np.linspace(self.zs, self.zm, _N_FLARE + 1)])
        w = np.array([self.half_width(zz) for zz in z])
        self.z = z
        self.w = w
        # right-wall segments A -> B in (x, z); the left wall is the mirror image
        self._ax, self._az = w[:-1], z[:-1]
        self._bx, self._bz = w[1:], z[1:]
        dx, dz = self._bx - self._ax, self._bz - self._az
        self._dx, self._dz = dx, dz
        self._len2 = dx * dx + dz * dz

    # ----------------------------------------------------------------- shape
    def half_width(self, z: float) -> float:
        if z <= self.zs:
            return self.ws
        xi = min((z - self.zs) / (self.zm - self.zs), 1.0)
        return self.ws + (self.wm - self.ws) * xi ** self.p.flare_exponent

    def wall_slope(self, z: float) -> float:
        """dw/dz of the inner wall (0 in the seat)."""
        if z <= self.zs or z > self.zm:
            return 0.0
        n = self.p.flare_exponent
        xi = (z - self.zs) / (self.zm - self.zs)
        return (self.wm - self.ws) * n * xi ** (n - 1) / (self.zm - self.zs)

    @property
    def clearance(self) -> float:
        """Radial clearance between the seated yo-yo and the seat wall."""
        return self.ws - self.p.R

    # --------------------------------------------------------------- regions
    def near(self, x: float, z: float) -> bool:
        """True if a disc centred at (x, z) can touch the funnel at all."""
        R = self.p.R
        return (z - R < self.zm) and (abs(x) < self.wm + R) and (z + R > 0.0)

    def outside_hit(self, x: float, z: float) -> bool:
        """Disc top is above the mouth plane but the centre is outside the inner
        wall: the yo-yo hits the rim or the outside of the funnel -> FAIL."""
        R = self.p.R
        if z - R >= self.zm:
            return False
        return abs(x) > self.half_width(min(max(z, 0.0), self.zm))

    # --------------------------------------------------------------- contact
    def contacts(self, x: float, z: float) -> list[Contact]:
        """Wall contacts of a disc (radius R) centred at local (x, z)."""
        if not self.near(x, z):
            return []
        R = self.p.R
        out = []
        for side in (1.0, -1.0):
            xs = side * x  # mirror the left wall onto the right one
            t = ((xs - self._ax) * self._dx + (z - self._az) * self._dz) / self._len2
            t = np.clip(t, 0.0, 1.0)
            px = self._ax + t * self._dx
            pz = self._az + t * self._dz
            ddx, ddz = xs - px, z - pz
            d2 = ddx * ddx + ddz * ddz
            k = int(np.argmin(d2))
            d = math.sqrt(d2[k])
            if d >= R or d < 1e-12:
                continue
            nx, nz = ddx[k] / d, ddz[k] / d
            if nx > 0:  # centre beyond the wall: no valid inward normal
                continue
            out.append(Contact(pen=R - d, nx=side * nx, nz=nz, cx=side * px[k], cz=pz[k]))
        return out

    # ------------------------------------------------------------- drawing
    def outline_local(self):
        """Closed polygons (x, z) of the funnel shell cross-section for drawing:
        right wall, left wall, top plate."""
        t = self.p.wall_thickness
        z, w = self.z, self.w
        # outward normal of the right inner wall (pointing away from the axis)
        dz = np.gradient(z)
        dw = np.gradient(w)
        nrm = np.hypot(dz, dw)
        ox, oz = dz / nrm, -dw / nrm
        outer_x, outer_z = w + t * ox, z + t * oz
        right = np.column_stack([np.concatenate([w, outer_x[::-1]]),
                                 np.concatenate([z, outer_z[::-1]])])
        left = right.copy()
        left[:, 0] *= -1
        top = np.array([[-self.ws - t, 0.0], [self.ws + t, 0.0],
                        [self.ws + t, -t], [-self.ws - t, -t]])
        return [right, left, top]


def magnet_force(p: Params, ell: float) -> float:
    """§4.2 magnet pull towards H, applied only while the yo-yo is inside."""
    d = max(ell - p.ell_min, 0.0)
    return p.F_mag0 / (1.0 + d / p.d_mag) ** 2


def wall_damping(p: Params, mass: float) -> float:
    """Penalty damping that gives restitution e_wall for a linear spring-damper."""
    e = min(max(p.e_wall, 1e-4), 0.9999)
    ln = math.log(e)
    zeta = -ln / math.sqrt(math.pi**2 + ln * ln)
    return 2.0 * zeta * math.sqrt(p.k_wall * mass)
