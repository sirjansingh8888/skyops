"""Project image pixels onto the ground for a camera carried by a drone.

Model: pinhole camera with horizontal field of view `hfov_deg`, mounted on a body that is at height
`alt_m` above flat ground, heading `yaw_deg` (clockwise from north), with the camera tilted
`tilt_deg` below the horizon (90 = straight down, nadir). Body roll/pitch can be added. Rays that do
not hit the ground (above the horizon) return None. Distances are metres in a local east/north frame
around the drone's GPS fix; `ground_to_latlon` converts them back to coordinates.

The AU-AIR camera geometry is not published precisely, so `tilt_deg` and `hfov_deg` are tunable from
the UI and the defaults are conservative estimates for a Parrot Bebop 2 looking forward-down.
"""
from __future__ import annotations

import math

import numpy as np

from skyops import config

DEFAULT_TILT_DEG = 45.0
DEFAULT_HFOV_DEG = 69.0
MAX_RANGE_M = 400.0


def _rotation(yaw_deg: float, tilt_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Matrix mapping camera-frame rays (x right, y down, z forward) to world (east, north, up)."""
    # camera -> body (forward, right, down)
    c2b = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    t = math.radians(tilt_deg)
    tilt = np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])  # pitch camera down
    p = math.radians(pitch_deg)  # body nose-up positive
    pitch = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    r = math.radians(roll_deg)   # right wing down positive
    roll = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    y = math.radians(yaw_deg)
    # body (forward, right, down) -> world (east, north, up)
    b2w = np.array([[math.sin(y), math.cos(y), 0], [math.cos(y), -math.sin(y), 0], [0, 0, -1]])
    return b2w @ roll @ pitch @ tilt @ c2b


class GroundProjector:
    def __init__(self, width: int, height: int, alt_m: float, yaw_deg: float, tilt_deg: float = DEFAULT_TILT_DEG,
                 hfov_deg: float = DEFAULT_HFOV_DEG, pitch_deg: float = 0.0, roll_deg: float = 0.0,
                 max_range_m: float = MAX_RANGE_M):
        self.w, self.h = width, height
        self.alt = max(float(alt_m), 0.5)
        self.f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.R = _rotation(yaw_deg, tilt_deg, pitch_deg, roll_deg)
        self.max_range = max_range_m

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float, float] | None:
        """(east_m, north_m, range_m) of the ground point seen at pixel (u, v), or None above the horizon."""
        ray = np.array([(u - self.w / 2) / self.f, (v - self.h / 2) / self.f, 1.0])
        d = self.R @ ray
        if d[2] >= -1e-6:
            return None
        s = self.alt / -d[2]
        e, n = float(d[0] * s), float(d[1] * s)
        rng = math.hypot(e, n)
        if rng > self.max_range:
            return None
        return e, n, rng

    def footprint(self, steps: int = 8) -> list[tuple[float, float]]:
        """Ground polygon (east, north) of the visible area, walking the image border; far edge clipped."""
        pts = []
        border = [(x, self.h - 1) for x in np.linspace(0, self.w - 1, steps)]
        border += [(self.w - 1, y) for y in np.linspace(self.h - 1, 0, steps)]
        border += [(x, 0) for x in np.linspace(self.w - 1, 0, steps)]
        border += [(0, y) for y in np.linspace(0, self.h - 1, steps)]
        for u, v in border:
            g = self.pixel_to_ground(u, v)
            if g is None:  # clip the ray at max range in its horizontal direction
                ray = self.R @ np.array([(u - self.w / 2) / self.f, (v - self.h / 2) / self.f, 1.0])
                hz = math.hypot(ray[0], ray[1]) or 1.0
                g = (ray[0] / hz * self.max_range, ray[1] / hz * self.max_range, self.max_range)
            pts.append((g[0], g[1]))
        return pts


def ground_to_latlon(lat0: float, lon0: float, east_m: float, north_m: float) -> tuple[float, float]:
    dlat = north_m / config.R_EARTH_M
    dlon = east_m / (config.R_EARTH_M * math.cos(math.radians(lat0)))
    return lat0 + math.degrees(dlat), lon0 + math.degrees(dlon)


def project_detections(dets: list[dict], proj: GroundProjector, lat0: float, lon0: float) -> list[dict]:
    """Attach ground position (from the box's bottom-centre, the contact point) to each detection."""
    out = []
    for d in dets:
        u, v = (d["x1"] + d["x2"]) / 2, d["y2"]
        g = proj.pixel_to_ground(u, v)
        rec = dict(d)
        if g is None:
            rec.update(lat=None, lon=None, range_m=None, east_m=None, north_m=None)
        else:
            la, lo = ground_to_latlon(lat0, lon0, g[0], g[1])
            rec.update(lat=round(la, 6), lon=round(lo, 6), range_m=round(g[2], 1), east_m=round(g[0], 1), north_m=round(g[1], 1))
        out.append(rec)
    return out


def footprint_latlon(proj: GroundProjector, lat0: float, lon0: float) -> list[list[float]]:
    return [[*ground_to_latlon(lat0, lon0, e, n)][::-1] for e, n in proj.footprint()]  # [lon, lat] for GeoJSON
