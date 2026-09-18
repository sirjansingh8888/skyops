"""Drone route planning with airspace deconfliction.

Given a start and a destination, plan a low-level corridor that
  * never enters an airport red zone (5 km) or another operator's active geofence  -> hard obstacles
  * avoids airport yellow zones (12 km, ATC permission) and low-level manned traffic -> soft penalties
and compare it with the straight line. The planner is an 8-connected A* over a local east/north grid
followed by line-of-sight smoothing; at drone-delivery scales (5-60 km) it runs in milliseconds.

The result explains itself: which hazards the direct line crosses, how long the detour is, how much of
the path needs permission, the closest low-level aircraft, flight time against battery endurance, and a
GO / CAUTION / NO-GO verdict.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from skyops import config, utm
from skyops.airspace.airports import AIRPORTS
from skyops.airspace.loader import Airspace, get_airspace
from skyops.airspace.predict import predict_at
from skyops.mission import LOW_TRAFFIC_ALT_M, RED_ZONE_KM, YELLOW_ZONE_KM

YELLOW_COST = 2.5     # cost multiplier inside a yellow zone
TRAFFIC_COST = 3.0    # cost multiplier near low-level manned traffic
TRAFFIC_RADIUS_KM = 3.0
GEOFENCE_BUFFER_KM = 0.5


class _Frame:
    """Local tangent-plane frame (km east / km north) around a reference point."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = math.cos(math.radians(lat0)) * config.R_EARTH_M / 1000.0 * math.pi / 180.0
        self.ky = config.R_EARTH_M / 1000.0 * math.pi / 180.0

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return self.lat0 + y / self.ky, self.lon0 + x / self.kx


def _collect_hazards(frame: _Frame, airspace: Airspace, t_idx: int, exclude_mission: str | None) -> dict:
    red, yellow, fences, traffic = [], [], [], []
    for ap in AIRPORTS:
        x, y = frame.to_xy(ap["lat"], ap["lon"])
        red.append(dict(x=x, y=y, r=RED_ZONE_KM, name=ap["name"], icao=ap["icao"]))
        yellow.append(dict(x=x, y=y, r=YELLOW_ZONE_KM, name=ap["name"], icao=ap["icao"]))
    for m in utm.list_missions():
        if m["status"] == "rejected" or m["id"] == exclude_mission:
            continue
        x, y = frame.to_xy(m["lat"], m["lon"])
        fences.append(dict(x=x, y=y, r=m["radius_km"] + GEOFENCE_BUFFER_KM, name=m["name"], id=m["id"]))
    snap = airspace.snapshot(t_idx)
    low = snap[(~snap["on_ground"]) & (snap["altitude_m"] < LOW_TRAFFIC_ALT_M)]
    for r in low.itertuples():
        x, y = frame.to_xy(float(r.latitude), float(r.longitude))
        traffic.append(dict(x=x, y=y, r=TRAFFIC_RADIUS_KM, callsign=r.callsign, horizon_s=0, altitude_ft=round(float(r.altitude_ft))))
    pred = predict_at(airspace, t_idx)
    pred = pred[(pred["altitude_m"] < LOW_TRAFFIC_ALT_M) & (pred["icao24"].isin(set(snap.loc[~snap["on_ground"], "icao24"])))]
    for r in pred.itertuples():
        x, y = frame.to_xy(float(r.latitude), float(r.longitude))
        traffic.append(dict(x=x, y=y, r=TRAFFIC_RADIUS_KM, callsign=airspace.callsign(r.icao24), horizon_s=int(r.horizon_s),
                            altitude_ft=round(float(r.altitude_m) / config.FT_TO_M)))
    return dict(red=red, yellow=yellow, fences=fences, traffic=traffic)


def _inside(circles: list[dict], x: float, y: float) -> dict | None:
    for c in circles:
        if (x - c["x"]) ** 2 + (y - c["y"]) ** 2 <= c["r"] ** 2:
            return c
    return None


def _sample_line(p: tuple[float, float], q: tuple[float, float], step_km: float) -> np.ndarray:
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / step_km) + 1)
    return np.stack([np.linspace(p[0], q[0], n), np.linspace(p[1], q[1], n)], axis=1)


def _profile(points: np.ndarray, hz: dict) -> dict:
    """Walk a polyline (km frame) and summarise what it crosses."""
    seg = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    length = float(seg.sum())
    red_hits, fence_hits, yellow_names, yellow_km = {}, {}, {}, 0.0
    closest_traffic = None
    for i, (x, y) in enumerate(points):
        w = float(seg[min(i, len(seg) - 1)]) if len(seg) else 0.0
        c = _inside(hz["red"], x, y)
        if c:
            red_hits[c["icao"]] = c["name"]
        f = _inside(hz["fences"], x, y)
        if f:
            fence_hits[f["id"]] = f["name"]
        yz = _inside(hz["yellow"], x, y)
        if yz:
            yellow_km += w
            yellow_names[yz["icao"]] = yz["name"]
        for tr in hz["traffic"]:
            d = math.hypot(x - tr["x"], y - tr["y"])
            if closest_traffic is None or d < closest_traffic["dist_km"]:
                closest_traffic = dict(dist_km=round(d, 2), callsign=tr["callsign"], horizon_s=tr["horizon_s"], altitude_ft=tr["altitude_ft"])
    return dict(length_km=round(length, 2), red_zones=list(red_hits.values()), geofences=list(fence_hits.values()),
                yellow_zones=list(yellow_names.values()), yellow_km=round(min(yellow_km, length), 2), closest_low_traffic=closest_traffic)


def _astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    ny, nx = cost.shape
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    g = np.full(cost.shape, np.inf)
    g[start] = 0.0
    came: dict[tuple[int, int], tuple[int, int]] = {}
    heap = [(math.hypot(goal[0] - start[0], goal[1] - start[1]), 0.0, start)]
    closed = np.zeros(cost.shape, bool)
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        closed[cur] = True
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for dy, dx, d in moves:
            ny_, nx_ = cur[0] + dy, cur[1] + dx
            if not (0 <= ny_ < ny and 0 <= nx_ < nx) or closed[ny_, nx_] or not np.isfinite(cost[ny_, nx_]):
                continue
            if dy and dx and not (np.isfinite(cost[cur[0] + dy, cur[1]]) and np.isfinite(cost[cur[0], cur[1] + dx])):
                continue  # no corner cutting past an obstacle
            ng = gc + d * 0.5 * (cost[cur] + cost[ny_, nx_])
            if ng < g[ny_, nx_]:
                g[ny_, nx_] = ng
                came[(ny_, nx_)] = cur
                heapq.heappush(heap, (ng + math.hypot(goal[0] - ny_, goal[1] - nx_), ng, (ny_, nx_)))
    return None


def plan_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float, alt_m: float = 100.0, speed_ms: float = 15.0,
               endurance_min: float = 30.0, t_idx: int | None = None, airspace: Airspace | None = None,
               exclude_mission: str | None = None) -> dict:
    A = airspace or get_airspace()
    t_idx = A.clip(t_idx)
    frame = _Frame((start_lat + end_lat) / 2, (start_lon + end_lon) / 2)
    s, e = frame.to_xy(start_lat, start_lon), frame.to_xy(end_lat, end_lon)
    direct_km = math.hypot(e[0] - s[0], e[1] - s[1])
    hz = _collect_hazards(frame, A, t_idx, exclude_mission)
    step = float(np.clip(direct_km / 120.0, 0.1, 0.5))
    direct = _profile(_sample_line(s, e, step), hz)
    reasons: list[dict] = []

    def result(verdict: str, path_xy: np.ndarray | None, planned: dict | None) -> dict:
        order = {"critical": 0, "alert": 1, "warning": 2, "info": 3}
        reasons.sort(key=lambda r: order[r["level"]])
        line = lambda pts: [[round(frame.to_latlon(x, y)[1], 6), round(frame.to_latlon(x, y)[0], 6)] for x, y in pts]  # noqa: E731
        return dict(verdict=verdict, start=dict(lat=start_lat, lon=start_lon), end=dict(lat=end_lat, lon=end_lon), alt_m=alt_m,
                    speed_ms=speed_ms, t_idx=t_idx, direct=dict(**direct, path=line([s, e])),
                    planned=(dict(**planned, path=line(path_xy)) if planned is not None else None), reasons=reasons)

    if alt_m > 120:
        reasons.append(dict(level="warning", text=f"Cruise altitude {alt_m:.0f} m is above the 120 m routine ceiling"))
    for label, p in (("Start", s), ("Destination", e)):
        c = _inside(hz["red"], *p)
        if c:
            reasons.append(dict(level="critical", text=f"{label} lies inside the {RED_ZONE_KM:.0f} km red zone of {c['name']}: no route can be approved"))
        f = _inside(hz["fences"], *p)
        if f:
            reasons.append(dict(level="critical", text=f"{label} lies inside the active geofence '{f['name']}'"))
    if any(r["level"] == "critical" for r in reasons):
        return result("NO-GO", None, None)

    # --- cost grid -----------------------------------------------------------------------
    margin = max(8.0, 0.35 * direct_km)
    xs, ys = [s[0], e[0]], [s[1], e[1]]
    x0, x1, y0, y1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    for c in hz["red"] + hz["fences"]:  # make room to go around any obstacle that touches the box
        if x0 - c["r"] < c["x"] < x1 + c["r"] and y0 - c["r"] < c["y"] < y1 + c["r"]:
            x0, x1, y0, y1 = min(x0, c["x"] - c["r"] - 3), max(x1, c["x"] + c["r"] + 3), min(y0, c["y"] - c["r"] - 3), max(y1, c["y"] + c["r"] + 3)
    cell = float(np.clip(direct_km / 90.0, 0.2, 1.0))
    gx, gy = np.arange(x0, x1 + cell, cell), np.arange(y0, y1 + cell, cell)
    X, Y = np.meshgrid(gx, gy)
    cost = np.ones_like(X)
    for c in hz["yellow"]:
        cost[(X - c["x"]) ** 2 + (Y - c["y"]) ** 2 <= c["r"] ** 2] = YELLOW_COST
    for c in hz["traffic"]:
        m = (X - c["x"]) ** 2 + (Y - c["y"]) ** 2 <= c["r"] ** 2
        cost[m] = np.maximum(cost[m], TRAFFIC_COST)
    pad = cell * 0.75  # inflate hard obstacles so the smoothed path keeps clear of the boundary
    for c in hz["red"] + hz["fences"]:
        cost[(X - c["x"]) ** 2 + (Y - c["y"]) ** 2 <= (c["r"] + pad) ** 2] = np.inf

    def to_cell(p: tuple[float, float]) -> tuple[int, int]:
        return int(np.clip(round((p[1] - y0) / cell), 0, len(gy) - 1)), int(np.clip(round((p[0] - x0) / cell), 0, len(gx) - 1))

    cs, ce = to_cell(s), to_cell(e)
    cost[cs], cost[ce] = min(cost[cs], YELLOW_COST) if np.isfinite(cost[cs]) else 1.0, min(cost[ce], YELLOW_COST) if np.isfinite(cost[ce]) else 1.0
    cells = _astar(cost, cs, ce)
    if cells is None:
        reasons.append(dict(level="critical", text="No corridor exists between the two points without entering a red zone or an active geofence"))
        return result("NO-GO", None, None)

    pts = [s] + [(float(gx[c]), float(gy[r])) for r, c in cells[1:-1]] + [e]

    def clear(p, q) -> bool:  # line of sight that stays out of hard obstacles and does not raise the soft cost class
        worst = max(cost[to_cell(p)], cost[to_cell(q)])
        for x, y in _sample_line(p, q, cell * 0.5):
            v = cost[to_cell((x, y))]
            if not np.isfinite(v) or v > worst:
                return False
        return True

    smooth, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not clear(pts[i], pts[j]):
            j -= 1
        smooth.append(pts[j])
        i = j
    dense = np.concatenate([_sample_line(smooth[k], smooth[k + 1], step) for k in range(len(smooth) - 1)])
    planned = _profile(dense, hz)
    eta_min = planned["length_km"] * 1000.0 / max(speed_ms, 0.1) / 60.0
    planned.update(waypoints=len(smooth), detour_pct=round(100.0 * (planned["length_km"] / max(direct_km, 1e-6) - 1.0), 1),
                   eta_min=round(eta_min, 1), battery_pct=round(100.0 * eta_min / max(endurance_min, 0.1)), cell_km=round(cell, 2))

    # --- explain -------------------------------------------------------------------------
    if direct["red_zones"]:
        reasons.append(dict(level="alert", text=f"The direct line crosses the red zone of {', '.join(direct['red_zones'])}; the planned corridor goes around it (+{planned['detour_pct']} % distance)"))
    if direct["geofences"]:
        reasons.append(dict(level="alert", text=f"The direct line crosses the active geofence(s) {', '.join(direct['geofences'])}; the planned corridor avoids them"))
    if planned["yellow_km"] > 0.05:
        reasons.append(dict(level="warning", text=f"{planned['yellow_km']:.1f} km of the corridor is inside the yellow zone of {', '.join(planned['yellow_zones'])}: ATC permission required"))
    ct = planned["closest_low_traffic"]
    if ct and ct["dist_km"] < TRAFFIC_RADIUS_KM:
        when = "now" if ct["horizon_s"] == 0 else f"in {ct['horizon_s']} s"
        reasons.append(dict(level="warning", text=f"Low-level traffic {ct['callsign']} ({ct['altitude_ft']} ft) comes within {ct['dist_km']} km of the corridor {when}"))
    if planned["battery_pct"] > 100:
        reasons.append(dict(level="critical", text=f"Flight time {planned['eta_min']:.0f} min exceeds the {endurance_min:.0f} min endurance"))
    elif planned["battery_pct"] > 80:
        reasons.append(dict(level="warning", text=f"Flight time {planned['eta_min']:.0f} min uses {planned['battery_pct']} % of the {endurance_min:.0f} min endurance"))
    if not reasons:
        reasons.append(dict(level="info", text="Direct corridor is clear: green zone, no geofences, no low-level traffic nearby"))
    levels = {r["level"] for r in reasons}
    verdict = "NO-GO" if "critical" in levels else "CAUTION" if levels & {"alert", "warning"} else "GO"
    return result(verdict, np.array(smooth), planned)
